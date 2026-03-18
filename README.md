# ersilia-tabpfn-client

Remote TabPFN inference client and server for running predictions on a GPU machine.

It supports:
- serving TabPFN on a remote machine
- sending `X` and partially empty `y` from a client machine
- API key authentication
- CSV and HDF5 input/output
- binary request transport with `msgpack`

## Install

Create and activate an environment:

```bash
conda create -n pmnet python=3.11
conda activate pmnet
```

Install the package:

```bash
pip install -e .
```

If you want to run the server too:

```bash
pip install -e ".[server]"
```

## TabPFN model notes

- default server model version is `v2`
- `v2` works without gated Hugging Face access
- `v2.5` is the latest checkpoint, but requires accepting the model terms and authenticating with Hugging Face

To use `v2.5`:

```bash
hf auth login
```

Then accept the model terms at `https://huggingface.co/Prior-Labs/tabpfn_2_5`.

## Configuration

Client config is stored in `~/tabpfn/.env`.

Write the API key and server URL:

```bash
tabpfn configure --secret my-api-key --url http://my-gpu-host:8197
```

You can also use environment variables instead:

```bash
export TABPFN_API_KEY=my-api-key
export TABPFN_SERVER_URL=http://my-gpu-host:8197
```

The client checks env vars first, then `~/tabpfn/.env`.

## Start the server

On the remote GPU machine:

```bash
tabpfn serve --host 0.0.0.0 --port 8197 --api-key my-api-key
```

Use the latest gated checkpoint instead:

```bash
TABPFN_MODEL_VERSION=v2.5 tabpfn serve --host 0.0.0.0 --port 8197 --api-key my-api-key
```

Or explicitly from the CLI:

```bash
tabpfn serve --host 0.0.0.0 --port 8197 --api-key my-api-key --model-version v2.5
```

## Check server status

```bash
tabpfn status
```

This reports:
- whether the server is reachable
- whether CUDA is available
- GPU name
- which models are already loaded in memory

## Unload models

Models stay in GPU memory after the first prediction. You can free that memory manually or automatically.

### Manual unload

```bash
tabpfn unload
```

This tells the server to remove all loaded models from GPU memory and run `torch.cuda.empty_cache()`.

### Auto-unload on idle

Start the server with `--idle-timeout` to automatically unload models after a period of inactivity:

```bash
tabpfn serve --host 0.0.0.0 --port 8197 --api-key my-api-key --idle-timeout 600
```

This unloads all models if no prediction request arrives for 600 seconds (10 minutes). The next prediction request will reload the model automatically.

You can also set it via environment variable:

```bash
TABPFN_IDLE_TIMEOUT=600 tabpfn serve --host 0.0.0.0 --port 8197 --api-key my-api-key
```

Set to `0` or omit to disable auto-unload.

## Input validation

Both client and server validate inputs before sending or processing:
- max rows: 50,000
- max columns: 10,000
- `X` must be 2d, `y` must be 1d
- `X` and `y` must have the same number of rows
- at least one training row (non-`NaN` y) and one test row (`NaN` y) required

Validation errors are raised as `tabpfn_client.errors.ValidationError` on the client and returned as HTTP 400 on the server.

## Chunked prediction

When you have many test rows (e.g. 5,000) but the server handles them better in smaller batches, use `--chunk-size` to split the test rows into chunks. Each chunk is sent with the full training context, and the results are combined automatically.

```bash
tabpfn predict -i input.csv -o output.csv --chunk-size 500
```

This sends 10 requests of 500 test rows each (all sharing the same training context), then concatenates the predictions into a single output.

Without `--chunk-size`, all test rows are sent in a single request.

## Input format

The tool expects:
- `X`: all rows, all features
- `y`: full label vector with known labels for context rows
- unknown target rows marked as `NaN`

Rows with non-`NaN` `y` are used as training context.
Rows with `NaN` `y` are predicted by TabPFN.

### CSV input

CSV must contain:
- feature columns
- one `y` column

Example:

```csv
f0,f1,f2,y
0.1,1.2,3.4,1
0.2,1.5,3.8,0
0.4,1.1,3.2,
0.7,1.0,2.9,
```

Empty `y` cells become `NaN` and are treated as rows to predict.

### HDF5 input

HDF5 must contain two datasets:
- `X`
- `y`

Example:

```python
import h5py
import numpy as np

X = np.random.rand(4, 3)
y = np.array([1, 0, np.nan, np.nan])

with h5py.File("input.h5", "w") as f:
  f.create_dataset("X", data=X)
  f.create_dataset("y", data=y)
```

## Run prediction

### End-to-end with repository example data

The repository already includes example files in `data/`:
- `data/test_input.csv`
- `data/output.csv`
- `data/output.h5`

`data/test_input.csv` contains:
- 300 rows
- 30 feature columns
- 1 `y` column
- 200 labeled context rows
- 100 unlabeled rows to predict

Start the server:

```bash
tabpfn serve --host 0.0.0.0 --port 8197 --api-key my-api-key --model-version v2
```

On the client machine, configure access:

```bash
tabpfn configure --secret my-api-key --url http://my-gpu-host:8197
```

Check connectivity:

```bash
tabpfn status
```

Run prediction with the bundled example input:

```bash
tabpfn predict -i data/test_input.csv -o data/output.csv --task classification
```

Write HDF5 output instead:

```bash
tabpfn predict -i data/test_input.csv -o data/output.h5 --task classification
```

The expected output shapes for this example are:
- `data/output.csv`: 100 rows, columns `predictions`, `prob_0`, `prob_1`
- `data/output.h5`: datasets `predictions` with shape `(100,)` and `probabilities` with shape `(100, 2)`

Preview the CSV result:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv('data/output.csv')
print(df.head())
PY
```

CSV to CSV:

```bash
tabpfn predict -i input.csv -o output.csv
```

CSV to HDF5:

```bash
tabpfn predict -i input.csv -o output.h5
```

HDF5 to CSV:

```bash
tabpfn predict -i input.h5 -o output.csv
```

Regression mode:

```bash
tabpfn predict -i input.csv -o output.csv --task regression
```

Set ensemble size:

```bash
tabpfn predict -i input.csv -o output.csv --n-estimators 16
```

## Output format

### Classification output

CSV output contains:
- `predictions`
- `prob_0`, `prob_1`, ...

Example:

```csv
predictions,prob_0,prob_1
1,0.08,0.92
0,0.99,0.01
```

HDF5 output contains:
- `predictions`
- `probabilities`

### Regression output

CSV output contains:
- `predictions`

HDF5 output contains:
- `predictions`

## Python API

All functionality is available as a Python API for integration into other tools.

### Setup

```python
import tabpfn_client

tabpfn_client.configure(secret="my-api-key", url="http://my-gpu-host:8197")
```

### Check server status

```python
info = tabpfn_client.status()
print(info["gpu_available"])
print(info["models_loaded"])
```

### Predict from arrays

```python
import numpy as np

X = np.random.rand(300, 30)
y = np.concatenate([np.array([0, 1] * 100, dtype=np.float64), np.full(100, np.nan)])

predictions, probabilities = tabpfn_client.predict(X, y, task="classification")
```

### Predict with chunking

```python
predictions, probabilities = tabpfn_client.predict(
  X, y, task="classification", chunk_size=500
)
```

### Predict from file

```python
predictions, probabilities = tabpfn_client.predict_from_file(
  "data/test_input.csv",
  output_path="data/output.csv",
  task="classification",
)
```

### Unload models

```python
tabpfn_client.unload()
```

### Full example

```python
import numpy as np
import tabpfn_client
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

tabpfn_client.configure(secret="my-api-key", url="http://my-gpu-host:8197")

X, y_true = load_breast_cancer(return_X_y=True)
y = y_true.copy().astype(np.float64)
y[200:] = np.nan

predictions, probabilities = tabpfn_client.predict(X, y, task="classification")

print(f"accuracy: {accuracy_score(y_true[200:], predictions):.4f}")
print(f"probabilities shape: {probabilities.shape}")

tabpfn_client.unload()
```

## Typical remote workflow

On the GPU server:

```bash
git clone https://github.com/ersilia-os/ersilia-tabpfn-client.git
cd ersilia-tabpfn-client
conda activate pmnet
pip install -e .[server]
tabpfn serve --host 0.0.0.0 --port 8197 --api-key my-api-key --model-version v2
```

On the client machine:

```bash
git clone https://github.com/ersilia-os/ersilia-tabpfn-client.git
cd ersilia-tabpfn-client
pip install -e .
tabpfn configure --secret my-api-key --url http://my-gpu-host:8197
tabpfn status
tabpfn predict -i input.csv -o output.csv
```

## Notes

- transport uses `msgpack`, not JSON
- the server loads models lazily on first prediction
- API key auth is optional, but recommended
- `tabpfn status` does not require model download
- `tabpfn predict` triggers model loading if needed
- `tabpfn unload` frees GPU memory without restarting the server
- `--idle-timeout` auto-unloads after inactivity; the watchdog checks every 30 seconds
- input is validated client-side before sending (max 50k rows, 10k columns)
- `--chunk-size` splits large test sets into batches sharing the same training context
- all CLI functionality is also available via `import tabpfn_client`

## Development

Lint and test:

```bash
ruff check src/
ruff format --check src/
pytest tests/ -v
```

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit organization fueling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for antimicrobial drug discovery.

![Ersilia Logo](assets/Ersilia_Brand.png)
