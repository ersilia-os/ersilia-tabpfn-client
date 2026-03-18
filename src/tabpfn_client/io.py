import numpy as np
import pandas as pd
import h5py

from tabpfn_client.errors import SerializationError


def read_input(path):
  path = str(path)
  if path.endswith(".csv"):
    return _read_csv(path)
  elif path.endswith(".h5") or path.endswith(".hdf5"):
    return _read_h5(path)
  raise SerializationError(f"unsupported input format: {path}")


def write_output(path, predictions, probabilities=None):
  path = str(path)
  if path.endswith(".csv"):
    return _write_csv(path, predictions, probabilities)
  elif path.endswith(".h5") or path.endswith(".hdf5"):
    return _write_h5(path, predictions, probabilities)
  raise SerializationError(f"unsupported output format: {path}")


def _read_csv(path):
  df = pd.read_csv(path)
  if "y" not in df.columns:
    raise SerializationError("csv must have a 'y' column")
  y = df["y"].values.astype(np.float64)
  X = df.drop(columns=["y"]).values.astype(np.float64)
  return X, y


def _read_h5(path):
  with h5py.File(path, "r") as f:
    if "X" not in f:
      raise SerializationError("h5 file must contain dataset 'X'")
    if "y" not in f:
      raise SerializationError("h5 file must contain dataset 'y'")
    X = f["X"][:]
    y = f["y"][:]
  return X.astype(np.float64), y.astype(np.float64)


def _write_csv(path, predictions, probabilities=None):
  data = {"predictions": predictions}
  if probabilities is not None:
    for i in range(probabilities.shape[1]):
      data[f"prob_{i}"] = probabilities[:, i]
  pd.DataFrame(data).to_csv(path, index=False)


def _write_h5(path, predictions, probabilities=None):
  with h5py.File(path, "w") as f:
    f.create_dataset("predictions", data=predictions)
    if probabilities is not None:
      f.create_dataset("probabilities", data=probabilities)
