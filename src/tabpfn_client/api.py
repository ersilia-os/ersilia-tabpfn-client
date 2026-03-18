import os

import numpy as np

from tabpfn_client.client import (
  check_status as _check_status,
  predict as _predict,
  unload_models as _unload_models,
)
from tabpfn_client.constants import CONFIG_DIR, ENV_FILE, ENV_API_KEY, ENV_SERVER_URL
from tabpfn_client.io import read_input, write_output
from tabpfn_client.validate import validate_input


def configure(secret=None, url=None):
  CONFIG_DIR.mkdir(parents=True, exist_ok=True)
  lines = {}
  if ENV_FILE.exists():
    for line in ENV_FILE.read_text().strip().splitlines():
      line = line.strip()
      if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        lines[k.strip()] = v.strip()
  if secret:
    lines[ENV_API_KEY] = secret
    os.environ[ENV_API_KEY] = secret
  if url:
    lines[ENV_SERVER_URL] = url
    os.environ[ENV_SERVER_URL] = url
  if lines:
    content = "\n".join(f"{k}={v}" for k, v in lines.items()) + "\n"
    ENV_FILE.write_text(content)
  return ENV_FILE


def status():
  return _check_status()


def predict(X, y, task="classification", config=None, chunk_size=None):
  X = np.asarray(X, dtype=np.float64)
  y = np.asarray(y, dtype=np.float64)
  validate_input(X, y)
  result = _predict(X, y, task=task, config=config, chunk_size=chunk_size)
  predictions = np.asarray(result["predictions"])
  probabilities = None
  if "probabilities" in result:
    probabilities = np.asarray(result["probabilities"])
  return predictions, probabilities


def predict_from_file(
  input_path, output_path=None, task="classification", config=None, chunk_size=None
):
  X, y = read_input(input_path)
  predictions, probabilities = predict(X, y, task=task, config=config, chunk_size=chunk_size)
  if output_path:
    write_output(output_path, predictions, probabilities)
  return predictions, probabilities


def unload():
  return _unload_models()
