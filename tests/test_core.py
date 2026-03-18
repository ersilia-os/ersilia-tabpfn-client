from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tabpfn_client.codec import decode_request, decode_response, encode_request, encode_response
from tabpfn_client.io import read_input, write_output
from tabpfn_client.validate import validate_input, MAX_ROWS, MAX_COLUMNS
from tabpfn_client.errors import ValidationError


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


# --- codec ---


def test_codec_roundtrip():
  X = np.random.rand(10, 5)
  y = np.random.rand(10)
  raw = encode_request(X, y, "classification")
  data = decode_request(raw)
  np.testing.assert_array_almost_equal(data["X"], X)
  np.testing.assert_array_almost_equal(data["y"], y)
  assert data["task"] == "classification"


def test_response_roundtrip():
  preds = np.random.rand(5)
  probs = np.random.rand(5, 2)
  raw = encode_response(preds, probs)
  data = decode_response(raw)
  np.testing.assert_array_almost_equal(data["predictions"], preds)
  np.testing.assert_array_almost_equal(data["probabilities"], probs)


# --- io with repo data ---


def test_csv_io_uses_repo_example_data():
  X, y = read_input(DATA_DIR / "test_input.csv")
  assert X.shape == (300, 30)
  assert y.shape == (300,)
  assert np.sum(~np.isnan(y)) == 200
  assert np.sum(np.isnan(y)) == 100


def test_repo_example_output_csv_is_well_formed():
  df = pd.read_csv(DATA_DIR / "output.csv")
  assert list(df.columns) == ["predictions", "prob_0", "prob_1"]
  assert df.shape == (100, 3)
  assert set(df["predictions"].unique()).issubset({0.0, 1.0})


def test_write_csv(tmp_path):
  preds = np.array([0, 1, 0])
  probs = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
  out = tmp_path / "out.csv"
  write_output(out, preds, probs)
  df = pd.read_csv(out)
  assert "predictions" in df.columns
  assert "prob_0" in df.columns
  assert len(df) == 3


def test_write_h5(tmp_path):
  import h5py

  preds = np.array([1.0, 2.0, 3.0])
  out = tmp_path / "out.h5"
  write_output(out, preds)
  with h5py.File(out, "r") as f:
    saved = np.array(f.get("predictions"))
    np.testing.assert_array_equal(saved, preds)


def test_repo_example_output_h5_is_well_formed():
  import h5py

  with h5py.File(DATA_DIR / "output.h5", "r") as f:
    assert sorted(f.keys()) == ["predictions", "probabilities"]
    predictions = np.array(f.get("predictions"))
    probabilities = np.array(f.get("probabilities"))
    assert predictions.shape == (100,)
    assert probabilities.shape == (100, 2)


# --- validation ---


def test_validate_input_good():
  X = np.random.rand(20, 5)
  y = np.concatenate([np.array([0, 1] * 5, dtype=np.float64), np.full(10, np.nan)])
  X_out, y_out = validate_input(X, y)
  assert X_out.shape == (20, 5)
  assert y_out.shape == (20,)


def test_validate_input_wrong_X_dim():
  with pytest.raises(ValidationError, match="2d"):
    validate_input(np.array([1, 2, 3]), np.array([1, 2, 3]))


def test_validate_input_wrong_y_dim():
  with pytest.raises(ValidationError, match="1d"):
    validate_input(np.random.rand(3, 2), np.random.rand(3, 2))


def test_validate_input_row_mismatch():
  with pytest.raises(ValidationError, match="rows"):
    validate_input(np.random.rand(5, 2), np.array([1, 2, np.nan]))


def test_validate_input_too_many_rows():
  X = np.random.rand(MAX_ROWS + 1, 2)
  y = np.concatenate([np.ones(MAX_ROWS), np.full(1, np.nan)])
  with pytest.raises(ValidationError, match="too many rows"):
    validate_input(X, y)


def test_validate_input_too_many_columns():
  X = np.random.rand(10, MAX_COLUMNS + 1)
  y = np.concatenate([np.ones(5), np.full(5, np.nan)])
  with pytest.raises(ValidationError, match="too many columns"):
    validate_input(X, y)


def test_validate_input_no_train():
  X = np.random.rand(5, 2)
  y = np.full(5, np.nan)
  with pytest.raises(ValidationError, match="no training rows"):
    validate_input(X, y)


def test_validate_input_no_test():
  X = np.random.rand(5, 2)
  y = np.ones(5)
  with pytest.raises(ValidationError, match="no test rows"):
    validate_input(X, y)


# --- python api imports ---


def test_api_imports():
  from tabpfn_client import configure, status, predict, predict_from_file, unload

  assert callable(configure)
  assert callable(status)
  assert callable(predict)
  assert callable(predict_from_file)
  assert callable(unload)
