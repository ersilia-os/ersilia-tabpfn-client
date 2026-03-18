import numpy as np

from tabpfn_client.errors import ValidationError

MAX_ROWS = 50_000
MAX_COLUMNS = 10_000


def validate_input(X, y):
  X = np.asarray(X, dtype=np.float64)
  y = np.asarray(y, dtype=np.float64)
  if X.ndim != 2:
    raise ValidationError(f"X must be 2d, got {X.ndim}d")
  if y.ndim != 1:
    raise ValidationError(f"y must be 1d, got {y.ndim}d")
  if X.shape[0] != y.shape[0]:
    raise ValidationError(f"X has {X.shape[0]} rows but y has {y.shape[0]} entries")
  if X.shape[0] > MAX_ROWS:
    raise ValidationError(f"too many rows: {X.shape[0]} (max {MAX_ROWS})")
  if X.shape[1] > MAX_COLUMNS:
    raise ValidationError(f"too many columns: {X.shape[1]} (max {MAX_COLUMNS})")
  n_train = int(np.sum(~np.isnan(y)))
  n_test = int(np.sum(np.isnan(y)))
  if n_train == 0:
    raise ValidationError("no training rows (all y values are nan)")
  if n_test == 0:
    raise ValidationError("no test rows (all y values are non-nan)")
  return X, y
