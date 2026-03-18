import numpy as np
import httpx

from tabpfn_client.codec import encode_request, decode_response
from tabpfn_client.constants import (
  API_KEY_HEADER,
  PREDICT_ENDPOINT,
  STATUS_ENDPOINT,
  UNLOAD_ENDPOINT,
  get_api_key,
  get_server_url,
)
from tabpfn_client.errors import AuthError, ConfigError, ServerError
from tabpfn_client.logger import logger
from tabpfn_client.validate import validate_input

CONTENT_TYPE = "application/x-msgpack"
TIMEOUT = 300.0


def _build_headers():
  key = get_api_key()
  headers = {"Content-Type": CONTENT_TYPE}
  if key:
    headers[API_KEY_HEADER] = key
  return headers


def _base_url():
  url = get_server_url()
  if not url:
    raise ConfigError(
      "server url not configured. set TABPFN_SERVER_URL or run: tabpfn configure --url <URL>"
    )
  return url


def _post_predict(url, headers, body):
  try:
    r = httpx.post(
      f"{url}{PREDICT_ENDPOINT}",
      content=body,
      headers=headers,
      timeout=TIMEOUT,
    )
  except httpx.ConnectError as e:
    raise ServerError(f"cannot connect to {url}: {e}")
  if r.status_code == 401:
    raise AuthError("authentication failed, check your api key")
  if r.status_code != 200:
    raise ServerError(f"server returned {r.status_code}: {r.text}")
  return decode_response(r.content)


def check_status():
  url = _base_url()
  headers = _build_headers()
  try:
    r = httpx.get(f"{url}{STATUS_ENDPOINT}", headers=headers, timeout=10.0)
  except httpx.ConnectError as e:
    raise ServerError(f"cannot connect to {url}: {e}")
  if r.status_code == 401:
    raise AuthError("authentication failed, check your api key")
  if r.status_code != 200:
    raise ServerError(f"server returned {r.status_code}: {r.text}")
  return r.json()


def predict(X, y, task="classification", config=None, chunk_size=None):
  X, y = validate_input(X, y)
  url = _base_url()
  headers = _build_headers()

  train_mask = ~np.isnan(y)
  X_train = X[train_mask]
  y_train = y[train_mask]
  test_indices = np.where(np.isnan(y))[0]
  X_test = X[test_indices]
  n_test = X_test.shape[0]

  if chunk_size is None or chunk_size <= 0 or n_test <= chunk_size:
    body = encode_request(X, y, task, config)
    return _post_predict(url, headers, body)

  all_preds = []
  all_probs = []
  n_chunks = int(np.ceil(n_test / chunk_size))
  logger.info(f"splitting {n_test} test rows into {n_chunks} chunks of {chunk_size}")

  for i in range(n_chunks):
    start = i * chunk_size
    end = min(start + chunk_size, n_test)
    X_chunk = X_test[start:end]

    X_combined = np.vstack([X_train, X_chunk])
    y_combined = np.concatenate([y_train, np.full(X_chunk.shape[0], np.nan)])

    body = encode_request(X_combined, y_combined, task, config)
    logger.info(f"chunk {i + 1}/{n_chunks}: {X_chunk.shape[0]} test rows")
    result = _post_predict(url, headers, body)
    all_preds.append(np.asarray(result["predictions"]))
    if "probabilities" in result:
      all_probs.append(np.asarray(result["probabilities"]))

  merged = {"predictions": np.concatenate(all_preds)}
  if all_probs:
    merged["probabilities"] = np.vstack(all_probs)
  return merged


def unload_models():
  url = _base_url()
  headers = _build_headers()
  try:
    r = httpx.post(f"{url}{UNLOAD_ENDPOINT}", headers=headers, timeout=10.0)
  except httpx.ConnectError as e:
    raise ServerError(f"cannot connect to {url}: {e}")
  if r.status_code == 401:
    raise AuthError("authentication failed, check your api key")
  if r.status_code != 200:
    raise ServerError(f"server returned {r.status_code}: {r.text}")
  return r.json()
