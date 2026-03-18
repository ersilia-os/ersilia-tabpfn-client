import numpy as np
import msgpack
import msgpack_numpy as mn

mn.patch()


def encode_request(X, y, task, config=None):
  payload = {
    "X": np.asarray(X, dtype=np.float64),
    "y": np.asarray(y, dtype=np.float64),
    "task": task,
  }
  if config:
    payload["config"] = config
  return msgpack.packb(payload, use_bin_type=True)


def decode_request(raw):
  return msgpack.unpackb(raw, raw=False)


def encode_response(predictions, probabilities=None, extra=None):
  payload = {"predictions": np.asarray(predictions, dtype=np.float64)}
  if probabilities is not None:
    payload["probabilities"] = np.asarray(probabilities, dtype=np.float64)
  if extra:
    payload["extra"] = extra
  return msgpack.packb(payload, use_bin_type=True)


def decode_response(raw):
  return msgpack.unpackb(raw, raw=False)
