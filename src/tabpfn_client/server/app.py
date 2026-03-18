import asyncio
import gc
import os
import time

import numpy as np
from fastapi import FastAPI, Request, Response, HTTPException

from tabpfn_client.codec import decode_request, encode_response
from tabpfn_client.constants import API_KEY_HEADER, ENV_API_KEY
from tabpfn_client.logger import logger

app = FastAPI(title="tabpfn-server")
_model_cache = {}
_last_activity = time.time()
_idle_task = None

CONTENT_TYPE = "application/x-msgpack"


def _touch():
  global _last_activity
  _last_activity = time.time()


def _get_api_key():
  return os.environ.get(ENV_API_KEY)


def _verify_key(request):
  expected = _get_api_key()
  if not expected:
    return
  provided = request.headers.get(API_KEY_HEADER)
  if provided != expected:
    raise HTTPException(status_code=401, detail="invalid api key")


def _resolve_version():
  from tabpfn.constants import ModelVersion

  v = os.environ.get("TABPFN_MODEL_VERSION", "v2").lower().strip()
  if v in ("v2", "2"):
    return ModelVersion.V2
  return ModelVersion.V2_5


def _get_model(task):
  if task not in _model_cache:
    version = _resolve_version()
    logger.info(f"loading tabpfn model for task={task} version={version}")
    if task == "classification":
      from tabpfn import TabPFNClassifier

      model = TabPFNClassifier.create_default_for_version(version, device="auto", n_estimators=8)
    elif task == "regression":
      from tabpfn import TabPFNRegressor

      model = TabPFNRegressor.create_default_for_version(version, device="auto", n_estimators=8)
    else:
      raise HTTPException(status_code=400, detail=f"unknown task: {task}")
    _model_cache[task] = model
    logger.success(f"model loaded for task={task}")
  return _model_cache[task]


def _unload_models():
  if not _model_cache:
    return []
  unloaded = list(_model_cache.keys())
  _model_cache.clear()
  gc.collect()
  try:
    import torch

    if torch.cuda.is_available():
      torch.cuda.empty_cache()
  except Exception:
    pass
  logger.info(f"unloaded models: {unloaded}")
  return unloaded


async def _idle_watchdog():
  timeout = int(os.environ.get("TABPFN_IDLE_TIMEOUT", "0"))
  if timeout <= 0:
    return
  logger.info(f"idle watchdog started, timeout={timeout}s")
  while True:
    await asyncio.sleep(30)
    if not _model_cache:
      continue
    elapsed = time.time() - _last_activity
    if elapsed >= timeout:
      logger.warning(f"idle for {int(elapsed)}s, unloading models")
      _unload_models()


@app.get("/status")
async def status():
  import torch

  return {
    "status": "ok",
    "gpu_available": torch.cuda.is_available(),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "models_loaded": list(_model_cache.keys()),
    "idle_seconds": int(time.time() - _last_activity),
  }


@app.post("/predict")
async def predict(request: Request):
  _verify_key(request)
  _touch()
  raw = await request.body()
  data = decode_request(raw)

  X = np.asarray(data["X"], dtype=np.float64)
  y = np.asarray(data["y"], dtype=np.float64)
  task = data.get("task", "classification")
  config = data.get("config", {})

  from tabpfn_client.validate import validate_input, ValidationError

  try:
    X, y = validate_input(X, y)
  except ValidationError as e:
    raise HTTPException(status_code=400, detail=str(e))

  train_mask = ~np.isnan(y)
  X_train = X[train_mask]
  y_train = y[train_mask]
  X_test = X[~train_mask]

  model = _get_model(task)
  n_estimators = config.get("n_estimators")
  if n_estimators:
    model.n_estimators = int(n_estimators)

  logger.info(
    f"predict: task={task} train={X_train.shape[0]} test={X_test.shape[0]} features={X.shape[1]}"
  )
  t0 = time.perf_counter()
  model.fit(X_train, y_train)

  predictions = model.predict(X_test)
  probabilities = None
  if task == "classification":
    probabilities = model.predict_proba(X_test)

  elapsed = time.perf_counter() - t0
  logger.success(f"prediction done in {elapsed:.3f}s")

  body = encode_response(predictions, probabilities)
  return Response(content=body, media_type=CONTENT_TYPE)


@app.post("/unload")
async def unload(request: Request):
  _verify_key(request)
  unloaded = _unload_models()
  return {"unloaded": unloaded}


@app.on_event("startup")
async def on_startup():
  global _idle_task
  logger.info("tabpfn server starting")
  preload = os.environ.get("TABPFN_PRELOAD", "").strip()
  if preload:
    for task in preload.split(","):
      task = task.strip()
      if task:
        _get_model(task)
  _touch()
  _idle_task = asyncio.create_task(_idle_watchdog())
