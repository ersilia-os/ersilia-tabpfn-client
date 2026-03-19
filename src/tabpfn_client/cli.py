import argparse

from tabpfn_client.errors import TabPFNClientError
from tabpfn_client.logger import logger


def cmd_status(args):
  from tabpfn_client.client import check_status
  from tabpfn_client.constants import get_server_url

  url = get_server_url()
  logger.info(f"connecting to {url}")
  info = check_status()
  logger.success("server is reachable")
  logger.info(f"server url    : {url}")
  logger.info(f"gpu available : {info.get('gpu_available')}")
  logger.info(f"gpu name      : {info.get('gpu_name', 'n/a')}")
  logger.info(f"models loaded : {info.get('models_loaded', [])}")
  idle = info.get("idle_seconds")
  if idle is not None:
    logger.info(f"idle          : {idle}s")


def cmd_predict(args):
  import numpy as np

  from tabpfn_client.client import predict
  from tabpfn_client.io import read_input, write_output

  logger.info(f"reading input from {args.input}")
  X, y = read_input(args.input)
  n_train = int(np.sum(~np.isnan(y)))
  n_test = int(np.sum(np.isnan(y)))
  logger.info(f"samples: {X.shape[0]} (train={n_train}, test={n_test}), features: {X.shape[1]}")

  task = args.task
  config = {}
  if args.n_estimators:
    config["n_estimators"] = args.n_estimators

  chunk_size = args.chunk_size

  logger.info(f"sending predict request (task={task})")
  result = predict(X, y, task=task, config=config, chunk_size=chunk_size)
  predictions = np.asarray(result["predictions"])
  probabilities = None
  if "probabilities" in result:
    probabilities = np.asarray(result["probabilities"])

  write_output(args.output, predictions, probabilities)
  logger.success(f"output written to {args.output}")


def cmd_configure(args):
  from tabpfn_client.constants import CONFIG_DIR, ENV_FILE, ENV_API_KEY, ENV_SERVER_URL

  CONFIG_DIR.mkdir(parents=True, exist_ok=True)
  lines = {}
  if ENV_FILE.exists():
    for line in ENV_FILE.read_text().strip().splitlines():
      line = line.strip()
      if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        lines[k.strip()] = v.strip()
  if args.secret:
    lines[ENV_API_KEY] = args.secret
  if args.url:
    lines[ENV_SERVER_URL] = args.url
  content = "\n".join(f"{k}={v}" for k, v in lines.items()) + "\n"
  ENV_FILE.write_text(content)
  logger.success(f"config written to {ENV_FILE}")
  if ENV_SERVER_URL in lines:
    logger.info(f"server url : {lines[ENV_SERVER_URL]}")
  if ENV_API_KEY in lines:
    logger.info(f"api key    : {lines[ENV_API_KEY][:8]}...")


def cmd_unload(args):
  from tabpfn_client.client import unload_models

  result = unload_models()
  unloaded = result.get("unloaded", [])
  if unloaded:
    logger.success(f"unloaded models: {unloaded}")
  else:
    logger.info("no models were loaded")


def cmd_serve(args):
  import os

  from tabpfn_client.server.run import run_server

  if args.model_version:
    os.environ["TABPFN_MODEL_VERSION"] = args.model_version
  if args.idle_timeout is not None:
    os.environ["TABPFN_IDLE_TIMEOUT"] = str(args.idle_timeout)
  run_server(host=args.host, port=args.port, api_key=args.api_key)


def build_parser():
  p = argparse.ArgumentParser(prog="tabpfn", description="TabPFN remote inference client")
  sub = p.add_subparsers(dest="cmd", required=True)

  sub.add_parser("status", help="check server status")

  sub.add_parser("unload", help="unload models from server gpu memory")

  p_pred = sub.add_parser("predict", help="run prediction via remote server")
  p_pred.add_argument("-i", "--input", required=True, help="input file (.csv or .h5)")
  p_pred.add_argument("-o", "--output", required=True, help="output file (.csv or .h5)")
  p_pred.add_argument(
    "-t", "--task", default="classification", choices=["classification", "regression"]
  )
  p_pred.add_argument("-n", "--n-estimators", type=int, default=None)
  p_pred.add_argument(
    "-c",
    "--chunk-size",
    type=int,
    default=None,
    help="split test rows into chunks of this size (reuses training context per chunk)",
  )

  p_cfg = sub.add_parser("configure", help="configure api key and server url")
  p_cfg.add_argument("--secret", default=None, help="api key for authentication")
  p_cfg.add_argument("--url", default=None, help="server url (e.g. http://gpu-host:8197)")

  p_srv = sub.add_parser("serve", help="start the tabpfn inference server")
  p_srv.add_argument("--host", default=None)
  p_srv.add_argument("--port", type=int, default=None)
  p_srv.add_argument("--api-key", default=None)
  p_srv.add_argument(
    "--model-version", default=None, choices=["v2", "v2.5"], help="tabpfn model version"
  )
  p_srv.add_argument(
    "--idle-timeout",
    type=int,
    default=None,
    help="auto-unload models after N seconds of inactivity (0=disabled)",
  )

  return p


_DISPATCH = {
  "status": cmd_status,
  "unload": cmd_unload,
  "predict": cmd_predict,
  "configure": cmd_configure,
  "serve": cmd_serve,
}


def main():
  parser = build_parser()
  args = parser.parse_args()
  try:
    _DISPATCH[args.cmd](args)
  except TabPFNClientError as e:
    logger.error(str(e))
    raise SystemExit(2)
  except KeyboardInterrupt:
    logger.error("interrupted")
    raise SystemExit(130)


if __name__ == "__main__":
  main()
