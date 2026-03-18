import os
from pathlib import Path

CONFIG_DIR = Path.home() / "tabpfn"
ENV_FILE = CONFIG_DIR / ".env"

DEFAULT_SERVER_HOST = "0.0.0.0"
DEFAULT_SERVER_PORT = 8197

ENV_API_KEY = "TABPFN_API_KEY"
ENV_SERVER_URL = "TABPFN_SERVER_URL"

API_KEY_HEADER = "X-API-Key"

PREDICT_ENDPOINT = "/predict"
STATUS_ENDPOINT = "/status"
UNLOAD_ENDPOINT = "/unload"


def get_api_key():
  key = os.environ.get(ENV_API_KEY)
  if key:
    return key
  if ENV_FILE.exists():
    for line in ENV_FILE.read_text().strip().splitlines():
      line = line.strip()
      if line.startswith(f"{ENV_API_KEY}="):
        return line.split("=", 1)[1].strip().strip("\"'")
  return None


def get_server_url():
  url = os.environ.get(ENV_SERVER_URL)
  if url:
    return url.rstrip("/")
  if ENV_FILE.exists():
    for line in ENV_FILE.read_text().strip().splitlines():
      line = line.strip()
      if line.startswith(f"{ENV_SERVER_URL}="):
        return line.split("=", 1)[1].strip().strip("\"'").rstrip("/")
  return None
