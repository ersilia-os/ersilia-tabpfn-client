import os

from tabpfn_client.constants import DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT, ENV_API_KEY, ENV_FILE


def run_server(host=None, port=None, api_key=None):
  import uvicorn

  host = host or DEFAULT_SERVER_HOST
  port = port or DEFAULT_SERVER_PORT

  if api_key:
    os.environ[ENV_API_KEY] = api_key
  elif ENV_FILE.exists():
    for line in ENV_FILE.read_text().strip().splitlines():
      line = line.strip()
      if line.startswith(f"{ENV_API_KEY}="):
        os.environ[ENV_API_KEY] = line.split("=", 1)[1].strip().strip("\"'")
        break

  uvicorn.run(
    "tabpfn_client.server.app:app",
    host=host,
    port=port,
    log_level="info",
  )
