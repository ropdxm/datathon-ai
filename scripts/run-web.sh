#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ -f "$PROJECT_DIR/.env" ]]; then
  set -a
  # .env is a local, trusted shell configuration; never included in the archive.
  source "$PROJECT_DIR/.env"
  set +a
fi

if [[ ! -x "$PROJECT_DIR/.venv-web/bin/python" ]]; then
  WEB_PYTHON="${PYTHON_BIN:-python3}"
  "$WEB_PYTHON" -c 'import sys; assert sys.version_info >= (3, 11), "Нужен Python 3.11 или новее; задайте PYTHON_BIN"'
  "$WEB_PYTHON" -m venv "$PROJECT_DIR/.venv-web"
fi
if ! "$PROJECT_DIR/.venv-web/bin/python" -c 'import fastapi, uvicorn, multipart, av, PIL, cv2' 2>/dev/null; then
  "$PROJECT_DIR/.venv-web/bin/python" -m pip install -r "$PROJECT_DIR/requirements-web.txt"
fi

export FORTE_DATA_DIR="${FORTE_DATA_DIR:-$PROJECT_DIR/.forte-runtime}"
mkdir -p "$FORTE_DATA_DIR"
if [[ -z "${FORTE_API_KEY:-}" ]]; then
  FORTE_API_KEY="$("$PROJECT_DIR/.venv-web/bin/python" - <<'PY'
from pathlib import Path
import os
import secrets

path = Path(os.environ['FORTE_DATA_DIR']) / 'jury-api-key'
if not path.exists():
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        stream.write(secrets.token_urlsafe(32))
path.chmod(0o600)
print(path.read_text().strip())
PY
)"
  export FORTE_API_KEY
  printf 'API-ключ сохранён в %s/jury-api-key\n' "$FORTE_DATA_DIR"
fi

printf 'Открыть сервис: http://localhost:%s\n' "${PORT:-8000}"
exec "$PROJECT_DIR/.venv-web/bin/python" -m uvicorn webapp.api:app \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" --workers 1 \
  --proxy-headers --forwarded-allow-ips "${FORWARDED_ALLOW_IPS:-127.0.0.1}"
