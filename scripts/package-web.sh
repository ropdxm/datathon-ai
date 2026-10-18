#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
"$PROJECT_DIR/.venv-web/bin/python" - <<'PY'
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path.cwd()
destination = root / 'outputs' / 'forte-web-mvp.zip'
destination.parent.mkdir(parents=True, exist_ok=True)
sources = ['webapp', 'scripts/run-web.sh', 'scripts/package-web.sh', 'scripts/create-demo.py',
           'requirements-web.txt', '.env.example', '.gitignore',
           'README.md', 'README_WEB.md', 'docs/web-api.md', 'examples/custom_analyzer.py', 'tests/web',
           'colab_forte/forte_benchmark']
with ZipFile(destination, 'w', ZIP_DEFLATED) as archive:
    for source in sources:
        item = root / source
        paths = item.rglob('*') if item.is_dir() else [item]
        for path in paths:
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                archive.write(path, path.relative_to(root))
print(destination)
PY
