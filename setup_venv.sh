#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

printf '\nEnvironment ready. Activate it with:\nsource .venv/bin/activate\n'
