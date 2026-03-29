#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

uv run python feishu_bot.py
