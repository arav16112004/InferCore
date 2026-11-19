#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m worker.model
if ! command -v cargo >/dev/null 2>&1; then
  echo 'Rust missing: install Rust 1.91.1 using rustup, then retry. See README.' >&2
  exit 1
fi
cargo build --release --locked
