#!/bin/bash
# Інтерфейс toksym у браузері: http://127.0.0.1:7373/  (python3 з numpy — з ~/venv)
cd "$(dirname "$0")/.."
export PATH="$HOME/venv/bin:$PATH"
exec kit/participant/toksym-linux ui --no-open
