#!/usr/bin/env sh
# One command: set everything up (first run only) and start the Aimodel app.
#   ./start.sh            open it on this device, or on your phone with --phone
#   ./start.sh --phone    let your phone on the same Wi-Fi connect (prints a link)
# Any other options go to the server, e.g. ./start.sh --offline --port 9000
set -e
cd "$(dirname "$0")"

PY=python3
command -v "$PY" >/dev/null 2>&1 || PY=python
command -v "$PY" >/dev/null 2>&1 || { echo "Python 3 is needed. Install it from python.org (or: pkg install python in Termux)."; exit 1; }

if [ -n "$TERMUX_VERSION" ] || [ -d /data/data/com.termux ]; then
  # Termux: pip can't build numpy on a phone, so use the packaged one.
  "$PY" -c "import numpy" 2>/dev/null || pkg install -y python-numpy
else
  [ -d .venv ] || "$PY" -m venv .venv
  . .venv/bin/activate
  python -c "import numpy" 2>/dev/null || pip install -q -r requirements.txt
  PY=python
fi

args=""
for a in "$@"; do
  if [ "$a" = "--phone" ]; then args="$args --host 0.0.0.0"; else args="$args $a"; fi
done
# shellcheck disable=SC2086
exec "$PY" -m aimodel.server $args
