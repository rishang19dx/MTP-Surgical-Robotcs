#!/usr/bin/env bash
# Start a deformation demo: the physics/PyBullet driver and 3D Slicer, connected over OpenIGTLink (port 18945).
#   bash demos/launch_demo.sh <spine|muscle|liver|vein> [driver args...]
# Env: MTP_DEMO_MAGNIFY, MTP_DEMO_SNAPSHOT_DIR, MTP_DEMO_SNAPSHOT_S, MTP_DEMO_EXIT_AFTER_S (passed to Slicer).
set -euo pipefail
cd "$(dirname "$0")/.."
SCENE=${1:?scene}; shift || true
export DISPLAY="${DISPLAY:-:0}"
SLICER=""
for c in "$HOME/Downloads/Slicer-5.12.4-linux-amd64/Slicer" "$(command -v Slicer 2>/dev/null || true)"; do
  [ -n "$c" ] && [ -x "$c" ] && SLICER="$c" && break
done
./.venv/bin/python -m demos.run_demo --scene "$SCENE" --magnify "${MTP_DEMO_MAGNIFY:-1}" "$@" &
DRIVER=$!
if [ -n "$SLICER" ]; then
  MTP_DEMO_SCENE="demos/scenes/$SCENE" "$SLICER" --no-splash --python-script demos/slicer_demo.py &
  SL=$!
fi
wait $DRIVER || true
[ -n "${SL:-}" ] && kill $SL 2>/dev/null || true
