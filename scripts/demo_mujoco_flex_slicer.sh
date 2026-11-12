#!/usr/bin/env bash
# ==============================================================================
# MuJoCo flex soft-tissue demonstration with the Slicer bridge (DEC-053, DEMO ONLY).
#
# Runs the unified server with the MuJoCo rigid backend (robot, probe) and the live MuJoCo 4 flex tissue
# (.venv-mj4 process), streams to 3D Slicer, and records:
#   <out>/raw/flex/     offscreen MuJoCo renders of the flex tissue + probe, with flex_frames.csv (force, depth)
#   <out>/raw/slicer/   Slicer main-window grabs, with snapshots.csv (streamed telemetry)
#   <out>/raw/viewer/   MuJoCo robot viewer snapshots (--gui)
# then assembles GIFs with ReconstructionPipeline/scripts/make_mujoco_flex_demo.py.
#
# Usage: bash scripts/demo_mujoco_flex_slicer.sh [--seconds 60] [--mode raster_l4] [--out demonstrations/mujoco_flex]
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SECONDS_RUN=60
MODE="raster_l4"
OUT="demonstrations/mujoco_flex"
PORT=18944
while [ $# -gt 0 ]; do
    case "$1" in
        --seconds) SECONDS_RUN="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        *) echo "unknown option $1"; exit 2 ;;
    esac
done

export DISPLAY="${DISPLAY:-:0}"
[ -x .venv-mj4/bin/python ] || { echo "[ERROR] .venv-mj4 missing (AGENTS.md invariant 1)"; exit 1; }
SLICER_BIN=""
for cand in "${HOME}/Downloads/Slicer-5.12.4-linux-amd64/Slicer" "${PROJECT_ROOT}/Slicer-App/Slicer" \
            "/opt/slicer/Slicer" "$(command -v Slicer 2>/dev/null || true)"; do
    if [ -n "${cand}" ] && [ -x "${cand}" ]; then SLICER_BIN="${cand}"; break; fi
done
[ -n "${SLICER_BIN}" ] || { echo "[ERROR] 3D Slicer not found"; exit 1; }

RAW="${OUT}/raw"
rm -rf "${RAW}"
mkdir -p "${RAW}/flex" "${RAW}/slicer" "${RAW}/viewer"

cleanup() {
    [ -n "${SLICER_PID:-}" ] && kill "${SLICER_PID}" 2>/dev/null || true
    if [ -n "${SERVER_PID:-}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill -INT "${SERVER_PID}" 2>/dev/null || true
        for _ in $(seq 1 30); do kill -0 "${SERVER_PID}" 2>/dev/null || break; sleep 0.5; done
        kill "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "[*] Server: MuJoCo rigid backend + live MuJoCo flex tissue (demo only), mode ${MODE}"
MTP_FLEX_SNAPSHOT_DIR="${RAW}/flex" MTP_FLEX_SNAPSHOT_S=0.2 \
MTP_VIEWER_SNAPSHOT_DIR="${RAW}/viewer" MTP_VIEWER_SNAPSHOT_S=2 \
    ./.venv/bin/python -u surgical_physics_bridge/src/unified_server.py --mode "${MODE}" --fps 30 --gui \
    --physics-backend mujoco --tissue-backend mujoco-flex --port "${PORT}" > "${RAW}/server.log" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 240); do
    grep -q "Unified server | port" "${RAW}/server.log" 2>/dev/null && break
    kill -0 "${SERVER_PID}" 2>/dev/null || { echo "[ERROR] server exited"; tail -30 "${RAW}/server.log"; exit 1; }
    sleep 0.5
done

echo "[*] Slicer with the bridge (snapshots every 2 s, exits after ${SECONDS_RUN} s)"
MTP_SLICER_SNAPSHOT_DIR="${RAW}/slicer" MTP_SLICER_SNAPSHOT_S=2 MTP_SLICER_EXIT_AFTER_S="${SECONDS_RUN}" \
    "${SLICER_BIN}" --no-splash --python-script "${PROJECT_ROOT}/Slicer/slicer_surgical_physics_bridge.py" \
    > "${RAW}/slicer.log" 2>&1 &
SLICER_PID=$!
wait "${SLICER_PID}" || true
SLICER_PID=""
cleanup
SERVER_PID=""

./.venv/bin/python -m ReconstructionPipeline.scripts.make_mujoco_flex_demo --raw "${RAW}" --out "${OUT}"
