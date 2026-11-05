#!/usr/bin/env bash
# ==============================================================================
# MTP - Surgical Physics Bridge & Autonomous Robotic Ultrasound
# Dual-Window PyBullet Native GUI & 3D Slicer Clinical Suite Launcher
# ==============================================================================
#
# This script launches:
# 1. Native PyBullet 3D GUI Window:
#    - Franka Emika Panda 7-DoF robot manipulator with mounted ultrasound probe
#    - Operating table and patient torso soft-tissue phantom (TotalSegmentator CT)
#    - 1 kHz physics in its own process; probe force/indentation from the tissue model
# 2. 3D Slicer Clinical Application (Slicer 5.12+):
#    - Side-by-side view: 2D Live B-mode ultrasound slice & 3D patient space
#    - Dynamic 3D ultrasound transducer probe model
#    - Real-time incremental bone surface point cloud reconstruction
# ==============================================================================

set -euo pipefail

# ANSI Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Default parameters
export DISPLAY="${DISPLAY:-:0}"
SCAN_MODE="orbital_l4"
PUBLISH_FPS="30.0"
PORT="18944"
PATIENT_DIR="${PROJECT_ROOT}/data/sample_totalsegmentator_patient"
URDF_PATH=""

print_banner() {
    echo -e "${CYAN}${BOLD}"
    echo "================================================================================"
    echo "    SURGICAL PHYSICS BRIDGE: DUAL-WINDOW PYBULLET & 3D SLICER STUDIO           "
    echo "================================================================================"
    echo -e "${NC}"
}

print_help() {
    print_banner
    echo "Usage: bash scripts/launch_dual_pybullet_slicer.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --mode <mode>              orbital_l4|raster_l4|orbital|raster|nbv|diffusion|cfm|interactive (default: orbital_l4)"
    echo "  --fps <n>                  Publishing rate to Slicer in Hz (default: 30.0)"
    echo "  --port <port>              OpenIGTLink socket port (default: 18944)"
    echo "  --patient-dir <path>       Custom TotalSegmentator patient directory"
    echo "  --urdf <path>              Custom robot URDF path"
    echo "  --help                     Display this help message and exit"
    echo ""
}

# Parse CLI options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            SCAN_MODE="$2"
            shift 2
            ;;
        --fps)
            PUBLISH_FPS="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --patient-dir)
            PATIENT_DIR="$2"
            shift 2
            ;;
        --urdf)
            URDF_PATH="$2"
            shift 2
            ;;
        --help|-h)
            print_help
            exit 0
            ;;
        *)
            echo -e "${RED}[ERROR] Unknown option: $1${NC}"
            print_help
            exit 1
            ;;
    esac
done

print_banner

# 1. Verify Virtual Environment
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
if [ ! -f "${PYTHON_BIN}" ]; then
    echo -e "${RED}[ERROR] Virtual environment not found at ${PYTHON_BIN}${NC}"
    echo "Please initialize the environment with: python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt"
    exit 1
fi
echo -e "${GREEN}[✓] Python environment:${NC} ${PYTHON_BIN}"

# 2. Verify or Generate TotalSegmentator Patient Dataset
if [ ! -d "${PATIENT_DIR}" ] || [ ! -f "${PATIENT_DIR}/ct.nii.gz" ]; then
    echo -e "${YELLOW}[*] TotalSegmentator patient dataset not found at ${PATIENT_DIR}. Generating synthetic clinical patient...${NC}"
    "${PYTHON_BIN}" -m ReconstructionPipeline.data.prepare_patient_assets --export-totalseg "${PATIENT_DIR}"
fi
echo -e "${GREEN}[✓] TotalSegmentator Patient:${NC} ${PATIENT_DIR}"

# 3. Locate 3D Slicer Executable
SLICER_CANDIDATES=(
    "${HOME}/Downloads/Slicer-5.12.4-linux-amd64/Slicer"
    "${PROJECT_ROOT}/Slicer-App/Slicer"
    "/opt/slicer/Slicer"
    "/usr/local/bin/Slicer"
)
SLICER_BIN=""
for cand in "${SLICER_CANDIDATES[@]}"; do
    if [ -x "${cand}" ]; then
        SLICER_BIN="${cand}"
        break
    fi
done

if [ -z "${SLICER_BIN}" ]; then
    # Try which Slicer
    if command -v Slicer &> /dev/null; then
        SLICER_BIN="$(command -v Slicer)"
    fi
fi

if [ -z "${SLICER_BIN}" ]; then
    echo -e "${RED}[ERROR] 3D Slicer executable not found!${NC}"
    echo "Checked locations:"
    for cand in "${SLICER_CANDIDATES[@]}"; do
        echo "  - ${cand}"
    done
    exit 1
fi
echo -e "${GREEN}[✓] 3D Slicer executable:${NC} ${SLICER_BIN}"

# 4. Clean up any existing instances on this port
echo -e "${CYAN}[*] Clearing port ${PORT}...${NC}"
fuser -k "${PORT}/tcp" 2>/dev/null || true
sleep 1

# Process termination trap
cleanup() {
    echo -e "\n${YELLOW}[*] Shutting down simulation and visualization suite...${NC}"
    if [ -n "${SERVER_PID:-}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
    fi
    if [ -n "${SLICER_PID:-}" ] && kill -0 "${SLICER_PID}" 2>/dev/null; then
        kill "${SLICER_PID}" 2>/dev/null || true
    fi
    echo -e "${GREEN}[✓] All processes stopped cleanly.${NC}"
}
trap cleanup EXIT INT TERM

# 5. Launch Unified PyBullet Physics Server with Native GUI
echo -e "${CYAN}[*] Launching PyBullet Physics Suite (Franka Panda + Patient Soft Tissue)...${NC}"
SERVER_ARGS=(
    "${PROJECT_ROOT}/surgical_physics_bridge/src/unified_server.py"
    "--port" "${PORT}"
    "--gui"
    "--fps" "${PUBLISH_FPS}"
    "--mode" "${SCAN_MODE}"
    "--patient-dir" "${PATIENT_DIR}"
)
if [ -n "${URDF_PATH}" ]; then
    SERVER_ARGS+=("--urdf" "${URDF_PATH}")
fi

"${PYTHON_BIN}" "${SERVER_ARGS[@]}" &
SERVER_PID=$!

# Wait for OpenIGTLink socket server to initialize
echo -e "${CYAN}[*] Waiting for OpenIGTLink Server to bind port ${PORT}...${NC}"
for i in {1..20}; do
    if ss -tln 2>/dev/null | grep -q ":${PORT} "; then
        echo -e "${GREEN}[✓] OpenIGTLink Server ready on port ${PORT}!${NC}"
        break
    fi
    sleep 0.5
done

# 6. Launch 3D Slicer with Live Bridge Script
echo -e "${CYAN}[*] Launching 3D Slicer (Side-by-Side Live Ultrasound & 3D Bone Reconstruction)...${NC}"
"${SLICER_BIN}" \
    --python-script "${PROJECT_ROOT}/Slicer/slicer_surgical_physics_bridge.py" &
SLICER_PID=$!

echo -e "${GREEN}${BOLD}"
echo "================================================================================"
echo " [✓] DUAL-WINDOW CLINICAL SUITE RUNNING:"
echo "     Window 1: PyBullet 3D Robot Arm & Contact Physics Simulation"
echo "     Window 2: 3D Slicer 2D Live B-Mode Ultrasound & 3D Bone Reconstruction"
echo "================================================================================"
echo -e "${NC}"
echo "Press Ctrl+C to stop both sessions."

# Wait on processes
wait "${SERVER_PID}" "${SLICER_PID}"
