#!/usr/bin/env bash
# ==============================================================================
# MTP - Surgical Physics Bridge & Autonomous Robotic Ultrasound
# Remote TotalSegmentator Clinical Deployment & Benchmark Automation Script
# ==============================================================================
#
# QUICK TRANSFER GUIDE (From Local Machine to Remote Server via SSH):
# ------------------------------------------------------------------------------
# 1. Transfer codebase to remote server (excluding large binaries & local venv):
#    rsync -avz --progress \
#      --exclude '.venv' \
#      --exclude '__pycache__' \
#      --exclude '.pytest_cache' \
#      --exclude '*.log' \
#      --exclude 'demonstrations/*.mp4' \
#      --exclude 'demonstrations/*.gif' \
#      /path/to/local/MTP/ user@remote-gpu-host:/home/user/MTP/
#
# 2. SSH into remote machine and run:
#    ssh user@remote-gpu-host
#    cd /home/user/MTP
#    bash scripts/run_remote_totalsegmentator.sh --mode all --dataset-dir /path/to/totalsegmentator/subject
# ==============================================================================

set -euo pipefail

# Default configuration parameters
MODE="all"
DATASET_DIR=""
TARGET_VERTEBRAE="vertebrae_L1,vertebrae_L2,vertebrae_L3,vertebrae_L4,vertebrae_L5,sacrum"
NUM_TRIALS=3
TRAIN_TIMESTEPS=15000
DIFFUSION_EPOCHS=30
DEVICE=""
PYTHON_BIN=""

# Color formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

print_banner() {
    echo -e "${CYAN}${BOLD}"
    echo "================================================================================"
    echo "      AUTONOMOUS ROBOTIC ULTRASOUND - CLINICAL TOTALSEGMENTATOR SUITE          "
    echo "================================================================================"
    echo -e "${NC}"
}

print_help() {
    print_banner
    echo "Usage: bash scripts/run_remote_totalsegmentator.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --dataset-dir <path>     Path to TotalSegmentator patient directory (containing"
    echo "                           ct.nii.gz and segmentations/ or vertebrae_L*.nii.gz)"
    echo "  --mode <mode>            Pipeline execution mode:"
    echo "                             setup     : Bootstrap venv and install dependencies"
    echo "                             prepare   : Ingest and preprocess clinical patient CT"
    echo "                             train     : Train Submodular PPO and Diffusion Policy"
    echo "                             benchmark : Run 6-policy benchmark on clinical CT"
    echo "                             all       : Run setup -> prepare -> train -> benchmark (Default)"
    echo "  --trials <n>             Number of Monte Carlo trials per policy (default: 3)"
    echo "  --timesteps <n>          PPO environment steps (default: 15000)"
    echo "  --epochs <n>             Diffusion Policy epochs (default: 30)"
    echo "  --vertebrae <list>       Comma-separated vertebrae list (default: L1-L5,sacrum)"
    echo "  --device <cpu|cuda>      Compute device (auto-detected if omitted)"
    echo "  --help                   Display this help message and exit"
    echo ""
    echo "Examples:"
    echo "  # Complete automated end-to-end run on clinical subject s0010:"
    echo "  bash scripts/run_remote_totalsegmentator.sh --dataset-dir /data/totalsegmentator/s0010 --mode all"
    echo ""
    echo "  # Run only the 6-policy benchmark on an existing subject:"
    echo "  bash scripts/run_remote_totalsegmentator.sh --dataset-dir /data/totalsegmentator/s0010 --mode benchmark --trials 5"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-dir)
            DATASET_DIR="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --trials)
            NUM_TRIALS="$2"
            shift 2
            ;;
        --timesteps)
            TRAIN_TIMESTEPS="$2"
            shift 2
            ;;
        --epochs)
            DIFFUSION_EPOCHS="$2"
            shift 2
            ;;
        --vertebrae)
            TARGET_VERTEBRAE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ==============================================================================
# Phase 1: Environment & Dependency Setup
# ==============================================================================
setup_environment() {
    echo -e "\n${BLUE}${BOLD}[Phase 1/4] Checking Remote Environment & Dependencies...${NC}"

    # 1. Check GPU availability
    if command -v nvidia-smi &>/dev/null; then
        echo -e "${GREEN}[✓] NVIDIA GPU detected:${NC}"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
        if [[ -z "${DEVICE}" ]]; then
            DEVICE="cuda"
        fi
    else
        echo -e "${YELLOW}[!] No NVIDIA GPU detected. Falling back to CPU.${NC}"
        if [[ -z "${DEVICE}" ]]; then
            DEVICE="cpu"
        fi
    fi

    # 2. Python Virtual Environment Setup
    if [[ -d ".venv" && -f ".venv/bin/python" ]]; then
        echo -e "${GREEN}[✓] Existing virtual environment detected at ./.venv${NC}"
        PYTHON_BIN="./.venv/bin/python"
    else
        echo -e "${YELLOW}[*] Creating new Python virtual environment (.venv)...${NC}"
        if command -v python3.10 &>/dev/null; then
            python3.10 -m venv .venv
        elif command -v python3 &>/dev/null; then
            python3 -m venv .venv
        else
            echo -e "${RED}[ERROR] Python 3 not found on system. Please install python3-venv.${NC}"
            exit 1
        fi
        PYTHON_BIN="./.venv/bin/python"

        echo -e "${YELLOW}[*] Installing required packages...${NC}"
        ${PYTHON_BIN} -m pip install --upgrade pip setuptools wheel

        # Install PyTorch with appropriate CUDA or CPU wheels
        if [[ "${DEVICE}" == "cuda" ]]; then
            echo -e "${YELLOW}[*] Installing PyTorch with CUDA support...${NC}"
            ${PYTHON_BIN} -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 || \
            ${PYTHON_BIN} -m pip install torch torchvision
        else
            echo -e "${YELLOW}[*] Installing PyTorch (CPU version)...${NC}"
            ${PYTHON_BIN} -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        fi

        # Install required robotics, clinical medical imaging, and simulation libraries
        echo -e "${YELLOW}[*] Installing scientific & clinical dependencies...${NC}"
        ${PYTHON_BIN} -m pip install \
            nibabel \
            trimesh \
            open3d \
            pyigtl \
            pytest \
            scipy \
            matplotlib \
            imageio \
            imageio-ffmpeg \
            pybullet \
            tqdm
    fi

    # Verify critical package imports
    ${PYTHON_BIN} -c "
import torch, nibabel, trimesh, open3d, pyigtl, scipy
print(f'[✓] Python environment verified. PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}')
"
    echo -e "${GREEN}[✓] Environment setup completed successfully.${NC}"
}

# ==============================================================================
# Phase 2: Patient Asset Ingestion & Preprocessing
# ==============================================================================
prepare_dataset() {
    echo -e "\n${BLUE}${BOLD}[Phase 2/4] Ingesting TotalSegmentator Clinical Anatomy...${NC}"

    if [[ -z "${DATASET_DIR}" ]]; then
        echo -e "${YELLOW}[!] No --dataset-dir specified. Checking sample data or falling back to synthetic phantom.${NC}"
        if [[ -d "ReconstructionPipeline/data/sample_patient" ]]; then
            DATASET_DIR="ReconstructionPipeline/data/sample_patient"
            echo -e "${CYAN}[*] Using local sample directory: ${DATASET_DIR}${NC}"
        else
            echo -e "${YELLOW}[!] Proceeding with synthetic lumbar anatomy.${NC}"
            return 0
        fi
    fi

    if [[ ! -d "${DATASET_DIR}" ]]; then
        echo -e "${RED}[ERROR] Specified TotalSegmentator directory does not exist: ${DATASET_DIR}${NC}"
        exit 1
    fi

    echo -e "${CYAN}[*] Inspecting TotalSegmentator assets in: ${DATASET_DIR}${NC}"
    ${PYTHON_BIN} -c "
import os, glob
from ReconstructionPipeline.data.prepare_patient_assets import PatientAssetGenerator

subject_dir = '${DATASET_DIR}'
print(f'[*] Loading clinical subject from: {subject_dir}')
assets = PatientAssetGenerator.load_totalsegmentator_subject(
    subject_dir=subject_dir,
    crop_to_spine=True,
    margin_voxels=15
)

ct = assets['ct_volume']
spacing = assets['spacing_mm']
bone_pts = assets['bone_points']
skin_y = assets['skin_y_mm']

print('[✓] Patient Dataset Successfully Ingested:')
print(f'    - CT Dimensions:      {ct.shape} voxels')
print(f'    - Physical Spacing:   {spacing} mm')
print(f'    - Total Bone Points:  {len(bone_pts)} ground-truth surface coordinates')
print(f'    - Dorsal Skin Plane:  {skin_y:.1f} mm')
print(f'    - Loaded Structures:  {assets.get(\"loaded_structures\", [])}')
"
    echo -e "${GREEN}[✓] TotalSegmentator patient preprocessing complete.${NC}"
}

# ==============================================================================
# Phase 3: Train Active Policies on Clinical Volume
# ==============================================================================
train_policies() {
    echo -e "\n${BLUE}${BOLD}[Phase 3/4] Training Autonomous Scanning Policies on Clinical Anatomy...${NC}"
    mkdir -p checkpoints

    local patient_flag=""
    if [[ -n "${DATASET_DIR}" && -d "${DATASET_DIR}" ]]; then
        patient_flag="--patient-dir ${DATASET_DIR}"
    fi

    # 1. Train Submodular PPO
    echo -e "\n${CYAN}${BOLD}[3.1] Training Submodular PPO Active View Planner (${TRAIN_TIMESTEPS} timesteps)...${NC}"
    ${PYTHON_BIN} ReconstructionPipeline/planning/train_submodular_ppo.py \
        --total-timesteps "${TRAIN_TIMESTEPS}" \
        --rollout-steps 128 \
        --num-epochs 4 \
        --batch-size 64 \
        --save-path checkpoints/submodular_ppo.pt \
        --device "${DEVICE}" \
        ${patient_flag}

    # 2. Train 3D Diffusion Policy (Karras EDM)
    echo -e "\n${CYAN}${BOLD}[3.2] Training 3D Diffusion Policy with Soft-Body Feedback (${DIFFUSION_EPOCHS} epochs)...${NC}"
    ${PYTHON_BIN} ReconstructionPipeline/planning/train_diffusion_policy.py \
        --num-demo-episodes 25 \
        --epochs "${DIFFUSION_EPOCHS}" \
        --batch-size 32 \
        --save-path checkpoints/diffusion_policy.pt \
        --device "${DEVICE}" \
        ${patient_flag}

    echo -e "${GREEN}[✓] Policy training finished. Models saved in ./checkpoints/${NC}"
}

# ==============================================================================
# Phase 4: Run 6-Policy Comparative Benchmark
# ==============================================================================
run_benchmark() {
    echo -e "\n${BLUE}${BOLD}[Phase 4/4] Executing 6-Policy Reconstruction Benchmark (${NUM_TRIALS} Trials)...${NC}"

    local patient_flag=""
    if [[ -n "${DATASET_DIR}" && -d "${DATASET_DIR}" ]]; then
        patient_flag="--patient-dir ${DATASET_DIR}"
    fi

    local ppo_flag=""
    if [[ -f "checkpoints/submodular_ppo.pt" ]]; then
        ppo_flag="--ppo-ckpt checkpoints/submodular_ppo.pt"
    fi

    local diff_flag=""
    if [[ -f "checkpoints/diffusion_policy.pt" ]]; then
        diff_flag="--diffusion-ckpt checkpoints/diffusion_policy.pt"
    fi

    ${PYTHON_BIN} ReconstructionPipeline/evaluation/benchmark_runner.py \
        --num-trials "${NUM_TRIALS}" \
        --device "${DEVICE}" \
        ${patient_flag} \
        ${ppo_flag} \
        ${diff_flag}

    echo -e "\n${GREEN}${BOLD}[✓] Benchmark execution complete. Metrics verified against ground truth!${NC}"
}

# ==============================================================================
# Main Orchestration Loop
# ==============================================================================
print_banner
echo -e "${BOLD}Target Mode:${NC} ${MODE}"
echo -e "${BOLD}Compute Device:${NC} ${DEVICE:-auto}"
echo -e "${BOLD}Patient Dataset:${NC} ${DATASET_DIR:-Synthetic Phantom}"
echo -e "${BOLD}Project Root:${NC} ${PROJECT_ROOT}"

case "${MODE}" in
    setup)
        setup_environment
        ;;
    prepare)
        PYTHON_BIN="./.venv/bin/python"
        prepare_dataset
        ;;
    train)
        PYTHON_BIN="./.venv/bin/python"
        train_policies
        ;;
    benchmark)
        PYTHON_BIN="./.venv/bin/python"
        run_benchmark
        ;;
    all)
        setup_environment
        prepare_dataset
        train_policies
        run_benchmark
        ;;
    *)
        echo -e "${RED}[ERROR] Invalid mode: ${MODE}. Choose from: setup, prepare, train, benchmark, all${NC}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}${BOLD}================================================================================${NC}"
echo -e "${GREEN}${BOLD}             PIPELINE EXECUTION SUCCESSFULLY COMPLETED!                         ${NC}"
echo -e "${GREEN}${BOLD}================================================================================${NC}"
