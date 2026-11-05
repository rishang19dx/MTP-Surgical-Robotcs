# Deploying the Autonomous Ultrasound Pipeline to a Remote GPU Server (SSH) with TotalSegmentator

This guide explains how to copy the codebase to a remote GPU host over SSH and run the pipeline
(`scripts/run_remote_totalsegmentator.sh`) on a TotalSegmentator-format CT.

> **Constraints:**
> - **Workspace.** The script refuses to run unless the project root is inside `*/workspace/rishang/*`. Set
>   `ALLOW_OUTSIDE_WORKSPACE=1` only for local testing. It keeps pip, HF, torch and matplotlib caches and temp files
>   inside the project (`.cache/`, `.tmp/`).
> - **Dependencies.** They come from the single top-level `requirements.txt`.
> - **Target.** The pipeline targets **L4** (the `vertebrae_L4` default of the benchmark and training scripts).
>   The script's `--vertebrae` option is parsed but not passed to any Python entry point.

---

## 1. Fast Transfer to Remote Server via SSH (`rsync`)

Run this command from your local terminal to copy the repository to your remote SSH server. 
It excludes the local venv, caches and the gitignored `.agents/` scratch directory (which may contain credentials):

```bash
# Sync local repository to the dedicated workspace directory on remote server
rsync -avz --progress \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude '*.log' \
  --exclude '.agents' \
  --exclude 'ReconstructionPipeline/.fem_cache' \
  ./ user@172.18.17.35:/home/user/workspace/rishang/MTP/
```

---

## 2. Remote TotalSegmentator Patient Directory Structure

On your remote server, organize the clinical patient data in standard TotalSegmentator format:

```text
/data/totalsegmentator/s0010/
├── ct.nii.gz                               # Full clinical CT volume (Hounsfield Units)
└── segmentations/                          # Multi-organ segmentations folder
    ├── vertebrae_L1.nii.gz                 # L1 lumbar vertebra binary mask
    ├── vertebrae_L2.nii.gz                 # L2 lumbar vertebra binary mask
    ├── vertebrae_L3.nii.gz                 # L3 lumbar vertebra binary mask
    ├── vertebrae_L4.nii.gz                 # L4 lumbar vertebra binary mask
    ├── vertebrae_L5.nii.gz                 # L5 lumbar vertebra binary mask
    ├── sacrum.nii.gz                       # Sacrum binary mask
    └── skin.nii.gz                         # Patient body/skin envelope
```

> **Tip**: If you only have a raw `ct.nii.gz` file on the remote server without precomputed segmentations, you can generate the segmentations automatically using the TotalSegmentator CLI:
> ```bash
> pip install TotalSegmentator
> TotalSegmentator -i /path/to/ct.nii.gz -o /path/to/subject/segmentations --fast --roi_subset vertebrae_L1 vertebrae_L2 vertebrae_L3 vertebrae_L4 vertebrae_L5 sacrum skin
> ```

---

## 3. Running the Pipeline on the Remote Server

SSH into the remote server:

```bash
ssh user@172.18.17.35
cd /home/user/workspace/rishang/MTP
```

### Option A: Complete End-to-End Automation (`--mode all`)
Bootstraps virtual environment, installs dependencies, ingests the clinical patient, trains Submodular PPO & Diffusion Policy, and runs the 6-policy benchmark:

```bash
bash scripts/run_remote_totalsegmentator.sh \
  --mode all \
  --dataset-dir /data/totalsegmentator/s0010 \
  --trials 5
```

---

### Option B: Modular Step-by-Step Execution

#### Step 1: Environment Setup
```bash
bash scripts/run_remote_totalsegmentator.sh --mode setup
```

#### Step 2: Ingest & Preprocess Patient Anatomy
Loads the subject with `PatientAssetGenerator.load_totalsegmentator_subject` and prints the volume shape, spacing, bone-point count and skin height. This step only validates the data. The FEM tissue model is built (and cached in `ReconstructionPipeline/.fem_cache`) on first use by training or the benchmark.
```bash
bash scripts/run_remote_totalsegmentator.sh \
  --mode prepare \
  --dataset-dir /data/totalsegmentator/s0010
```

#### Step 3: Train Autonomous Scanning Policies on Clinical Patient
Trains Submodular PPO (`--timesteps`, script default 15,000) and the CFM + EDM diffusion policies (`--epochs`, script default 30; 25 demo episodes) in the FEM environment. Note that the checkpoints behind `results/benchmark_latest.md` used the Python defaults instead: 50,000 PPO steps and 200 BC episodes with 60 epochs.
```bash
bash scripts/run_remote_totalsegmentator.sh \
  --mode train \
  --dataset-dir /data/totalsegmentator/s0010 \
  --timesteps 20000 \
  --epochs 40
```

#### Step 4: Run the 6-Policy Clinical Benchmark
Runs `benchmark_runner.py` for all six policies (raster, orbital, NBV, PPO, CFM, diffusion). Any learned-policy checkpoint missing from `checkpoints/` is an error, so run `--mode train` first:
```bash
bash scripts/run_remote_totalsegmentator.sh \
  --mode benchmark \
  --dataset-dir /data/totalsegmentator/s0010 \
  --trials 5
```

---

## 4. Retrieving Checkpoints & Results Back to Local Machine

Once remote training and benchmarking complete, pull the trained checkpoints and benchmark logs back to your local workstation:

```bash
# Run from your local terminal:
rsync -avz --progress \
  user@172.18.17.35:/home/user/workspace/rishang/MTP/checkpoints/ ./checkpoints/
rsync -avz --progress \
  user@172.18.17.35:/home/user/workspace/rishang/MTP/results/ ./results/
```

---

## 5. Output

The benchmark prints a table and writes `results/benchmark_<timestamp>.json` and `results/benchmark_latest.md`.
Earlier versions of this guide showed an example table (for example "Diffusion_Policy 82.1 %" and "Random_Exploration").
Those numbers were never produced by the code and have been removed. For the current numbers on the sample
patient, see [`results/benchmark_latest.md`](../results/benchmark_latest.md).
