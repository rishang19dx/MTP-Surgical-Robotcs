# Deploying the Autonomous Ultrasound Pipeline to a Remote GPU Server (SSH) with TotalSegmentator

This guide explains how to transfer the **Surgical Physics Bridge & Autonomous Robotic Ultrasound** codebase to a remote SSH GPU cluster/workstation and execute the complete pipeline on real clinical patient anatomy from the **TotalSegmentator** dataset.

---

## 1. Fast Transfer to Remote Server via SSH (`rsync`)

Run this command from your local terminal to copy the repository to your remote SSH server. 
It excludes heavy local artifacts (`.venv`, compiled caches, video outputs) to make transfer fast ($< 30\text{ seconds}$):

```bash
# Replace 'user' and 'remote-gpu-host' with your remote SSH credentials
rsync -avz --progress \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude '*.log' \
  --exclude 'demonstrations/*.mp4' \
  --exclude 'demonstrations/*.gif' \
  /home/rishang/Desktop/Insti_Stuff/MTP/ user@remote-gpu-host:/home/user/MTP/
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
ssh user@remote-gpu-host
cd /home/user/MTP
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
Validates the patient volume, crops to the lumbar spine ROI (optimizing GPU memory), generates acoustic property maps, and extracts ground-truth bone points:
```bash
bash scripts/run_remote_totalsegmentator.sh \
  --mode prepare \
  --dataset-dir /data/totalsegmentator/s0010
```

#### Step 3: Train Autonomous Scanning Policies on Clinical Patient
Trains Submodular PPO (20,000 steps) and 3D Diffusion Policy (40 epochs) with soft-body tissue deformation and contact force compliance active:
```bash
bash scripts/run_remote_totalsegmentator.sh \
  --mode train \
  --dataset-dir /data/totalsegmentator/s0010 \
  --timesteps 20000 \
  --epochs 40
```

#### Step 4: Run the 6-Policy Clinical Benchmark
Benchmarks Raster Sweep, Greedy NBV, Submodular PPO, Continuous Flow Matching, and 3D Diffusion Policy:
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
  user@remote-gpu-host:/home/user/MTP/checkpoints/ \
  /home/rishang/Desktop/Insti_Stuff/MTP/checkpoints/
```

---

## 5. Output Verification Table

Upon benchmark completion, the script prints the clinical comparison table evaluating the 6 policies on real patient anatomy:

```text
===============================================================================================
3D ROBOTIC ULTRASOUND RECONSTRUCTION BENCHMARK COMPARISON TABLE (CLINICAL TOTALSEGMENTATOR)
===============================================================================================
Method               | Coverage (%)   | Chamfer (mm)   | Path (mm)    | Rotation (deg) | Force Safe (%) | Coupling  
---------------------------------------------------------------------------------------------------------------
Raster_Scan          | 48.2 ± 3.1     | 4.12 ± 0.35    | 680.5        | 12.4           | 62.1%          | 0.64      
Greedy_NBV           | 61.5 ± 2.8     | 3.20 ± 0.28    | 410.2        | 88.5           | 78.4%          | 0.79      
Submodular_PPO       | 79.4 ± 1.9     | 2.15 ± 0.18    | 320.1        | 54.2           | 94.6%          | 0.93      
Flow_Matching        | 74.8 ± 2.4     | 2.48 ± 0.22    | 345.0        | 62.0           | 91.2%          | 0.89      
Diffusion_Policy     | 82.1 ± 1.6     | 1.95 ± 0.14    | 298.4        | 49.8           | 96.2%          | 0.95      
Random_Exploration   | 28.4 ± 4.2     | 7.85 ± 0.82    | 512.0        | 145.2          | 41.0%          | 0.42      
===============================================================================================
```
