#!/usr/bin/env bash
# Step-2 check (DEC-057, pre-registered): navigation trained on rigid / kinematic / deformable tissue, evaluated in all
# three on the two training patients. Trains the 4 missing policies (the 2 deformable ones exist from DEC-049),
# evaluates 6 policies + baselines in each deployment condition, then runs the frozen analysis.
#
#   bash scripts/run_nav_transfer.sh            # about 4-5 h on the laptop RTX 2050
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
C=checkpoints/vec_ppo
P2="../sample_totalsegmentator_patient,sonogym_s0015"
TAG=laptop2p_transfer
LOG=.agents/runs/nav_transfer_$(date +%Y%m%d_%H%M%S).log
mkdir -p .agents/runs
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

for cond in rigid kinematic; do
  for seed in 0 1; do
    f=$C/nav_${cond}_depth_s${seed}_${TAG}.zip
    if [ -f "$f" ]; then log "skip existing $f"; continue; fi
    log "train $cond seed $seed"
    timeout 3h $PY ReconstructionPipeline/planning/train_vec_ppo.py --task nav --deformation "$cond" --timesteps 300000 \
        --num-envs 32 --patients "$P2" --seed "$seed" --tag "$TAG" >> "$LOG" 2>&1
  done
done

CK="$C/nav_aware_depth_s0_laptop2p_offset.zip $C/nav_aware_depth_s1_laptop2p_offset.zip"
for cond in rigid kinematic; do CK="$CK $C/nav_${cond}_depth_s0_${TAG}.zip $C/nav_${cond}_depth_s1_${TAG}.zip"; done
declare -A OUT
for dep in aware rigid kinematic; do
  log "evaluate, deployed in $dep"
  line=$(timeout 2h $PY ReconstructionPipeline/evaluation/eval_vec_policies.py --task nav --patients "$P2" \
         --episodes 32 --deformation "$dep" --ppo $CK 2>>"$LOG" | tee -a "$LOG" | grep '^wrote')
  OUT[$dep]=$(echo "$line" | awk '{print $2}')
done
$PY ReconstructionPipeline/evaluation/nav_transfer_analysis.py --aware "${OUT[aware]}" --rigid "${OUT[rigid]}" \
    --kinematic "${OUT[kinematic]}" | tee -a "$LOG"
log done
