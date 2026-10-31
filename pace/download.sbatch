#!/bin/bash
#SBATCH --job-name=fwedu-download
#SBATCH --account=coc                 # confirm with: sacctmgr show assoc user=$USER
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00 
#SBATCH --output=/storage/ice1/2/4/sliu736/umoe/logs/%x_%j.out   # adjust to your real scratch path
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sliu73@gatech.edu

# ---- Setup ----
source ~/UMoE/env.sh
mkdir -p ~/scratch/umoe/data/fineweb-edu

# ---- Download with retries (resumes partial files automatically) ----
for attempt in 1 2 3 4 5; do
    echo "=== Download attempt $attempt at $(date) ==="
    hf download HuggingFaceFW/fineweb-edu --repo-type dataset \
        --include "sample/10BT/000_00000.parquet" \
        --local-dir ~/scratch/umoe/data/fineweb-edu \
        && { echo "=== Download complete ==="; exit 0; }
    echo "Attempt $attempt failed; retrying in 30s"
    sleep 30
done

echo "=== All attempts failed ==="
exit 1
