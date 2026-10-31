#!/bin/bash
#SBATCH --job-name=logtest
#SBATCH --account=coc
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/storage/ice1/2/4/sliu736/umoe/logs/%x_%j.out

echo "Log test on $(hostname) at $(date)"
