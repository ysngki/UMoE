module load anaconda3
module load cuda/12.6
conda activate ~/scratch/umoe/envs/umoe

export HF_HOME=~/scratch/umoe/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export WANDB_MODE=offline
export WANDB_DIR=~/scratch/umoe/logs
export TORCH_CUDA_ARCH_LIST="8.0;9.0"   # A100 = 8.0, H100/H200 = 9.0
