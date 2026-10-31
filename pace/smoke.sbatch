#!/bin/bash
#SBATCH --job-name=umoe-smoke
#SBATCH --account=coc
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/storage/ice1/2/4/sliu736/umoe/logs/%x_%j.out

source ~/UMoE/env.sh
cd ~/UMoE

nvidia-smi
python -c "import torch; print('CUDA ok:', torch.cuda.is_available(), torch.cuda.get_device_name())"
python -c "import grouped_gemm; print('grouped_gemm ok')"

python train.py \
    --config_name small-vanilla-UMoE-1024 \
    --run_name smoke \
    --output_dir /storage/ice1/2/4/sliu736/umoe/ckpt \
    --wikitext_103 \
    --dataset_text_field text \
    --disable_iter_shuffle \
    --world_size 1 \
    --micro_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --global_batch_size 4 \
    --max_steps 50 \
    --save_steps 25 \
    --logging_steps 5 \
    --num_proc 16 \
    --disable_wandb
