#!/bin/bash
#SBATCH -J vggt-omega
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o /data2/buaajianghengchen/storageCode/vggt-omega/logs/%x-%j.out
#SBATCH -e /data2/buaajianghengchen/storageCode/vggt-omega/logs/%x-%j.err

set -euo pipefail

mkdir -p /data2/buaajianghengchen/storageCode/vggt-omega/logs
cd /data2/buaajianghengchen/storageCode/vggt-omega

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

source ~/.bashrc
source /home/buaajianghengchen/anaconda3/etc/profile.d/conda.sh
conda activate /data2/buaajianghengchen/conda_envs/vggtomega

python demo_colmap.py \
  --input_path ~/datasets/buaa1 \
  --output_path output/buaa1 \
  --checkpoint checkpoints/VGGT-Omega-1B-512/model.pt
