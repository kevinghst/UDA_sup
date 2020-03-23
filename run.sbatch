#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --time=12:00:00
#SBATCH --job-name="test"

source activate tf-gpu

cd /scratch/wz1232/UDA_pytorch
python main.py --uda_mode --uda_test_mode --check_after 2999