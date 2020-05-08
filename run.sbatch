#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --time=48:00:00
#SBATCH --job-name="UDA"

source activate mixmatch

cd /scratch/wz1232/UDA_pytorch
$1
$2
$3
$4
$5
$6
$7