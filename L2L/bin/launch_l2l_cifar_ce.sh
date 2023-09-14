#!/bin/bash

# shellcheck disable=SC2206
#SBATCH --job-name=ce_cifar_l2l_params3_psize15
#SBATCH --account=raise-ctp2
#SBATCH --output=./outputs/ce_cifar_l2l.%j
#SBATCH --error=./errors/ce_cifar_l2l.%j
#SBATCH --partition=dc-gpu
#SBATCH --nodes=4
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00

ml --force purge

ml Stages/2023 GCCcore/.11.3.0 SciPy-Stack/2022a
ml Stages/2023 GCC/11.3.0 SciPy-Stack/2022a OpenMPI/4.1.4 PyTorch/1.12.0-CUDA-11.7 torchvision/0.13.1-CUDA-11.7
source /p/scratch/raise-ctp2/stefanovic1/l2l_juwels/env_juwels/bin/activate

COMMAND="l2l-cifar-ce.py --lr 0.05 --momentum 0.9 --dampening 0.7 --weight-decay 0.000001 --pop-size 15"
echo $COMMAND
python $COMMAND