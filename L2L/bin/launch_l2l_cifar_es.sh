#!/bin/bash

# shellcheck disable=SC2206
#SBATCH --job-name=meeting
#SBATCH --account=raise-ctp2
#SBATCH --output=./outputs/meeting.%j
#SBATCH --error=./errors/meeting.%j
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=2
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --threads-per-core=1


ml --force purge

ml Stages/2023 GCCcore/.11.3.0 SciPy-Stack/2022a
ml Stages/2023 GCC/11.3.0 SciPy-Stack/2022a OpenMPI/4.1.4 PyTorch/1.12.0-CUDA-11.7 torchvision/0.13.1-CUDA-11.7
source /p/scratch/raise-ctp2/stefanovic1/l2l_juwels/env_juwels/bin/activate

##export CUDA_VISIBLE_DEVICES=0

COMMAND="l2l-cifar-es.py --lr 0.0002 --momentum 0.1 --dampening 0.01 --weight-decay 0.0001 --pop-size 15 --denominator 4"
echo $COMMAND

python $COMMAND
