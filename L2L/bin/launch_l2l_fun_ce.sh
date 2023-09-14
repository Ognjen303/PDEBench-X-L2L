#!/bin/bash

# shellcheck disable=SC2206
#SBATCH --job-name=fun_ce_l2l_test
#SBATCH --account=raise-ctp2
#SBATCH --output=./outputs/out_fun_ce_l2l_test.%j
#SBATCH --error=./errors/err_fun_ce_l2l_test.%j
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00

ml --force purge

ml Stages/2023 GCCcore/.11.3.0 SciPy-Stack/2022a
ml Stages/2023 GCC/11.3.0 SciPy-Stack/2022a OpenMPI/4.1.4 PyTorch/1.12.0-CUDA-11.7 torchvision/0.13.1-CUDA-11.7
source /p/scratch/raise-ctp2/stefanovic1/l2l_juwels/env_juwels/bin/activate

python l2l-fun-ce.py