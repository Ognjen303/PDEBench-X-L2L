#!/bin/bash

# shellcheck disable=SC2206
#SBATCH --job-name=PINN_l2lpdeUnet
#SBATCH --account=raise-ctp2
#SBATCH --output=./outputs/PINN_l2lpdeUnet.%j
#SBATCH --error=./errors/PINN_l2lpdeUnet.%j
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=128
##SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

ml --force purge

ml Stages/2023 GCCcore/.11.3.0 SciPy-Stack/2022a
ml Stages/2023 GCC/11.3.0 Python/3.10.4 CUDA/11.7

## ml GCC/11.3.0 SciPy-Stack/2022a OpenMPI/4.1.4 PyTorch/1.12.0-CUDA-11.7 torchvision/0.13.1-CUDA-11.7 
source /p/scratch/raise-ctp2/stefanovic1/couple_env/bin/activate

## TensorFlow/2.11.0-CUDA-11.7
## export CUDA_VISIBLE_DEVICES=0

export DDEBACKEND="pytorch"

COMMAND="l2l-pdebench-es.py --lr 0.005 --pop-size 7 --n-iteration 2 --noise-std 0.001 --epochs 10 --batch-size 1000 --optimizer-lr 0.000001 --config config_Adv.yaml"
echo $COMMAND

python $COMMAND