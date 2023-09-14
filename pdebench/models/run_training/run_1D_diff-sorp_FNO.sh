#!/bin/bash
#SBATCH --account=raise-ctp2           # Who pays?
#SBATCH --nodes=1                        # How many compute nodes
#SBATCH --job-name=pdebench
#SBATCH --ntasks-per-node=1              # How many mpi processes/node
#SBATCH --cpus-per-task=128                # How many cpus per mpi proc
#SBATCH --output=../outputs/out_1D_diff-sorp_FNO.%j # Where to write results
#SBATCH --error=../errors/err_1D_diff-sorp_FNO.%j
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00          # For how long can it run?
#SBATCH --partition=dc-gpu-devel         # Machine partition

ml --force purge
ml Stages/2023 GCC/11.3.0 Python/3.10.4 CUDA/11.7


source /p/scratch/raise-ctp2/stefanovic1/pdebench-daad/venv/bin/activate

export PYTHONPATH="${PYTHONPATH}:/p/scratch/raise-ctp2/stefanovic1/pdebench-daad/"
export HYDRA_FULL_ERROR=1

CUDA_VISIBLE_DEVICES='0' python3 -u ../train_models_forward.py +args=config_diff-sorp.yaml ++args.model_name='FNO' ++args.data_path='../../data/1D/diffusion-sorption/Train/' ++args.filename='1D_diff-sorp_NA_NA' ++args.modes=16 ++args.width=64 ++args.if_training=True