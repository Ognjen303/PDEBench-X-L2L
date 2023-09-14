#!/bin/bash
#SBATCH --account=raise-ctp2           # Who pays?
#SBATCH --nodes=1                        # How many compute nodes
#SBATCH --job-name=pdebench
#SBATCH --ntasks-per-node=1              # How many mpi processes/node
#SBATCH --cpus-per-task=128                # How many cpus per mpi proc
#SBATCH --output=../outputs/performance/out_1D_Burgers_Sols_Nu1.0_Unet.%j # Where to write results
#SBATCH --error=../errors/performance/err_1D_Burgers_Sols_Nu1.0_Unet.%j
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00          # For how long can it run?
#SBATCH --partition=dc-gpu-devel         # Machine partition

ml --force purge
ml Stages/2023 GCC/11.3.0 Python/3.10.4 CUDA/11.7


source /p/scratch/raise-ctp2/stefanovic1/pdebench-daad/venv/bin/activate

export PYTHONPATH="${PYTHONPATH}:/p/scratch/raise-ctp2/stefanovic1/pdebench-daad/"
export HYDRA_FULL_ERROR=1

CUDA_VISIBLE_DEVICES='0' python3 -u ../train_models_forward.py +args=config_Bgs.yaml ++args.model_name='Unet' ++args.filename='../../data/1D/Burgers/Train/1D_Burgers_Sols_Nu1.0.hdf5' ++args.if_training=False ++args.model_to_load='../../data/1D/Burgers/Train/1D_Burgers_Sols_Nu1.0_Unet-PF-2012152941.pt'

CUDA_VISIBLE_DEVICES='0' python3 -u ../train_models_forward.py +args=config_Bgs.yaml ++args.model_name='Unet' ++args.filename='../../data/1D/Burgers/Train/1D_Burgers_Sols_Nu1.0.hdf5' ++args.if_training=False ++args.model_to_load='../../data/1D/Burgers/Train/1D_Burgers_Sols_Nu1.0_Unet-PF-2012152942.pt'

CUDA_VISIBLE_DEVICES='0' python3 -u ../train_models_forward.py +args=config_Bgs.yaml ++args.model_name='Unet' ++args.filename='../../data/1D/Burgers/Train/1D_Burgers_Sols_Nu1.0.hdf5' ++args.if_training=False ++args.model_to_load='../../data/1D/Burgers/Train/1D_Burgers_Sols_Nu1.0_Unet-PF-2012152943.pt'