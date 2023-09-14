#!/bin/bash
#SBATCH --account=raise-ctp2           # Who pays?
#SBATCH --nodes=1                        # How many compute nodes
#SBATCH --job-name=pdebench
#SBATCH --ntasks-per-node=1              # How many mpi processes/node
#SBATCH --cpus-per-task=128                # How many cpus per mpi proc
#SBATCH --output=../outputs/performance/out_1D_adv_beta4.0_PINN_training.%j        # Where to write results
#SBATCH --error=../errors/performance/err_1D_adv_beta4.0_PINN_training.%j
#SBATCH --gres=gpu:4
#SBATCH --time=00:45:00          # For how long can it run?
#SBATCH --partition=dc-gpu         # Machine partition

ml --force purge
ml Stages/2023 GCC/11.3.0 Python/3.10.4 CUDA/11.7 TensorFlow/2.11.0-CUDA-11.7

source /p/scratch/raise-ctp2/stefanovic1/pdebench-daad/venv/bin/activate

export PYTHONPATH="${PYTHONPATH}:/p/scratch/raise-ctp2/stefanovic1/pdebench-daad/"
export DDEBACKEND="pytorch"
export HYDRA_FULL_ERROR=1

echo $PYTHONPATH

CUDA_VISIBLE_DEVICES='0' python3 -u ../train_models_forward.py +args=config_pinn_pde1d.yaml ++args.model_name='PINN' ++args.filename='1D_Advection_Sols_beta4.0.hdf5' ++args.root_path='../../data/1D/Advection/Train/' ++args.save_path='../newly_trained_models/Advection/PINN/'
