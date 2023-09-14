#!/bin/bash
#SBATCH --account=raise-ctp2           
#SBATCH --nodes=4                        
#SBATCH --job-name=pdebench
#SBATCH --ntasks-per-node=1              
#SBATCH --cpus-per-task=128               
#SBATCH --output=../outputs/Unet_epochs500_lr_0.01_pdebench_baseline.%j        
#SBATCH --error=../errors/Unet_epochs500_lr_0.01_pdebench_baseline.%j
#SBATCH --gres=gpu:4
#SBATCH --time=00:55:00         
#SBATCH --partition=dc-gpu-devel

ml --force purge
ml Stages/2023 GCC/11.3.0 Python/3.10.4 CUDA/11.7


source /p/scratch/raise-ctp2/stefanovic1/pdebench-daad/venv/bin/activate

export PYTHONPATH="${PYTHONPATH}:/p/scratch/raise-ctp2/stefanovic1/pdebench-daad/"
export HYDRA_FULL_ERROR=1

CUDA_VISIBLE_DEVICES='0' python3 -u ../train_models_forward.py +args=config_Adv.yaml ++args.model_name='Unet' ++args.filename='../../data/1D/Advection/Train/1D_Advection_Sols_beta4.0.hdf5' ++args.ar_mode=True ++args.pushforward=True ++args.unroll_step=20 ++args.if_training=True ++args.epochs=500 ++args.learning_rate=1e-2