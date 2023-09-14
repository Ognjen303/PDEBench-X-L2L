ml --force purge
ml Stages/2023 GCC/11.3.0 Python/3.10.4 CUDA/11.7
python3 -m venv ./venv --prompt pde_benchmark --system-site-packages
. ./venv/bin/activate
pip3 install --upgrade pip wheel
pip3 install -r requirements.txt
pip3 install .
pip3 install torch
pip3 install torchvision
pip3 install hydra-core


## ModuleNotFoundError No module named 'pdebench.models'

## HOW TO FIX THIS ERROR: 

## GO TO YOUR HOME FOLDER (WHICH YOU GET TO BY JUST TYPING CD). THEN OPEN .bashrc
## and add there the following line

export PYTHONPATH="${PYTHONPATH}:/p/scratch/raise-ctp2/stefanovic1/pdebench-daad"


## If it still doesn't work, then 'by-hand' type the following in console:

## $ export PYTHONPATH="${PYTHONPATH}:/p/scratch/raise-ctp2/stefanovic1/PDEBench" 
