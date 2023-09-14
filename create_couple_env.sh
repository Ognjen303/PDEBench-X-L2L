ml --force purge

ml Stages/2023 GCCcore/.11.3.0 SciPy-Stack/2022a

ml Stages/2023 GCC/11.3.0 Python/3.10.4 CUDA/11.7
# ml Stages/2023 GCC/11.3.0 SciPy-Stack/2022a OpenMPI/4.1.4 PyTorch/1.12.0-CUDA-11.7 torchvision/0.13.1-CUDA-11.7

python -m venv ../couple_env

. ../couple_env/bin/activate

cd ${PWD}/L2L

python -m pip install -e .

cd ${PWD}/..

pip3 install --upgrade pip wheel
pip3 install -r requirements.txt
pip3 install .
pip3 install torch
pip3 install torchvision
pip3 install hydra-core

export PYTHONPATH="${PYTHONPATH}:${PWD}"

deactivate couple_env

