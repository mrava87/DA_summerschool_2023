#!/bin/bash
# 
# Installer for da2023 environment
# 
# Run: ./install_env.sh
# 
# M. Ravasi, 11/06/2023

echo 'Creating da2023 environment'

# create conda env
conda env create -f environment.yml
source  $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate da2023
conda env list
echo 'Created and activated environment:' $(which python)

# check cupy-torch work as expected
echo 'Checking cupy version and running a command...'
python -c 'import cupy as cp; print(cp.__version__); import torch; print(torch.__version__);  print(torch.cuda.get_device_name(torch.cuda.current_device())); print(torch.ones(10).to("cuda:0"))'

echo 'Done!'

