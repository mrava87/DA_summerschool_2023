#!/bin/bash
# 
# Installer for da2023 CPU environment
# 
# Run: ./install_env_cpu.sh
# 
# M. Ravasi, 11/06/2023

echo 'Creating da2023 CPU environment'

# create conda env
#CONDA_SUBDIR=osx-64 conda env create -f environment_cpu.yml
conda env create -f environment_cpu.yml
source  $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate da2023cpu
conda env list
echo 'Created and activated environment:' $(which python)

# check torch work as expected
echo 'Checking torch version and running a command...'
python -c 'import torch; print(torch.__version__);  print(torch.ones(10))'

echo 'Done!'

