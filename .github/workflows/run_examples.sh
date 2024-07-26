#!/bin/bash
set -ev 

CONDA_ENV=dibasis
PYTHON=python

export HIPPYLIB_PATH=$(pwd)/hippylib
export HIPPYFLOW_PATH=$(pwd)/hippyflow

# This is for running MPI on root in the docker container 
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Run poisson example
cd examples/poisson

conda init 
source ~/.bashrc
conda activate $CONDA_ENV

$PYTHON driver_poisson_mean.py

mpirun -n 2 --oversubscribe python driver_poisson_mean.py

