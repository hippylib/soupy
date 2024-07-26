#!/bin/bash
set -ev 

CONDA_ENV=dibasis
PYTHON=python
export HIPPYLIB_PATH=$(pwd)/hippylib

# This is for running MPI on root in the docker container 
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Run the code
cd soupy/test

conda init 
source ~/.bashrc
conda activate $CONDA_ENV

$PYTHON -m unittest discover -v
$PYTHON -m unittest discover -v -p 'ptest_*'

mpirun -n 2 --oversubscribe $PYTHON ptest_meanVarRiskMeasureSAA.py
mpirun -n 2 --oversubscribe $PYTHON ptest_scipyCostWrapper.py
mpirun -n 2 --oversubscribe $PYTHON ptest_superquantileSAA.py
mpirun -n 2 --oversubscribe $PYTHON ptest_transformedMeanRiskMeasureSAA.py

