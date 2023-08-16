#!/bin/bash
set -ev 

PYTHON=python3

export HIPPYLIB_PATH=$(pwd)/hippylib

# Run the code
cd soupy/test
$PYTHON -m unittest discover -v
$PYTHON -m unittest discover -v -p 'ptest_*'

mpirun -n 4 $PYTHON ptest_meanVarRiskMeasureSAA.py
mpirun -n 4 $PYTHON ptest_scipyCostWrapper.py
mpirun -n 4 $PYTHON ptest_superquantileSAA.py
mpirun -n 4 $PYTHON ptest_transformedMeanRiskMeasureSAA.py
