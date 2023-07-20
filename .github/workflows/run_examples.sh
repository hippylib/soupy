#!/bin/bash

export HIPPYLIB_PATH=$(pwd)/hippylib

# Run poisson example
cd examples/poisson
python driver_poisson_mean_serial.py
mpirun -n 4 python driver_poisson_mean_mpi.py
python compare_results.py

# Run hyperelasticity example 
cd ../hyperelasticity
python driver_hyperelasticity_deterministic.py

cd ../semilinear_elliptic
python driver_semilinear_cvar.py -N 4
