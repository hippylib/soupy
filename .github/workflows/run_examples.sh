#!/bin/bash

export HIPPYLIB_PATH=$(pwd)/hippylib

# Run poisson example
cd examples/poisson
python3 driver_poisson_mean_serial.py
mpirun -n 2 python3 driver_poisson_mean_mpi.py
python3 compare_results.py

# Run hyperelasticity example 
cd ../hyperelasticity
python3 driver_hyperelasticity_deterministic.py

cd ../semilinear_elliptic
python3 driver_semilinear_cvar.py -N 4
