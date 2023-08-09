#!/bin/bash

export HIPPYLIB_PATH=$(pwd)/hippylib

# Run poisson example
cd examples/poisson

python3 driver_poisson_mean.py

mpirun -n 2 python3 driver_poisson_mean.py

