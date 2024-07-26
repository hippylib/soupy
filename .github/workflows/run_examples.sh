#!/bin/bash

export HIPPYLIB_PATH=$(pwd)/hippylib

# Run poisson example
cd examples/poisson

python driver_poisson_mean.py

mpirun -n 2 python driver_poisson_mean.py

