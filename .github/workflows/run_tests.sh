#!/bin/bash

export HIPPYLIB_PATH=$(pwd)/hippylib

# Run the code
cd soupy/test
python -m unittest 
