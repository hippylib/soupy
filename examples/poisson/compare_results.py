# Copyright (c) 2023, The University of Texas at Austin 
# & Georgia Institute of Technology
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the SOUPy package. For more information see
# https://github.com/hippylib/soupy/
#
# SOUPy is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 3.0 dated June 1991.


import sys 
import os 

import dolfin as dl  
import numpy as np 
import matplotlib.pyplot as plt 

sys.path.append(os.environ.get("HIPPYLIB_PATH"))
sys.path.append("../../")

import hippylib as hp 
import soupy 


def main():
    """
    Compares the solutions with/without mpi 
    """
    N_ELEMENTS_X = 32
    N_ELEMENTS_Y = 32 
    TOL = 1e-12

    RESULTS_DIRECTORY_MPI = "results_mpi"
    RESULTS_DIRECTORY_SERIAL = "results_serial"

    mesh = dl.UnitSquareMesh(N_ELEMENTS_X, N_ELEMENTS_Y)
    Vh_STATE = dl.FunctionSpace(mesh, "CG", 1)
    

    z_opt_no_mpi = dl.Function(Vh_STATE)
    with dl.HDF5File(mesh.mpi_comm(), "%s/z_opt.h5" %(RESULTS_DIRECTORY_SERIAL), "r") as load_file:
        load_file.read(z_opt_no_mpi, "control")

    z_opt_with_mpi = dl.Function(Vh_STATE)
    with dl.HDF5File(mesh.mpi_comm(), "%s/z_opt.h5" %(RESULTS_DIRECTORY_MPI), "r") as load_file:
        load_file.read(z_opt_with_mpi, "control")

    z_diff = z_opt_no_mpi.vector().copy()
    z_diff.axpy(-1.0, z_opt_with_mpi.vector())

    z_error = z_diff.inner(z_diff)
    print("Error between MPI and no MPI runs: %.3e" %(z_error))
    assert z_error < TOL 


if __name__ == "__main__":
    main()
