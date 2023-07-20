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


"""
This example implements a deterministic optimization problem for material 
design in hyperelasticity. The objective is to minimize the compliance of
the structure subject. The design variable takes values in [0, 1], which 
selects between two different material properties. 

See :code:`setupHyperelasticityProblem.py` for problem settings, including
mesh, geometry, and solver properties. 

The example uses a custom forward solver. See 
:code:`hyperelasticityControlPDE.py` for the PDE definition and solver
"""


import pickle
import sys, os
import argparse 

import numpy as np
import matplotlib.pyplot as plt 
import dolfin as dl
import scipy.optimize 
from mpi4py import MPI 

sys.path.append(os.environ.get('HIPPYLIB_PATH'))
sys.path.append('../../')

import hippylib as hp 
import soupy

from setupHyperelasticityProblem import hyperelasticity_problem_settings, setup_hyperelasticity_problem


# Optimization options for the form compiler
dl.parameters["form_compiler"]["cpp_optimize"] = True
dl.parameters["form_compiler"]["quadrature_degree"] = 4
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

dl.set_log_active(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--maxiter', type=int, default=100, help="Maximum number of SD iterations")
    parser.add_argument('-q', '--qoi_type', type=str, default="stiffness", choices=["all", "stiffness", "point"])
    parser.add_argument('-p', '--penalization', type=float, default=1e-1, help="Scaling of penalization")
    parser.add_argument('--show', default=False, action="store_true", help="Show figure")
    args = parser.parse_args()

    # Create mesh and setup model components 
    comm_mesh = MPI.COMM_WORLD

    settings = hyperelasticity_problem_settings()
    settings["qoi_type"] = args.qoi_type
    mesh, Vh, hyperelasticity_varf, control_model, prior = setup_hyperelasticity_problem(settings, comm_mesh)
    
    l2_penalty = soupy.L2Penalization(Vh, args.penalization)
    pde_cost = soupy.DeterministicControlCostFunctional(control_model, prior, l2_penalty)

    scipy_cost = soupy.ScipyCostWrapper(pde_cost, verbose=True)
    box_bounds = scipy.optimize.Bounds(lb=0.0, ub=1.0)

    x = control_model.generate_vector()
    x[soupy.PARAMETER].axpy(1.0, prior.mean)
    control_model.solveFwd(x[soupy.STATE], x)
    disp_fun_init = hp.vector2Function(x[soupy.STATE], Vh[soupy.STATE])

    z0_np = x[soupy.CONTROL].get_local() + 0.5
    options = {'gtol' : 1e-12, 'disp' : True, 'maxiter' : args.maxiter}
    results = scipy.optimize.minimize(scipy_cost.function(), z0_np, 
            method="L-BFGS-B", jac=scipy_cost.jac(), 
            bounds=box_bounds, options=options)
    z_opt = results['x']     

    x[soupy.CONTROL].set_local(z_opt)
    x[soupy.CONTROL].apply("")

    control_model.solveFwd(x[soupy.STATE], x)
    z_fun = dl.Function(Vh[soupy.CONTROL], x[soupy.CONTROL])

    with dl.HDF5File(mesh.mpi_comm(), "z_opt.h5", "w") as save_file:
        save_file.write(z_fun, "control")
    
    # ---------- postprocessing the plotting ----------- # 
    plt.figure()
    ax = dl.plot(disp_fun_init, mode="displacement")
    plt.colorbar(ax)
    plt.title("Displacement with soft material")
    plt.savefig("u_soft.png")

    plt.figure()
    ax = dl.plot(hp.vector2Function(x[soupy.CONTROL], Vh[soupy.CONTROL]))
    plt.colorbar(ax)
    plt.title("Optimal design")
    plt.savefig("z_opt.png")

    plt.figure()
    ax = dl.plot(hp.vector2Function(x[soupy.STATE], Vh[soupy.STATE]), mode="displacement")
    plt.colorbar(ax)
    plt.title("Optimal displacement")
    plt.savefig("u_opt.png")

    if args.show:
        plt.show()