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
# Software Foundation) version 3.0 dated June 2007.


"""
This example implements an optimization problem for material 
design of a hyperelastic beam. The objective is to minimize the compliance of
the structure subject to uncertain loading. 

The design variable takes values in [0, 1], which selects between 
two different material properties. A nominal external load 
is prescribed to be at the center of the beam, but is scaled by 
a multiplicative Gaussian random field.

- Set risk measure flag :code:`-r` to :code:`deterministic` 
    for Deterministic optimization using the mean parameter value 

- Set risk measure flag :code:`-r` to :code:`mean_var` for optimization of 
    the mean + variance risk measure 

- Flag :code:`beta` controls the variance weighting 

- Flag :code:`penalization` controls the l2 penalization on the design variable

See :code:`setupHyperelasticityProblem.py` for problem settings, including
mesh, geometry, and solver properties. 

The example uses a custom forward solver. See 
:code:`hyperelasticityControlPDE.py` for the PDE definition and solver

This example also shows how to use the :code:`ScipyCostWrapper` to convert
a :code:`ControlCostFunctional` to be compatible with :code:`scipy.optimize`

This driver supports MPI to parallelize the sampling of the parameter field.

To run deterministic: 
python driver_hyperelasticity.py


To run with mean + variance risk measure:
python driver_hyperelasticity.py -r mean_var

To run with mean + variance risk measure and parllel sampling (e.g.):
mpirun -n 4 python driver_hyperelasticity.py -r mean_var
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
dl.set_log_active(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--risk_measure', type=str, default="deterministic", choices=["deterministic", "mean_var"], help="Risk measure type")
    parser.add_argument('-n', '--sample_size', type=int, default=32, help="Sample size for risk measure computation")
    parser.add_argument('-b', '--beta', type=float, default=1.0, help="Variance weight for risk measure")
    parser.add_argument('-q', '--qoi_type', type=str, default="stiffness", choices=["all", "stiffness", "point"])
    parser.add_argument('-p', '--penalization', type=float, default=1e-1, help="Scaling of penalization")

    parser.add_argument('--maxiter', type=int, default=100, help="Maximum number of SD iterations")
    parser.add_argument('--show', default=False, action="store_true", help="Show figure")
    args = parser.parse_args()

    save_dir = "results_%s" %(args.risk_measure)
    os.makedirs(save_dir, exist_ok=True)

    # Create mesh and setup model components 
    comm_mesh = MPI.COMM_SELF
    comm_sampler = MPI.COMM_WORLD 

    settings = hyperelasticity_problem_settings()
    settings["qoi_type"] = args.qoi_type
    mesh, Vh, hyperelasticity_varf, control_model, prior = setup_hyperelasticity_problem(settings, comm_mesh)
    
    l2_penalty = soupy.L2Penalization(Vh, args.penalization)

    if args.risk_measure == "deterministic":
        pde_cost = soupy.DeterministicControlCostFunctional(control_model, prior, l2_penalty)

    else:
        # Use the mean variance risk measure to assemble cost 
        risk_settings = soupy.meanVarRiskMeasureSAASettings()
        risk_settings["beta"] = args.beta
        risk_settings["sample_size"] = args.sample_size
        risk_measure = soupy.MeanVarRiskMeasureSAA(control_model, prior, risk_settings, comm_sampler=comm_sampler)
        pde_cost = soupy.RiskMeasureControlCostFunctional(risk_measure, l2_penalty)

    # ------------------  Using the scipy cost interface ------------------ # 
    scipy_cost = soupy.ScipyCostWrapper(pde_cost, verbose=True)
    box_bounds = scipy.optimize.Bounds(lb=0.0, ub=1.0)

    # Solve the PDE for the beam using entirely soft material 
    x = control_model.generate_vector()
    x[soupy.PARAMETER].axpy(1.0, prior.mean)
    control_model.solveFwd(x[soupy.STATE], x)
    disp_fun_init = hp.vector2Function(x[soupy.STATE], Vh[soupy.STATE])

    # Solve optimal design problem using 0.5 as initial guess 
    z0_np = x[soupy.CONTROL].get_local() + 0.5
    options = {'gtol' : 1e-12, 'disp' : True, 'maxiter' : args.maxiter}
    results = scipy.optimize.minimize(scipy_cost.function(), z0_np, 
            method="L-BFGS-B", jac=scipy_cost.jac(), 
            bounds=box_bounds, options=options)

    # Optimal design 
    z_opt = results['x']     

    # --------------------------------------------------------------------- # 

    x[soupy.CONTROL].set_local(z_opt)
    x[soupy.CONTROL].apply("")

    control_model.solveFwd(x[soupy.STATE], x)
    z_fun = dl.Function(Vh[soupy.CONTROL], x[soupy.CONTROL])
    
    
    # ---------- postprocessing the plotting ----------- # 

    plt.figure()
    ax = dl.plot(disp_fun_init, mode="displacement")
    plt.colorbar(ax)
    plt.title("Displacement with soft material")
    plt.savefig("%s/u_soft.png" %(save_dir))

    plt.figure()
    ax = dl.plot(hp.vector2Function(x[soupy.CONTROL], Vh[soupy.CONTROL]))
    plt.colorbar(ax)
    plt.title("Optimal design")
    plt.savefig("%s/z_opt.png" %(save_dir))

    plt.figure()
    ax = dl.plot(hp.vector2Function(x[soupy.STATE], Vh[soupy.STATE]), mode="displacement")
    plt.colorbar(ax)
    plt.title("Optimal displacement")
    plt.savefig("%s/u_opt.png" %(save_dir))

    if comm_sampler.Get_rank() == 0:
        np.save('%s/z_opt.npy' %(save_dir), z_opt)

    if args.show:
        plt.show()
