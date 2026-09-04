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
Minimization of the CVaR risk measure using the SAA approximation for a 
semi-linear elliptic PDE with a log-normal conductivity field. Objective 
function is the L2 misfit between the state and a target state. The example is 
taken from the paper: 

Dingcheng Luo, Thomas O'Leary, Peng Chen, and Omar Ghattas,
Efficient PDE-Constrained optimization under high-dimensional uncertainty using 
derivative-informed neural operators. Luo, O'Leary-Roseberry, Chen, Ghattas, 
to appear.

This example shows how to implement a finite dimensional optimization variable.
In this case, the control variable is a vector representing the strengths of 
a grid of localized sources. See :code:`semilinearEllipticControlPDE.py` for
more details.

This example also shows how to use the :code:`ScipyCostWrapper` to convert
a :code:`ControlCostFunctional` to be compatible with :code:`scipy.optimize`

This driver supports MPI to parallelize the sampling of the parameter field.
e.g. 

mpirun -n 4 python driver_semilinear_cvar.py
"""

import random 
import time
import os, sys
import pickle
import argparse

import scipy.optimize
import numpy as np 
import matplotlib.pyplot as plt
import dolfin as dl
from mpi4py import MPI

sys.path.append(os.environ.get('HIPPYLIB_PATH'))
sys.path.append('../../')

import hippylib as hp
import soupy

from semilinearEllipticControlPDE import setup_semilinear_elliptic_pde, \
        semilinear_elliptic_control_settings

from semilinearEllipticOUU import get_target, plot_sources

dl.set_log_active(False)

def print_on_root(print_str, mpi_comm=MPI.COMM_WORLD):
    if mpi_comm.Get_rank() == 0:
        print(print_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', type=str, default="sinusoid", help="Target case")
    parser.add_argument('-p', '--param', type=float, default=1.0, help="Parameter for target definition")
    parser.add_argument('-b', '--beta', type=float, default=0.95, help="CVaR percentile value")
    parser.add_argument('-e', '--epsilon', type=float, default=1e-4, help="CVaR smoothing parameter")

    parser.add_argument('-N', '--N_sample', type=int, default=16, help="Number of samples for SAA")
    parser.add_argument('-n', '--max_iter', type=int, default=100, help="Maximum number of optimization iterations")

    parser.add_argument('--nx', type=int, default=64, help="Number of elements in x direction")
    parser.add_argument('--ny', type=int, default=64, help="Number of elements in y direction")
    parser.add_argument('--N_sources', type=int, default=7, help="Number of sources per side")
    parser.add_argument('--loc_lower', type=float, default=0.1, help="Lower bound of location of sources")
    parser.add_argument('--loc_upper', type=float, default=0.9, help="Upper bound of location of sources")
    parser.add_argument('--well_width', type=float, default=0.08, help="Width of sources")

    parser.add_argument('--display', default=False, action="store_true", help="Display optimization iterations")
    parser.add_argument('--seed', type=int, default=0, help="Random seed for sampling the random parameter")
    args = parser.parse_args()
        
    # MPI 
    comm_mesh = MPI.COMM_SELF
    comm_sampler = MPI.COMM_WORLD
    comm_rank_sampler = comm_sampler.Get_rank()
    comm_size_sampler = comm_sampler.Get_size()
    print_on_root("Make communicators", comm_sampler)


    # ----------------- Initialize parameters ----------------- # 
    # Set parsed parameters


    # control problem parameters
    semilinear_elliptic_settings = semilinear_elliptic_control_settings()
    semilinear_elliptic_settings['nx'] = args.nx
    semilinear_elliptic_settings['ny'] = args.ny
    semilinear_elliptic_settings['well_width'] = args.well_width
    semilinear_elliptic_settings['loc_lower'] = args.loc_lower
    semilinear_elliptic_settings['loc_upper'] = args.loc_upper
    semilinear_elliptic_settings['n_wells_per_side'] = args.N_sources

    #  Save directory 
    save_dir = "results/PDE_mesh%dx%d_sources%d_%gto%g_width%g_target_%s_p%g_cvar_beta%g_eps%g_SAA_ns%d_maxiter%d" %(args.nx, args.ny, 
        args.N_sources, args.loc_lower, args.loc_upper, args.well_width,
        args.target, args.param, 
        args.beta, args.epsilon,
        args.N_sample, args.max_iter)

    if args.seed > 0:
        # Do the sampling randomly 
        save_dir += "_seed%d" %(args.seed)
        random.seed(1)
        for i in range(args.seed):
            seed = random.randrange(100_000_000)
    else:
        seed = 1 

    if comm_rank_sampler == 0:
        print("Make directory: ", save_dir)
        os.makedirs(save_dir, exist_ok=True)

    # ----------------- Make problem ----------------- # 
    print_on_root("Making semilinear elliptic problem", comm_sampler)
    mesh, pde, Vh, prior = setup_semilinear_elliptic_pde(semilinear_elliptic_settings, comm_mesh=comm_mesh)

    print_on_root("Making target, QoI, and problem", comm_sampler)
    u_target_expr = get_target(args.target, args.param)
    u_target = dl.interpolate(u_target_expr, Vh[hp.STATE])
    qoi = soupy.L2MisfitControlQoI(Vh, u_target.vector())
    control_model = soupy.ControlModel(pde, qoi)

    print_on_root("Making SAA risk measure and cost", comm_sampler)
    rm_param = soupy.superquantileRiskMeasureSAASettings()
    rm_param["beta"] = args.beta
    rm_param["sample_size"] = args.N_sample
    rm_param["seed"] = seed 
    rm_param["epsilon"] = 1e-4
    pde_rm = soupy.SuperquantileRiskMeasureSAA(control_model, prior, settings=rm_param, comm_sampler=comm_sampler)
    pde_cost = soupy.RiskMeasureControlCostFunctional(pde_rm, None)

    # Scipy cost 
    print_on_root("Convert to scipy cost")
    scipy_cost = soupy.ScipyCostWrapper(pde_cost)

    # Box constraint with CVaR. t is unconstrained 
    dim = semilinear_elliptic_settings["n_wells_per_side"]**2
    lb = np.ones(dim) * semilinear_elliptic_settings["strength_lower"]
    ub = np.ones(dim) * semilinear_elliptic_settings["strength_upper"]
    lb = np.append(lb, np.array([-np.infty]))
    ub = np.append(ub, np.array([np.infty]))
    box_bounds = scipy.optimize.Bounds(lb=lb, ub=ub)

    # ----------------- Optimize ----------------- # 
    print_on_root("Start optimization")
    opt_options = {"maxiter" : args.max_iter, "disp" : args.display} 

    # Generate the initial guess array as zeros. 
    zt = pde_cost.generate_vector(soupy.CONTROL)
    zt0_np = zt.get_local()

    # Flush the out buffer before starting optimization 
    sys.stdout.flush()

    tsolve_0 = time.time()
    results = scipy.optimize.minimize(scipy_cost.function(), zt0_np, method="L-BFGS-B", jac=scipy_cost.jac(), bounds=box_bounds, options=opt_options)
    tsolve_1 = time.time()
    tsolve = tsolve_1 - tsolve_0

    zt_opt_np = results["x"]
    z_opt_np = zt_opt_np[:-1]
    print_on_root("Time to solve: %g" %tsolve)
    
    n_linear_solves_proc = np.array([pde.n_linear_solves], dtype=np.int32)
    n_linear_solves_total = np.zeros_like(n_linear_solves_proc)
    comm_sampler.Reduce(n_linear_solves_proc, n_linear_solves_total, root=0)

    print_on_root("Number of function evals: %d" %(scipy_cost.n_func))
    print_on_root("Number of gradient evals: %d" %(scipy_cost.n_grad))
    print_on_root("Number of linear PDE solves (single proc): %d" %(pde.n_linear_solves))
    print_on_root("Number of linear PDE solves (all procs): %d" %(n_linear_solves_total[0]))
    
    zt.set_local(zt_opt_np)
    pde_rm.computeComponents(zt)
    risk_opt = pde_rm.cost()
    cvar_opt = pde_rm.superquantile()

    # -----------------  Post processing ----------------- #
    inds_all = [soupy.STATE, soupy.PARAMETER, soupy.ADJOINT, soupy.CONTROL]
    x_fun = [dl.Function(Vh[ind]) for ind in inds_all]
    x = [x_fun[ind].vector() for ind in inds_all]

    # Set parameter
    noise = dl.Vector(comm_mesh)
    prior.init_vector(noise, "noise")
    hp.parRandom.normal(1.0, noise)
    prior.sample(noise, x[soupy.PARAMETER])

    # Set control 
    x[soupy.CONTROL].axpy(1.0, zt.get_vector())

    # Solve at optimal 
    control_model.solveFwd(x[soupy.STATE], x)

    if comm_rank_sampler == 0:
        np.save("%s/z_opt.npy" %(save_dir), z_opt_np)

        with open("%s/results.p" %(save_dir), "wb") as results_file:
            pickle.dump(results, results_file)

        costs = dict() 
        costs["n_func"] = scipy_cost.n_func 
        costs["n_grad"] = scipy_cost.n_grad
        costs["n_linear_solves_total"] = n_linear_solves_total[0] 
        costs["n_linear_solves_proc"] = n_linear_solves_proc[0]
        costs["risk_opt"] = risk_opt
        costs["cvar_opt"] = cvar_opt

        with open("%s/costs.p" %(save_dir), "wb") as costs_file:
            pickle.dump(costs, costs_file)

        plot_sources(z_opt_np, semilinear_elliptic_settings["n_wells_per_side"], 
                     semilinear_elliptic_settings["loc_lower"], semilinear_elliptic_settings["loc_upper"])
        plt.savefig("%s/control.png" %(save_dir))

        plt.figure()
        hp.nb.plot(u_target, mytitle="Target state")
        plt.savefig("%s/target.png" %(save_dir))
        
        plt.figure()
        hp.nb.plot(x_fun[soupy.STATE], mytitle="Sample state at optimal control")
        plt.savefig("%s/state.png" %(save_dir))
            
        plt.figure()
        hp.nb.plot(x_fun[soupy.PARAMETER], mytitle="Sample parameter")
        plt.savefig("%s/parameter.png" %(save_dir))

