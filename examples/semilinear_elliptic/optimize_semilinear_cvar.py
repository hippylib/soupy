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
sys.path.append(os.environ.get('SOUPY_PATH'))

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
    parser = argparse.ArgumentParser(description="Solving OUU problem")
    parser.add_argument('-t', '--target', type=str, default="sinusoid", help="target case")
    parser.add_argument('-p', '--param', type=float, default=1.0, help="Parameter for target definition")
    parser.add_argument('-b', '--beta', type=float, default=0.95, help="CVaR percentile value")
    parser.add_argument('--epsilon', type=float, default=1e-4, help="CVaR approximation parameter")

    parser.add_argument('-N', '--N_sample', type=int, default=16, help="Number of samples for SD")
    parser.add_argument('-n', '--max_iter', type=int, default=100, help="Maximum number of SD iterations")
    parser.add_argument('-e', '--N_eval', type=int, default=100, help="Number of samples for evaluation of optimum")

    parser.add_argument('--nx', type=int, default=64, help="Number of elements in x direction")
    parser.add_argument('--ny', type=int, default=64, help="Number of elements in y direction")
    parser.add_argument('--N_sources', type=int, default=7, help="Number of sources per side")
    parser.add_argument('--loc_lower', type=float, default=0.1, help="Lower bound of location of sources")
    parser.add_argument('--loc_upper', type=float, default=0.9, help="Upper bound of location of sources")
    parser.add_argument('--well_width', type=float, default=0.08, help="Upper bound of location of sources")

    parser.add_argument('--display', default=False, action="store_true", help="Display optimization iterations")
    parser.add_argument('--postprocess', default=False, action="store_true", help="Upper bound of location of sources")
    parser.add_argument('--seed', type=int, default=0, help="Random seed for sampling the initial guess and random parameter")
    args = parser.parse_args()
        
    # MPI 
    comm_mesh = MPI.COMM_SELF
    comm_sampler = MPI.COMM_WORLD
    comm_rank_sampler = comm_sampler.Get_rank()
    comm_size_sampler = comm_sampler.Get_size()
    print_on_root("Split communicators", comm_sampler)


    # ----------------- Initialize parameters ----------------- # 
    # Set parsed parameters

    MODEL = "PDE" 

    # control problem parameters
    semilinear_elliptic_settings = semilinear_elliptic_control_settings()
    semilinear_elliptic_settings['nx'] = args.nx
    semilinear_elliptic_settings['ny'] = args.ny
    semilinear_elliptic_settings['well_width'] = args.well_width
    semilinear_elliptic_settings['loc_lower'] = args.loc_lower
    semilinear_elliptic_settings['loc_upper'] = args.loc_upper
    semilinear_elliptic_settings['n_wells_per_side'] = args.N_sources

    #  Parse save directory 
    save_dir = "results_cvar_eps%g/%s_mesh%dx%d_sources%d_%gto%g_width%g_target_%s_p%g_beta%g_SAA_ns%d_maxiter%d" %(args.epsilon,
        MODEL, args.nx, args.ny, 
        args.N_sources, args.loc_lower, args.loc_upper, args.well_width,
        args.target, args.param, 
        args.beta, args.N_sample,  args.max_iter)

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
    qoi = soupy.L2MisfitControlQoI(mesh, Vh, u_target.vector())
    control_model = soupy.ControlModel(pde, qoi)

    print_on_root("Making SAA risk measure and cost", comm_sampler)
    rm_param = soupy.superquantileRiskMeasureSAASettings()
    rm_param["beta"] = args.beta
    rm_param["sample_size"] = args.N_sample
    rm_param["seed"] = seed 
    rm_param["epsilon"] = 1e-4
    pde_rm = soupy.SuperquantileRiskMeasureSAA_MPI(control_model, prior, settings=rm_param, comm_sampler=comm_sampler)
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
    z0 = pde_cost.generate_vector(soupy.CONTROL)
    z0_np = z0.get_local()

    # Flush the out buffer before starting optimization 
    sys.stdout.flush()

    tsolve_0 = time.time()
    results = scipy.optimize.minimize(scipy_cost.function(), z0_np, method="L-BFGS-B", jac=scipy_cost.jac(), bounds=box_bounds, options=opt_options)
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
    
    z0.set_local(zt_opt_np)
    pde_rm.computeComponents(z0)
    risk_opt = pde_rm.cost()
    cvar_opt = pde_rm.superquantile()

    # -----------------  Post processing ----------------- #
    if comm_rank_sampler == 0:
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
        np.save("%s/z_opt.npy" %(save_dir), z_opt_np)
