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
Hyperelastic beam design using Taylor approximations for risk-averse optimization.

This example uses Taylor approximations (constant, linear, quadratic) of the
QoI statistics instead of SAA for the risk measure computation.

Approximation methods:
- Taylor constant: Q(m) ≈ Q(m̄) [deterministic at prior mean, variance = 0]
- Taylor linear: First-order expansion, closed-form mean/variance
- Taylor quadratic: Second-order expansion with dominant Hessian modes

With --correction flag, MC correction is applied as control variate to reduce
approximation error. Taylor constant + MC correction = SAA (Sample Average Approx).

Usage examples:
    # Taylor constant (deterministic)
    python driver_hyperelasticity_taylor.py -a constant

    # Taylor linear with variance weight beta=1.0
    python driver_hyperelasticity_taylor.py -a linear --beta 1.0

    # Taylor quadratic with 10 dominant Hessian modes
    python driver_hyperelasticity_taylor.py -a quadratic --beta 1.0 --n-tr 10

    # Taylor linear with MC correction (16 samples)
    python driver_hyperelasticity_taylor.py -a linear --correction --num-mc 16

    # Taylor quadratic with MC correction
    python driver_hyperelasticity_taylor.py -a quadratic --beta 1.0 --n-tr 10 --correction --num-mc 16

    # Taylor constant + MC = SAA
    python driver_hyperelasticity_taylor.py -a constant --correction --num-mc 32

    # Full options with verbose output
    python driver_hyperelasticity_taylor.py -a quadratic --beta 2.0 --n-tr 20 \\
        --correction --num-mc 16 --maxiter 50 --print-every 5 -v

Arguments:
    -a, --approximation : Taylor order {constant, linear, quadratic}
    -b, --beta          : Variance weight for risk measure (default: 1.0)
    --n-tr              : Number of dominant Hessian modes for quadratic (default: 10)
    --correction        : Enable Monte Carlo correction
    --num-mc            : Number of MC samples for correction (default: 16)
    -q, --qoi_type      : QoI type {all, stiffness, point} (default: stiffness)
    -p, --penalization  : Penalization scaling (default: 0.1)
    --maxiter           : Max L-BFGS-B iterations (default: 100)
    --print-every       : Print iteration info every N iterations (default: 1)
    --show              : Show matplotlib figures
    -v, --verbose       : Verbose output from PDE solvers
"""

import pickle
import sys
import os
import argparse
import logging

# Configure macOS compiler BEFORE importing dolfin
_soupy_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, os.path.join(_soupy_root, "soupy", "utils"))
try:
    from macos_config import configure_macos_compiler, configure_dolfin_form_compiler
    configure_macos_compiler()
except ImportError:
    pass
sys.path.pop(0)

import numpy as np
import matplotlib.pyplot as plt
import dolfin as dl
import scipy.optimize
from mpi4py import MPI

sys.path.append(os.environ.get('HIPPYLIB_PATH'))
sys.path.append('../../')

import hippylib as hp
import soupy

from soupy.approximations.taylor import (
    TaylorConstantControlCostFunctional,
    TaylorLinearControlCostFunctional,
    TaylorQuadraticControlCostFunctional,
)

from setupHyperelasticityProblem import hyperelasticity_problem_settings, setup_hyperelasticity_problem

# Suppress verbose FFC/UFL logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)

# Optimization options for the form compiler
dl.parameters["form_compiler"]["cpp_optimize"] = True
dl.set_log_active(False)

try:
    configure_dolfin_form_compiler(dl)
except:
    pass


class QuietScipyCostWrapper:
    """A quieter version of ScipyCostWrapper that prints iteration summary."""

    def __init__(self, cost_functional, comm_rank=0, print_every=1):
        self.cost_functional = cost_functional
        self.comm_rank = comm_rank
        self.print_every = print_every
        self.iter_count = 0
        self.cost_history = []
        self.grad_norm_history = []
        self._z = cost_functional.generate_vector(soupy.CONTROL)
        self._g = cost_functional.generate_vector(soupy.CONTROL)

    def function(self):
        def f(z_np):
            self._z.set_local(z_np)
            self._z.apply("")
            cost = self.cost_functional.cost(self._z, order=0)
            self.cost_history.append(cost)
            return cost
        return f

    def jac(self):
        def g(z_np):
            self._z.set_local(z_np)
            self._z.apply("")
            self.cost_functional.cost(self._z, order=1)
            grad_norm = self.cost_functional.grad(self._g)
            self.grad_norm_history.append(grad_norm)
            self.iter_count += 1
            if self.comm_rank == 0 and self.iter_count % self.print_every == 0:
                print(f"  Iter {self.iter_count:4d}: cost = {self.cost_history[-1]:.6e}, ||grad|| = {grad_norm:.6e}")
            return self._g.get_local()
        return g


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperelasticity with Taylor approximations")
    parser.add_argument('-a', '--approximation', type=str, default="linear",
                        choices=["constant", "linear", "quadratic"],
                        help="Taylor approximation order")
    parser.add_argument('-b', '--beta', type=float, default=1.0,
                        help="Variance weight for risk measure")
    parser.add_argument('--n-tr', type=int, default=10,
                        help="Number of dominant Hessian modes (quadratic only)")
    parser.add_argument('--correction', action='store_true',
                        help="Enable Monte Carlo correction")
    parser.add_argument('--num-mc', type=int, default=16,
                        help="Number of MC samples for correction")
    parser.add_argument('-q', '--qoi_type', type=str, default="stiffness",
                        choices=["all", "stiffness", "point"])
    parser.add_argument('-p', '--penalization', type=float, default=1e-1,
                        help="Scaling of penalization")
    parser.add_argument('--maxiter', type=int, default=100,
                        help="Maximum number of L-BFGS-B iterations")
    parser.add_argument('--show', default=False, action="store_true",
                        help="Show figure")
    parser.add_argument('-v', '--verbose', default=False, action="store_true",
                        help="Verbose output")
    parser.add_argument('--print-every', type=int, default=1,
                        help="Print iteration info every N iterations (default: 1)")
    args = parser.parse_args()

    save_dir = f"results_taylor_{args.approximation}"
    if args.correction:
        save_dir += "_mc"
    os.makedirs(save_dir, exist_ok=True)

    # Create mesh and setup model components
    comm_mesh = MPI.COMM_SELF
    comm_sampler = MPI.COMM_WORLD
    rank = comm_sampler.Get_rank()

    if rank == 0:
        print("=" * 70)
        print("Hyperelasticity Optimal Design with Taylor Approximation")
        print("=" * 70)
        print(f"  Approximation:   Taylor {args.approximation}")
        if args.approximation == "quadratic":
            print(f"  N_tr (modes):    {args.n_tr}")
        print(f"  Beta (var wt):   {args.beta}")
        if args.correction:
            print(f"  MC correction:   Yes, N_mc={args.num_mc}")
        else:
            print(f"  MC correction:   No")
        print(f"  QoI type:        {args.qoi_type}")
        print(f"  Penalization:    {args.penalization}")
        print(f"  Max iterations:  {args.maxiter}")
        print("=" * 70)
        sys.stdout.flush()

    settings = hyperelasticity_problem_settings()
    settings["qoi_type"] = args.qoi_type

    if rank == 0:
        print("\nSetting up problem...")
        sys.stdout.flush()

    mesh, Vh, hyperelasticity_varf, control_model, prior = setup_hyperelasticity_problem(settings, comm_mesh)

    if rank == 0:
        print(f"  Mesh: {mesh.num_cells()} cells, {mesh.num_vertices()} vertices")
        print(f"  Control DOFs: {Vh[soupy.CONTROL].dim()}")
        sys.stdout.flush()

    l2_penalty = soupy.L2Penalization(Vh, args.penalization)

    # Create Taylor approximation cost functional
    if rank == 0:
        print(f"\nCreating Taylor {args.approximation} cost functional...")
        sys.stdout.flush()

    if args.approximation == "constant":
        taylor_settings = {
            "beta": args.beta,
            "correction": args.correction,
            "N_mc": args.num_mc,
            "verbose": args.verbose,
        }
        pde_cost = TaylorConstantControlCostFunctional(
            control_model, prior, l2_penalty, taylor_settings
        )

    elif args.approximation == "linear":
        taylor_settings = {
            "beta": args.beta,
            "correction": args.correction,
            "N_mc": args.num_mc,
            "verbose": args.verbose,
        }
        pde_cost = TaylorLinearControlCostFunctional(
            control_model, prior, l2_penalty, taylor_settings
        )

    elif args.approximation == "quadratic":
        taylor_settings = {
            "N_tr": args.n_tr,
            "beta": args.beta,
            "correction": args.correction,
            "N_mc": args.num_mc,
            "verbose": args.verbose,
        }
        pde_cost = TaylorQuadraticControlCostFunctional(
            control_model, prior, l2_penalty, taylor_settings
        )

    # Create scipy cost wrapper
    scipy_cost = QuietScipyCostWrapper(pde_cost, comm_rank=rank, print_every=args.print_every)
    box_bounds = scipy.optimize.Bounds(lb=0.0, ub=1.0)

    # Solve the PDE for the beam using entirely soft material
    x = control_model.generate_vector()
    x[soupy.PARAMETER].axpy(1.0, prior.mean)

    if rank == 0:
        print("\nSolving initial forward problem (soft material)...")
        sys.stdout.flush()

    control_model.solveFwd(x[soupy.STATE], x)
    disp_fun_init = hp.vector2Function(x[soupy.STATE], Vh[soupy.STATE])

    # Solve optimal design problem using 0.5 as initial guess
    z0_np = x[soupy.CONTROL].get_local() + 0.5

    if rank == 0:
        print("\nStarting optimization (L-BFGS-B)...")
        print("-" * 70)
        sys.stdout.flush()

    options = {'gtol': 1e-12, 'maxiter': args.maxiter}

    results = scipy.optimize.minimize(
        scipy_cost.function(), z0_np,
        method="L-BFGS-B", jac=scipy_cost.jac(),
        bounds=box_bounds, options=options
    )

    # Optimal design
    z_opt = results['x']

    x[soupy.CONTROL].set_local(z_opt)
    x[soupy.CONTROL].apply("")

    if rank == 0:
        print("\nSolving forward problem at optimal design...")
        sys.stdout.flush()

    control_model.solveFwd(x[soupy.STATE], x)
    z_fun = dl.Function(Vh[soupy.CONTROL], x[soupy.CONTROL])

    # Print optimization summary
    if rank == 0:
        print("-" * 70)
        print("Optimization Summary")
        print("-" * 70)
        print(f"  Converged:       {results['success']}")
        print(f"  Iterations:      {results['nit']}")
        print(f"  Function evals:  {results['nfev']}")
        print(f"  Final cost:      {results['fun']:.6e}")
        print(f"  ||grad||:        {np.linalg.norm(results['jac']):.6e}")
        print(f"  Design range:    [{z_opt.min():.4f}, {z_opt.max():.4f}]")

        # Print Taylor-specific statistics if available
        if args.approximation == "constant":
            cf = pde_cost
            print("-" * 70)
            print("Taylor Constant Statistics")
            print(f"  Q(m̄):            {cf.lin_mean:.6e}")
            if args.correction and args.num_mc > 0:
                saa_mean = np.mean(cf.Q_mc)
                saa_var = np.var(cf.Q_mc)
                print(f"  SAA Mean:        {saa_mean:.6e}")
                print(f"  SAA Variance:    {saa_var:.6e}")

        elif args.approximation == "linear":
            legacy = pde_cost._legacy
            taylor_mean = legacy.lin_mean
            taylor_var = legacy.lin_var - taylor_mean ** 2
            print("-" * 70)
            print("Taylor Linear Statistics")
            print(f"  Taylor Mean:     {taylor_mean:.6e}")
            print(f"  Taylor Variance: {taylor_var:.6e}")
            if args.correction and args.num_mc > 0:
                mean_diff = np.mean(legacy.lin_diff_mean)
                var_diff = np.mean(legacy.lin_diff_var)
                print(f"  Mean correction: {mean_diff:.6e}")
                print(f"  Var correction:  {var_diff:.6e}")
                print(f"  Corrected Mean:  {taylor_mean + mean_diff:.6e}")
                print(f"  Corrected Var:   {taylor_var + var_diff:.6e}")

        elif args.approximation == "quadratic":
            legacy = pde_cost._legacy
            print("-" * 70)
            print("Taylor Quadratic Statistics")
            print(f"  Quad Mean:       {legacy.quad_mean:.6e}")
            quad_var = legacy.quad_var - legacy.quad_mean ** 2
            print(f"  Quad Variance:   {quad_var:.6e}")
            print(f"  Eigenvalues:     {legacy.d[:min(5, len(legacy.d))]}")
            if args.correction and args.num_mc > 0:
                print(f"  Mean correction: {legacy.mean_diff:.6e}")
                print(f"  Var correction:  {legacy.var_diff:.6e}")

        print("=" * 70)
        sys.stdout.flush()

    # ---------- postprocessing the plotting ----------- #

    plt.figure()
    ax = dl.plot(disp_fun_init, mode="displacement")
    plt.colorbar(ax)
    plt.title("Displacement with soft material")
    plt.savefig(f"{save_dir}/u_soft.png")

    plt.figure()
    ax = dl.plot(hp.vector2Function(x[soupy.CONTROL], Vh[soupy.CONTROL]), vmin=0.0, vmax=1.0, extend='max')
    plt.colorbar(ax)
    plt.title(f"Optimal design (Taylor {args.approximation})")
    plt.savefig(f"{save_dir}/z_opt.png")

    plt.figure()
    ax = dl.plot(hp.vector2Function(x[soupy.STATE], Vh[soupy.STATE]), mode="displacement")
    plt.colorbar(ax)
    plt.title("Optimal displacement")
    plt.savefig(f"{save_dir}/u_opt.png")

    if comm_sampler.Get_rank() == 0:
        np.save(f'{save_dir}/z_opt.npy', z_opt)
        print(f"\nResults saved to: {save_dir}/")

    if args.show:
        plt.show()
