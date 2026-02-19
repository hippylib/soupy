# Copyright (c) 2023-2024, The University of Texas at Austin
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
Helmholtz cloaking optimization using Taylor approximations.

This driver minimizes the scattered field from an obstacle surrounded by a
cloak, using Taylor approximations (constant, linear, quadratic) for the
risk measure computation.

Approximation methods:
- Taylor constant: Q(m) ≈ Q(m̄) [deterministic at prior mean]
- Taylor linear: First-order expansion, closed-form mean/variance
- Taylor quadratic: Second-order expansion with dominant Hessian modes

Usage examples:
    # Taylor constant (deterministic)
    python driver_helmholtz_taylor.py -a constant

    # Taylor linear with variance weight
    python driver_helmholtz_taylor.py -a linear --beta 1.0

    # Taylor quadratic with 5 dominant Hessian modes
    python driver_helmholtz_taylor.py -a quadratic --beta 1.0 --n-tr 5

    # With MC correction
    python driver_helmholtz_taylor.py -a quadratic --correction --num-mc 8

Reference:
    Chen, Haberman, Ghattas (2021)
    "Optimal design of acoustic metamaterial cloaks under uncertainty"
    Journal of Computational Physics
"""

import os
import sys
import argparse

# Configure macOS compiler BEFORE importing dolfin or soupy
_soupy_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, os.path.join(_soupy_root, "soupy", "utils"))
try:
    from macos_config import configure_macos_compiler, configure_dolfin_form_compiler
    configure_macos_compiler()
except ImportError:
    pass
sys.path.pop(0)

# Now safe to import dolfin and soupy
sys.path.append(os.environ.get("HIPPYLIB_PATH", ""))
sys.path.append(_soupy_root)

import dolfin as dl
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from mpi4py import MPI

import hippylib as hp
import soupy
from soupy import (
    STATE, PARAMETER, ADJOINT, CONTROL,
)
from soupy.approximations.taylor import (
    TaylorConstantControlCostFunctional,
    TaylorLinearControlCostFunctional,
    TaylorQuadraticControlCostFunctional,
)

dl.set_log_active(False)
try:
    configure_dolfin_form_compiler(dl)
except:
    pass

# Import local setup
from setup_helmholtz_cloaking import setup_helmholtz_cloaking_problem


class QuietScipyCostWrapper:
    """Scipy cost wrapper with clean iteration output."""

    def __init__(self, cost_functional, comm_rank=0, print_every=1):
        self.cost_functional = cost_functional
        self.comm_rank = comm_rank
        self.print_every = print_every
        self.iter_count = 0
        self.cost_history = []
        self.grad_norm_history = []
        self._z = cost_functional.generate_vector(CONTROL)
        self._g = cost_functional.generate_vector(CONTROL)

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
                sys.stdout.flush()
            return self._g.get_local()
        return g


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helmholtz cloaking with Taylor approximations")
    parser.add_argument('-a', '--approximation', type=str, default="linear",
                        choices=["constant", "linear", "quadratic"],
                        help="Taylor approximation order")
    parser.add_argument('-b', '--beta', type=float, default=1.0,
                        help="Variance weight for risk measure")
    parser.add_argument('--n-tr', type=int, default=5,
                        help="Number of dominant Hessian modes (quadratic only)")
    parser.add_argument('--correction', action='store_true',
                        help="Enable Monte Carlo correction")
    parser.add_argument('--num-mc', type=int, default=8,
                        help="Number of MC samples for correction")
    parser.add_argument('--wavenumber', type=float, default=6.28,
                        help="Wavenumber k0 (default: 2*pi)")
    parser.add_argument('--prior-gamma', type=float, default=10.0,
                        help="Prior gamma (lower=larger variance)")
    parser.add_argument('--prior-delta', type=float, default=50.0,
                        help="Prior delta (lower=larger variance)")
    parser.add_argument('--penalty', type=float, default=1e-3,
                        help="Penalization weight")
    parser.add_argument('--maxiter', type=int, default=20,
                        help="Maximum number of optimization iterations")
    parser.add_argument('--print-every', type=int, default=1,
                        help="Print iteration info every N iterations")
    parser.add_argument('--optimizer', type=str, default="lbfgs",
                        choices=["lbfgs", "newton"],
                        help="Optimization method (L-BFGS-B or Inexact Newton-CG)")
    parser.add_argument('--newton-print-level', type=int, default=1,
                        help="Inexact Newton-CG print level (0=quiet)")
    parser.add_argument('--cg-max-iter', type=int, default=40,
                        help="Max CG iterations per Newton step")
    parser.add_argument('--cg-coarse-tol', type=float, default=0.5,
                        help="Coarsest CG relative tolerance (Eisenstat-Walker)")
    parser.add_argument('--cg-print-level', type=int, default=-1,
                        help="CG verbosity level (-1=off)")
    parser.add_argument('--show', default=False, action="store_true",
                        help="Show matplotlib figures")
    parser.add_argument('-v', '--verbose', default=False, action="store_true",
                        help="Verbose output")
    args = parser.parse_args()

    # Setup directories
    save_dir = f"results_taylor_{args.approximation}"
    if args.correction:
        save_dir += "_mc"
    os.makedirs(save_dir, exist_ok=True)

    # MPI setup
    comm_mesh = MPI.COMM_SELF
    comm_sampler = MPI.COMM_WORLD
    rank = comm_sampler.Get_rank()

    # Print header
    if rank == 0:
        print("=" * 70)
        print("Helmholtz Cloaking Optimal Design with Taylor Approximation")
        print("=" * 70)
        print(f"  Wavenumber k0:   {args.wavenumber}")
        print(f"  Approximation:   Taylor {args.approximation}")
        if args.approximation == "quadratic":
            print(f"  N_tr (modes):    {args.n_tr}")
        print(f"  Beta (var wt):   {args.beta}")
        if args.correction:
            print(f"  MC correction:   Yes, N_mc={args.num_mc}")
        else:
            print(f"  MC correction:   No")
        print(f"  Prior:           gamma={args.prior_gamma}, delta={args.prior_delta}")
        print(f"  Penalization:    {args.penalty}")
        print(f"  Max iterations:  {args.maxiter}")
        print(f"  Optimizer:       {args.optimizer}")
        print("=" * 70)
        sys.stdout.flush()

    # Setup problem
    if rank == 0:
        print("\nSetting up problem...")
        sys.stdout.flush()

    mesh, Vh, control_model, prior, penalty, problem = setup_helmholtz_cloaking_problem(
        wavenumber=args.wavenumber,
        prior_gamma=args.prior_gamma,
        prior_delta=args.prior_delta,
        penalty_alpha=args.penalty,
        comm=comm_mesh
    )

    if rank == 0:
        print(f"  Mesh: {mesh.num_cells()} cells, {mesh.num_vertices()} vertices")
        print(f"  State DOFs: {Vh[STATE].dim()}")
        print(f"  Control DOFs: {Vh[CONTROL].dim()}")
        print(f"\nCreating Taylor {args.approximation} cost functional...")
        sys.stdout.flush()

    # Create cost functional based on approximation type
    if args.approximation == "constant":
        settings = {
            "correction": args.correction,
            "N_mc": args.num_mc,
            "beta": args.beta,
            "verbose": args.verbose,
        }
        cost_functional = TaylorConstantControlCostFunctional(
            control_model, prior, penalty, settings
        )

    elif args.approximation == "linear":
        settings = {
            "correction": args.correction,
            "N_mc": args.num_mc,
            "beta": args.beta,
            "verbose": args.verbose,
        }
        cost_functional = TaylorLinearControlCostFunctional(
            control_model, prior, penalty, settings
        )

    elif args.approximation == "quadratic":
        settings = {
            "N_tr": args.n_tr,
            "beta": args.beta,
            "correction": args.correction,
            "N_mc": args.num_mc,
            "verbose": args.verbose,
        }
        cost_functional = TaylorQuadraticControlCostFunctional(
            control_model, prior, penalty, settings
        )

    cost_history = []
    grad_norm_history = []
    result = None

    if args.optimizer == "lbfgs":
        # Create scipy cost wrapper
        scipy_cost = QuietScipyCostWrapper(cost_functional, comm_rank=rank,
                                            print_every=args.print_every)

        # Initial control
        z0 = cost_functional.generate_vector(CONTROL)
        z0_np = z0.get_local()

        # Optimize
        if rank == 0:
            print("\nStarting optimization (L-BFGS-B)...")
            print("-" * 70)
            sys.stdout.flush()

        result = scipy.optimize.minimize(
            scipy_cost.function(),
            z0_np,
            method='L-BFGS-B',
            jac=scipy_cost.jac(),
            options={'maxiter': args.maxiter, 'disp': False}
        )

        # Get optimal control
        z = cost_functional.generate_vector(CONTROL)
        z.set_local(result.x)
        z.apply("")
        cost_history = scipy_cost.cost_history
        grad_norm_history = scipy_cost.grad_norm_history
    else:
        if rank == 0:
            print("\nStarting optimization (Inexact Newton-CG)...")
            print("-" * 70)
            sys.stdout.flush()

        newton_params = soupy.InexactNewtonCG_ParameterList()
        newton_params["max_iter"] = args.maxiter
        newton_params["print_level"] = args.newton_print_level
        newton_params["cg_max_iter"] = args.cg_max_iter
        newton_params["cg_coarse_tolerance"] = args.cg_coarse_tol
        newton_params["cg_print_level"] = args.cg_print_level

        def _newton_cb(it, z_vec):
            cost_history.append(cost_functional.cost(z_vec, order=0))

        optimizer = soupy.InexactNewtonCG(cost_functional, newton_params, callback=_newton_cb)
        z = cost_functional.generate_vector(CONTROL)
        z.zero()
        z, result = optimizer.solve(z)
        grad_norm_history.append(result.get("final_grad_norm", 0.0))

    # Print summary
    if rank == 0:
        print("-" * 70)
        print("Optimization Summary")
        print("-" * 70)
        if args.optimizer == "lbfgs":
            print(f"  Converged:       {result.success}")
            print(f"  Iterations:      {result.nit}")
            print(f"  Function evals:  {result.nfev}")
            print(f"  Final cost:      {result.fun:.6e}")
            if grad_norm_history:
                print(f"  ||grad||:        {grad_norm_history[-1]:.6e}")
        else:
            print(f"  Converged:       {optimizer.converged}")
            print(f"  Iterations:      {optimizer.it}")
            print(f"  Final cost:      {result.get('final_cost', 0.0):.6e}")
            if grad_norm_history:
                print(f"  ||grad||:        {grad_norm_history[-1]:.6e}")
            print(f"  Termination:     {result.get('termination_reason', '')}")
        print("-" * 70)

        # Print Taylor statistics
        if args.approximation == "constant":
            Q0 = cost_functional.lin_mean
            print(f"\nTaylor Constant Statistics")
            print(f"  Q(m̄):            {Q0:.6e}")

        elif args.approximation == "linear":
            legacy = cost_functional._legacy
            taylor_mean = legacy.lin_mean
            taylor_var = legacy.lin_var - taylor_mean ** 2
            print(f"\nTaylor Linear Statistics")
            print(f"  Taylor Mean:     {taylor_mean:.6e}")
            print(f"  Taylor Variance: {taylor_var:.6e}")
            print(f"  Taylor Std:      {np.sqrt(max(0, taylor_var)):.6e}")

        elif args.approximation == "quadratic":
            legacy = cost_functional._legacy
            quad_mean = legacy.quad_mean
            quad_var = legacy.quad_var
            print(f"\nTaylor Quadratic Statistics")
            print(f"  Quad Mean:       {quad_mean:.6e}")
            print(f"  Quad Variance:   {quad_var:.6e}")
            print(f"  Quad Std:        {np.sqrt(max(0, quad_var)):.6e}")
            print(f"  Eigenvalues:     {legacy.d[:min(5, len(legacy.d))]}")

        print("=" * 70)

        # Save plots
        plt.figure()
        z_func = hp.vector2Function(z, Vh[CONTROL])
        mesh = Vh[CONTROL].mesh()
        coords = mesh.coordinates()
        cells = mesh.cells()
        vals = z_func.compute_vertex_values(mesh)
        c = plt.tripcolor(coords[:, 0], coords[:, 1], cells, vals, shading="gouraud")
        plt.gca().set_aspect("equal")
        plt.colorbar(c)
        plt.title(f"Optimal design (Taylor {args.approximation})")
        plt.savefig(os.path.join(save_dir, "optimal_control.png"), dpi=150)
        if args.show:
            plt.show()
        plt.close()

        # Plot convergence
        if cost_history:
            plt.figure()
            plt.semilogy(cost_history)
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.title("Convergence history")
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, "convergence.png"), dpi=150)
            if args.show:
                plt.show()
            plt.close()

        print(f"\nResults saved to: {save_dir}/")
