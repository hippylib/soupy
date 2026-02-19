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
Helmholtz cloaking with Gaussian Mixture Taylor CVaR approximations.

This driver uses the mixture Taylor CVaR approximations (linear or quadratic)
to optimize the cloaking design under uncertainty.

Usage examples:
    # Mixture linear CVaR with N_mix=7 components using HEP direction
    python driver_helmholtz_mixture_cvar.py -a linear --n-mix 7 --direction hep

    # Mixture quadratic CVaR with N_mix=7 and KLE direction
    python driver_helmholtz_mixture_cvar.py -a quadratic --n-mix 7 --direction kle
"""

import os
import sys
import argparse

import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
import dolfin as dl
from mpi4py import MPI

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

import hippylib as hp
import soupy
from soupy import (
    STATE, PARAMETER, ADJOINT, CONTROL,
)
from soupy.approximations.taylor import (
    TaylorMixtureLinearCVaRControlCostFunctional,
    TaylorMixtureQuadraticCVaRControlCostFunctional,
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
    parser = argparse.ArgumentParser(description="Helmholtz cloaking with Mixture Taylor CVaR")
    parser.add_argument('-a', '--approximation', type=str, default="linear",
                        choices=["linear", "quadratic"],
                        help="Taylor approximation order")
    parser.add_argument('-b', '--beta', type=float, default=0.95,
                        help="CVaR risk level (default: 0.95 for 95% CVaR)")
    parser.add_argument('--n-mix', type=int, default=7,
                        help="Number of mixture components (default: 7)")
    parser.add_argument('--direction', type=str, default=None,
                        choices=["kle", "hep"],
                        help="Decomposition direction (default: hep for linear, kle for quadratic)")
    parser.add_argument('--n-tr', type=int, default=10,
                        help="Number of dominant Hessian modes per component (quadratic only)")
    parser.add_argument('--num-mc', type=int, default=1000,
                        help="Number of surrogate MC samples per component for quadratic CVaR")
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
    parser.add_argument('--show', default=False, action="store_true",
                        help="Show matplotlib figures")
    parser.add_argument('-v', '--verbose', default=False, action="store_true",
                        help="Verbose output")
    args = parser.parse_args()

    # Default direction: hep for linear, kle for quadratic.
    if args.direction is None:
        args.direction = "hep" if args.approximation == "linear" else "kle"

    # MPI setup
    comm_mesh = MPI.COMM_SELF
    comm_sampler = MPI.COMM_WORLD
    rank = comm_sampler.Get_rank()

    # Setup directories
    save_dir = f"results_mixture_cvar_{args.approximation}_{args.direction}_nmix{args.n_mix}"
    os.makedirs(save_dir, exist_ok=True)

    # Print header
    if rank == 0:
        print("=" * 70)
        print("Helmholtz Cloaking with Gaussian Mixture Taylor CVaR")
        print("=" * 70)
        print(f"  Approximation:   Mixture {args.approximation} CVaR")
        print(f"  CVaR beta:       {args.beta}")
        print(f"  N_mix:           {args.n_mix}")
        print(f"  Direction:       {args.direction.upper()}")
        if args.approximation == "quadratic":
            print(f"  N_tr (modes):    {args.n_tr}")
            print(f"  N_mc (surrogate):{args.num_mc}")
        print(f"  Wavenumber k0:   {args.wavenumber}")
        print(f"  Prior:           gamma={args.prior_gamma}, delta={args.prior_delta}")
        print(f"  Penalization:    {args.penalty}")
        print(f"  Max iterations:  {args.maxiter}")
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
        print(f"\nCreating Mixture {args.approximation} CVaR cost functional...")
        sys.stdout.flush()

    if args.approximation == "linear":
        settings = {
            "beta": args.beta,
            "N_mix": args.n_mix,
            "direction": args.direction,
            "verbose": args.verbose,
        }
        cost_functional = TaylorMixtureLinearCVaRControlCostFunctional(
            control_model, prior, penalty, settings
        )
    else:
        settings = {
            "beta": args.beta,
            "N_mix": args.n_mix,
            "direction": args.direction,
            "N_tr": args.n_tr,
            "N_mc": args.num_mc,
            "verbose": args.verbose,
        }
        cost_functional = TaylorMixtureQuadraticCVaRControlCostFunctional(
            control_model, prior, penalty, settings
        )

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

    # Print summary
    if rank == 0:
        print("-" * 70)
        print("Optimization Summary")
        print("-" * 70)
        print(f"  Converged:       {result.success}")
        print(f"  Iterations:      {result.nit}")
        print(f"  Function evals:  {result.nfev}")
        print(f"  Final cost:      {result.fun:.6e}")
        if scipy_cost.grad_norm_history:
            print(f"  ||grad||:        {scipy_cost.grad_norm_history[-1]:.6e}")
        print("-" * 70)

        if args.approximation == "linear":
            print(f"\nMixture Linear CVaR Statistics (N_mix={args.n_mix}, {args.direction.upper()})")
            print(f"  VaR:             {cost_functional.var:.6e}")
            print(f"  CVaR:            {cost_functional.cvar:.6e}")
        else:
            print(f"\nMixture Quadratic CVaR Statistics (N_mix={args.n_mix}, {args.direction.upper()})")
            print(f"  t_opt:           {cost_functional.t_opt:.6e}")
            print(f"  CVaR:            {cost_functional.cvar:.6e}")

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
        plt.title(f"Optimal design (Mixture {args.approximation} CVaR, N={args.n_mix})")
        plt.savefig(os.path.join(save_dir, "optimal_control.png"), dpi=150)
        if args.show:
            plt.show()
        plt.close()

        # Plot convergence
        plt.figure()
        plt.semilogy(scipy_cost.cost_history)
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.title("Convergence history")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "convergence.png"), dpi=150)
        if args.show:
            plt.show()
        plt.close()

        print(f"\nResults saved to: {save_dir}/")
