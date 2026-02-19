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
Semilinear elliptic control using Gaussian Mixture Taylor approximations for CVaR.

This example uses Gaussian mixture Taylor approximations (linear, quadratic) of the QoI
to compute the CVaR (Conditional Value-at-Risk) risk measure efficiently.

The key idea is to decompose the prior N(m_bar, C) into a Gaussian mixture with
reduced variance along a dominant direction (KLE or HEP eigenvector), then use
Taylor approximations at each mixture component mean.

Reference:
    Luo, Chen, Chen, Ghattas (2024)
    "Gaussian mixture Taylor approximations of risk measures constrained by PDEs with Gaussian random field inputs"

Approximation methods:
- Mixture linear CVaR: Uses analytical Gaussian CVaR formula for each component
- Mixture quadratic CVaR: Uses surrogate MC sampling from generalized chi-squared

Comparison with single Taylor CVaR:
- Mixture Taylor can achieve 1-2 orders of magnitude better accuracy
- Cost scales with N_mix (number of mixture components)
- Typically N_mix = 7-15 is sufficient for good accuracy

Usage examples:
    # Mixture linear CVaR with N_mix=7 components using HEP direction
    python driver_semilinear_mixture_cvar.py -a linear --n-mix 7 --direction hep

    # Mixture quadratic CVaR with N_mix=7 and KLE direction (often better for quadratic)
    python driver_semilinear_mixture_cvar.py -a quadratic --n-mix 7 --direction kle

    # Mixture quadratic with more Hessian modes per component
    python driver_semilinear_mixture_cvar.py -a quadratic --n-mix 7 --n-tr 20

Arguments:
    -a, --approximation : Taylor order {linear, quadratic}
    -b, --beta          : CVaR risk level (default: 0.95 for 95% CVaR)
    --n-mix             : Number of mixture components (default: 7)
    --direction         : Decomposition direction {kle, hep} (default: hep for linear, kle for quadratic)
    --n-tr              : Number of dominant Hessian modes for quadratic (default: 10)
    --num-mc            : Number of surrogate MC samples for quadratic CVaR (default: 1000)
    -t, --target        : Target type {sinusoid, constant, arch} (default: sinusoid)
    -p, --param         : Target parameter (default: 1.0)
    --maxiter           : Max L-BFGS-B iterations (default: 100)
    --print-every       : Print iteration info every N iterations (default: 1)
    -v, --verbose       : Verbose output
"""

import time
import os
import sys
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

from soupy.approximations.taylor import (
    TaylorMixtureLinearCVaRControlCostFunctional,
    TaylorMixtureQuadraticCVaRControlCostFunctional,
)

from semilinearEllipticControlPDE import setup_semilinear_elliptic_pde, \
        semilinear_elliptic_control_settings
from semilinearEllipticOUU import get_target, plot_sources

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
logging.getLogger('dijitso').setLevel(logging.WARNING)

dl.set_log_active(False)
dl.parameters["std_out_all_processes"] = False


class QuietScipyCostWrapper:
    """Scipy cost wrapper with clean iteration output."""

    def __init__(self, cost_functional, comm_rank=0, print_every=1):
        self.cost_functional = cost_functional
        self.comm_rank = comm_rank
        self.print_every = print_every
        self.iter_count = 0
        self.n_func = 0
        self.n_grad = 0
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
            self.n_func += 1
            return cost
        return f

    def jac(self):
        def g(z_np):
            self._z.set_local(z_np)
            self._z.apply("")
            self.cost_functional.cost(self._z, order=1)
            grad_norm = self.cost_functional.grad(self._g)
            self.grad_norm_history.append(grad_norm)
            self.n_grad += 1
            self.iter_count += 1
            if self.comm_rank == 0 and self.iter_count % self.print_every == 0:
                print(f"  Iter {self.iter_count:4d}: cost = {self.cost_history[-1]:.6e}, ||grad|| = {grad_norm:.6e}")
                sys.stdout.flush()
            return self._g.get_local()
        return g


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semilinear elliptic with Mixture Taylor CVaR")
    parser.add_argument('-a', '--approximation', type=str, default="linear",
                        choices=["linear", "quadratic"],
                        help="Taylor approximation order")
    parser.add_argument('-b', '--beta', type=float, default=0.95,
                        help="CVaR risk level (default: 0.95 for 95%% CVaR)")
    parser.add_argument('--n-mix', type=int, default=7,
                        help="Number of mixture components (default: 7)")
    parser.add_argument('--direction', type=str, default=None,
                        choices=["kle", "hep"],
                        help="Decomposition direction (default: hep for linear, kle for quadratic)")
    parser.add_argument('--n-tr', type=int, default=10,
                        help="Number of dominant Hessian modes per component (quadratic only)")
    parser.add_argument('--num-mc', type=int, default=1000,
                        help="Number of surrogate MC samples per component for quadratic CVaR")
    parser.add_argument('-t', '--target', type=str, default="sinusoid",
                        help="Target case")
    parser.add_argument('-p', '--param', type=float, default=1.0,
                        help="Parameter for target definition")
    parser.add_argument('--nx', type=int, default=16,
                        help="Number of elements in x direction")
    parser.add_argument('--ny', type=int, default=16,
                        help="Number of elements in y direction")
    parser.add_argument('--N_sources', type=int, default=7,
                        help="Number of sources per side")
    parser.add_argument('--maxiter', type=int, default=100,
                        help="Maximum number of L-BFGS-B iterations")
    parser.add_argument('--print-every', type=int, default=1,
                        help="Print iteration info every N iterations (default: 1)")
    parser.add_argument('-v', '--verbose', default=False, action="store_true",
                        help="Verbose output")
    args = parser.parse_args()

    # Default direction: hep for linear, kle for quadratic (based on validation results)
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
        print("Semilinear Elliptic with Gaussian Mixture Taylor CVaR")
        print("=" * 70)
        print(f"  Approximation:   Mixture {args.approximation} CVaR")
        print(f"  CVaR beta:       {args.beta}")
        print(f"  N_mix:           {args.n_mix}")
        print(f"  Direction:       {args.direction.upper()}")
        if args.approximation == "quadratic":
            print(f"  N_tr (modes):    {args.n_tr}")
            print(f"  N_mc (surrogate):{args.num_mc}")
        print(f"  Target:          {args.target} (param={args.param})")
        print(f"  N sources:       {args.N_sources}x{args.N_sources}")
        print(f"  Max iterations:  {args.maxiter}")
        print("=" * 70)
        sys.stdout.flush()

    # Setup problem
    if rank == 0:
        print("\nSetting up problem...")
        sys.stdout.flush()

    semilinear_settings = semilinear_elliptic_control_settings()
    semilinear_settings['nx'] = args.nx
    semilinear_settings['ny'] = args.ny
    semilinear_settings['n_wells_per_side'] = args.N_sources

    mesh, pde, Vh, prior = setup_semilinear_elliptic_pde(semilinear_settings, comm_mesh=comm_mesh)

    if rank == 0:
        print(f"  Mesh: {mesh.num_cells()} cells, {mesh.num_vertices()} vertices")
        print(f"  Control DOFs: {Vh[soupy.CONTROL].dim()}")
        sys.stdout.flush()

    # Setup QoI and control model
    u_target_expr = get_target(args.target, args.param, comm_mesh)
    u_target = dl.interpolate(u_target_expr, Vh[hp.STATE])
    qoi = soupy.L2MisfitControlQoI(Vh, u_target.vector())
    control_model = soupy.ControlModel(pde, qoi)

    # Create cost functional based on approximation type
    if rank == 0:
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
            control_model, prior, None, settings
        )

    elif args.approximation == "quadratic":
        settings = {
            "beta": args.beta,
            "N_mix": args.n_mix,
            "direction": args.direction,
            "N_tr": args.n_tr,
            "N_mc": args.num_mc,
            "verbose": args.verbose,
        }
        cost_functional = TaylorMixtureQuadraticCVaRControlCostFunctional(
            control_model, prior, None, settings
        )

    # Create scipy cost wrapper
    scipy_cost = QuietScipyCostWrapper(cost_functional, comm_rank=rank, print_every=args.print_every)

    # Box constraints for control
    dim = semilinear_settings["n_wells_per_side"]**2
    lb = np.ones(dim) * semilinear_settings["strength_lower"]
    ub = np.ones(dim) * semilinear_settings["strength_upper"]
    box_bounds = scipy.optimize.Bounds(lb=lb, ub=ub)

    # Initial guess
    z0 = cost_functional.generate_vector(soupy.CONTROL)
    z0_np = z0.get_local()

    # Optimize
    if rank == 0:
        print("\nStarting optimization (L-BFGS-B)...")
        print("-" * 70)
        sys.stdout.flush()

    t0 = time.time()
    result = scipy.optimize.minimize(
        scipy_cost.function(),
        z0_np,
        method='L-BFGS-B',
        jac=scipy_cost.jac(),
        bounds=box_bounds,
        options={'maxiter': args.maxiter, 'disp': False}
    )
    t1 = time.time()

    # Get optimal control
    z_opt = cost_functional.generate_vector(soupy.CONTROL)
    z_opt.set_local(result.x)
    z_opt.apply("")

    # Solve forward at optimum (at prior mean)
    if rank == 0:
        print("\nSolving forward problem at optimal control (at prior mean)...")
        sys.stdout.flush()

    x = [None, None, None, None]
    x[soupy.STATE] = dl.Function(Vh[soupy.STATE]).vector()
    x[soupy.PARAMETER] = dl.Function(Vh[soupy.PARAMETER]).vector()
    x[soupy.ADJOINT] = dl.Function(Vh[soupy.ADJOINT]).vector()
    x[soupy.CONTROL] = z_opt

    x[soupy.PARAMETER].axpy(1.0, prior.mean)
    control_model.solveFwd(x[soupy.STATE], x)

    # Print summary
    if rank == 0:
        print("-" * 70)
        print("Optimization Summary")
        print("-" * 70)
        print(f"  Converged:       {result.success}")
        print(f"  Iterations:      {result.nit}")
        print(f"  Time:            {t1-t0:.2f} s")
        print(f"  Function evals:  {scipy_cost.n_func}")
        print(f"  Gradient evals:  {scipy_cost.n_grad}")
        print(f"  Final cost:      {result.fun:.6e}")
        if scipy_cost.grad_norm_history:
            print(f"  ||grad||:        {scipy_cost.grad_norm_history[-1]:.6e}")
        print("-" * 70)

        # Print Mixture CVaR statistics
        if args.approximation == "linear":
            print(f"Mixture Linear CVaR Statistics (N_mix={args.n_mix}, {args.direction.upper()})")
            print(f"  VaR (0.{int(args.beta*100)}):    {cost_functional.var:.6e}")
            print(f"  CVaR (0.{int(args.beta*100)}):   {cost_functional.cvar:.6e}")
            print(f"  Component Q0s:   {[f'{q:.4e}' for q in cost_functional.component_means]}")
            print(f"  Component stds:  {[f'{s:.4e}' for s in cost_functional.component_stds]}")

        elif args.approximation == "quadratic":
            print(f"Mixture Quadratic CVaR Statistics (N_mix={args.n_mix}, {args.direction.upper()})")
            print(f"  t_opt:           {cost_functional.t_opt:.6e}")
            print(f"  CVaR (0.{int(args.beta*100)}):   {cost_functional.cvar:.6e}")
            print(f"  Component Q0s:   {[f'{q:.4e}' for q in cost_functional.component_Q0]}")

        print("=" * 70)

        # Save results
        z_opt_np = result.x
        np.save(f"{save_dir}/z_opt.npy", z_opt_np)
        np.save(f"{save_dir}/cost_history.npy", np.array(scipy_cost.cost_history))

        # Plot sources
        fig, ax = plot_sources(z_opt_np, semilinear_settings["n_wells_per_side"],
                               semilinear_settings["loc_lower"], semilinear_settings["loc_upper"])
        plt.savefig(f"{save_dir}/control.png")
        plt.close()

        # Plot target
        plt.figure()
        hp.nb.plot(u_target, mytitle="Target state")
        plt.savefig(f"{save_dir}/target.png")
        plt.close()

        # Plot state at optimum
        state_fun = hp.vector2Function(x[soupy.STATE], Vh[soupy.STATE])
        plt.figure()
        hp.nb.plot(state_fun, mytitle="State at optimal (prior mean)")
        plt.savefig(f"{save_dir}/state.png")
        plt.close()

        # Plot convergence
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.semilogy(scipy_cost.cost_history)
        plt.xlabel('Function evaluation')
        plt.ylabel('Cost')
        plt.title('Cost history')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.semilogy(scipy_cost.grad_norm_history)
        plt.xlabel('Gradient evaluation')
        plt.ylabel('||grad||')
        plt.title('Gradient norm history')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/convergence.png")
        plt.close()

        print(f"\nResults saved to: {save_dir}/")
