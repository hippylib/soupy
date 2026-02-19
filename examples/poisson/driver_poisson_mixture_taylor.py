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
Poisson control using Gaussian Mixture Taylor approximations for mean + variance.

This example uses Gaussian mixture Taylor approximations (linear, quadratic) of the QoI
to compute the mean + variance risk measure efficiently.

The key idea is to decompose the prior N(m_bar, C) into a Gaussian mixture with
reduced variance along a dominant direction (KLE or HEP eigenvector), then use
Taylor approximations at each mixture component mean.

Reference:
    Luo, Chen, Chen, Ghattas (2024)
    "Gaussian mixture Taylor approximations of risk measures constrained by PDEs with Gaussian random field inputs"

Approximation methods:
- Mixture linear: Uses analytical Gaussian mixture mean/variance formulas
- Mixture quadratic: Uses analytical formulas with Hessian eigendecomposition

Comparison with single Taylor:
- Mixture Taylor can achieve 1-2 orders of magnitude better accuracy
- Cost scales with N_mix (number of mixture components)
- Typically N_mix = 7-15 is sufficient for good accuracy

Usage examples:
    # Mixture linear with N_mix=7 components using HEP direction
    python driver_poisson_mixture_taylor.py -a linear --n-mix 7 --direction hep

    # Mixture quadratic with N_mix=7 and KLE direction (often better for quadratic)
    python driver_poisson_mixture_taylor.py -a quadratic --n-mix 7 --direction kle

    # Mixture quadratic with more Hessian modes per component
    python driver_poisson_mixture_taylor.py -a quadratic --n-mix 7 --n-tr 10

    # Higher variance weight (beta=2.0)
    python driver_poisson_mixture_taylor.py -a linear --n-mix 7 --beta 2.0

Arguments:
    -a, --approximation : Taylor order {linear, quadratic}
    -b, --beta          : Variance weight (default: 1.0)
    --n-mix             : Number of mixture components (default: 7)
    --direction         : Decomposition direction {kle, hep} (default: hep for linear, kle for quadratic)
    --n-tr              : Number of dominant Hessian modes for quadratic (default: 10)
    --prior-gamma       : Prior gamma parameter (default: 10.0, lower=larger variance)
    --prior-delta       : Prior delta parameter (default: 50.0, lower=larger variance)
    --penalty           : Penalization weight (default: 1e-3)
    --maxiter           : Max L-BFGS-B iterations (default: 100)
    --print-every       : Print iteration info every N iterations (default: 1)
    --show              : Show matplotlib figures
    -v, --verbose       : Verbose output
"""

import os
import sys
import time
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
from soupy import ControlModel, L2Penalization, PDEVariationalControlProblem
from soupy.approximations.taylor import (
    TaylorMixtureLinearControlCostFunctional,
    TaylorMixtureQuadraticControlCostFunctional,
)
from soupy.modeling.controlQoI import L2MisfitControlQoI

dl.set_log_active(False)
try:
    configure_dolfin_form_compiler(dl)
except:
    pass


class QuietScipyCostWrapper:
    """Scipy cost wrapper with clean iteration output."""

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
                sys.stdout.flush()
            return self._g.get_local()
        return g


def setup_problem(args, comm_mesh):
    """Set up the Poisson control problem."""
    N_ELEMENTS_X = 20
    N_ELEMENTS_Y = 20
    PRIOR_MEAN = -2.0

    mesh = dl.UnitSquareMesh(comm_mesh, N_ELEMENTS_X, N_ELEMENTS_Y)
    Vh_STATE = dl.FunctionSpace(mesh, "CG", 1)
    Vh_PARAMETER = dl.FunctionSpace(mesh, "CG", 1)
    Vh_CONTROL = dl.FunctionSpace(mesh, "CG", 1)
    Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE, Vh_CONTROL]

    def residual(u, m, p, z):
        return dl.exp(m) * dl.inner(dl.grad(u), dl.grad(p)) * dl.dx - z * p * dl.dx

    def boundary(x, on_boundary):
        return on_boundary and (dl.near(x[0], 0) or dl.near(x[1], 0))

    boundary_value = dl.Expression("x[1]", degree=1, mpi_comm=comm_mesh)
    bc = dl.DirichletBC(Vh_STATE, boundary_value, boundary)
    bc0 = dl.DirichletBC(Vh_STATE, dl.Constant(0.0), boundary)
    pde = PDEVariationalControlProblem(Vh, residual, [bc], [bc0], is_fwd_linear=True)

    mean_vector = dl.interpolate(dl.Constant(PRIOR_MEAN), Vh_PARAMETER).vector()
    prior = hp.BiLaplacianPrior(
        Vh_PARAMETER, args.prior_gamma, args.prior_delta, mean=mean_vector, robin_bc=True
    )

    u_target = dl.Expression(
        "x[1] + sin(k*x[0]) * sin(k*x[1])",
        k=1.5 * np.pi,
        degree=2,
        mpi_comm=comm_mesh,
    )
    u_target_function = dl.interpolate(u_target, Vh_STATE)
    qoi = L2MisfitControlQoI(Vh, u_target_function.vector())

    control_model = ControlModel(pde, qoi)
    penalty = L2Penalization(Vh, args.penalty)

    return mesh, Vh, control_model, prior, penalty, u_target_function


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poisson control with Gaussian Mixture Taylor")
    parser.add_argument('-a', '--approximation', type=str, default="linear",
                        choices=["linear", "quadratic"],
                        help="Taylor approximation order")
    parser.add_argument('-b', '--beta', type=float, default=1.0,
                        help="Variance weight for risk measure")
    parser.add_argument('--n-mix', type=int, default=7,
                        help="Number of mixture components (default: 7)")
    parser.add_argument('--direction', type=str, default=None,
                        choices=["kle", "hep"],
                        help="Decomposition direction (default: hep for linear, kle for quadratic)")
    parser.add_argument('--n-tr', type=int, default=10,
                        help="Number of dominant Hessian modes (quadratic only)")
    parser.add_argument('--prior-gamma', type=float, default=10.0,
                        help="Prior gamma (lower=larger variance)")
    parser.add_argument('--prior-delta', type=float, default=50.0,
                        help="Prior delta (lower=larger variance)")
    parser.add_argument('--penalty', type=float, default=1e-3,
                        help="Penalization weight")
    parser.add_argument('--maxiter', type=int, default=100,
                        help="Maximum number of L-BFGS-B iterations")
    parser.add_argument('--print-every', type=int, default=1,
                        help="Print iteration info every N iterations (default: 1)")
    parser.add_argument('--show', default=False, action="store_true",
                        help="Show matplotlib figures")
    parser.add_argument('-v', '--verbose', default=False, action="store_true",
                        help="Verbose output")
    args = parser.parse_args()

    # Default direction: hep for linear, kle for quadratic (based on validation results)
    if args.direction is None:
        args.direction = "hep" if args.approximation == "linear" else "kle"

    # Setup directories
    save_dir = f"results_mixture_{args.approximation}_{args.direction}_nmix{args.n_mix}"
    os.makedirs(save_dir, exist_ok=True)

    # MPI setup
    comm_mesh = MPI.COMM_SELF
    comm_sampler = MPI.COMM_WORLD
    rank = comm_sampler.Get_rank()

    # Print header
    if rank == 0:
        print("=" * 70)
        print("Poisson Optimal Control with Gaussian Mixture Taylor")
        print("=" * 70)
        print(f"  Approximation:   Mixture {args.approximation}")
        print(f"  N_mix:           {args.n_mix}")
        print(f"  Direction:       {args.direction.upper()}")
        if args.approximation == "quadratic":
            print(f"  N_tr (modes):    {args.n_tr}")
        print(f"  Beta (var wt):   {args.beta}")
        print(f"  Prior:           gamma={args.prior_gamma}, delta={args.prior_delta}")
        print(f"  Penalization:    {args.penalty}")
        print(f"  Max iterations:  {args.maxiter}")
        print("=" * 70)
        sys.stdout.flush()

    # Setup problem
    if rank == 0:
        print("\nSetting up problem...")
        sys.stdout.flush()

    mesh, Vh, control_model, prior, penalty, u_target_function = setup_problem(args, comm_mesh)

    if rank == 0:
        print(f"  Mesh: {mesh.num_cells()} cells, {mesh.num_vertices()} vertices")
        print(f"  Control DOFs: {Vh[soupy.CONTROL].dim()}")
        print(f"\nCreating Mixture {args.approximation} cost functional...")
        sys.stdout.flush()

    # Create cost functional based on approximation type
    if args.approximation == "linear":
        settings = {
            "beta": args.beta,
            "N_mix": args.n_mix,
            "direction": args.direction,
            "verbose": args.verbose,
        }
        cost_functional = TaylorMixtureLinearControlCostFunctional(
            control_model, prior, penalty, settings
        )

    elif args.approximation == "quadratic":
        settings = {
            "beta": args.beta,
            "N_mix": args.n_mix,
            "direction": args.direction,
            "N_tr": args.n_tr,
            "N_mc": 0,  # Use analytical formulas only
            "verbose": args.verbose,
        }
        cost_functional = TaylorMixtureQuadraticControlCostFunctional(
            control_model, prior, penalty, settings
        )

    # Create scipy cost wrapper
    scipy_cost = QuietScipyCostWrapper(cost_functional, comm_rank=rank, print_every=args.print_every)

    # Initial control
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
        options={'maxiter': args.maxiter, 'disp': False}
    )
    t1 = time.time()

    # Get optimal control
    z = cost_functional.generate_vector(soupy.CONTROL)
    z.set_local(result.x)
    z.apply("")

    # Solve forward at optimum (at prior mean parameter)
    if rank == 0:
        print("\nSolving forward problem at optimal control (at prior mean)...")
        sys.stdout.flush()

    x = cost_functional.generate_vector()
    x[soupy.CONTROL].axpy(1.0, z)
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
        print(f"  Function evals:  {result.nfev}")
        print(f"  Final cost:      {result.fun:.6e}")
        if scipy_cost.grad_norm_history:
            print(f"  ||grad||:        {scipy_cost.grad_norm_history[-1]:.6e}")
        print("-" * 70)

        # Print Mixture statistics
        if args.approximation == "linear":
            print(f"Mixture Linear Statistics (N_mix={args.n_mix}, {args.direction.upper()})")
            print(f"  Mean:            {cost_functional.mixture_mean:.6e}")
            print(f"  Variance:        {cost_functional.mixture_var:.6e}")
            print(f"  Component Q0s:   {[f'{q:.4e}' for q in cost_functional.component_means]}")
            print(f"  Component stds:  {[f'{s:.4e}' for s in cost_functional.component_stds]}")

        elif args.approximation == "quadratic":
            print(f"Mixture Quadratic Statistics (N_mix={args.n_mix}, {args.direction.upper()})")
            print(f"  Mean:            {cost_functional.mixture_mean:.6e}")
            print(f"  Variance:        {cost_functional.mixture_var:.6e}")
            print(f"  Component Q0s:   {[f'{q:.4e}' for q in cost_functional.component_Q0]}")
            print(f"  Comp E[Q]:       {[f'{m:.4e}' for m in cost_functional.component_quad_mean]}")
            print(f"  Comp Var[Q]:     {[f'{v:.4e}' for v in cost_functional.component_quad_var]}")

        print("=" * 70)

        # Save plots
        plt.figure()
        hp.nb.plot(hp.vector2Function(x[soupy.CONTROL], Vh[soupy.CONTROL]))
        plt.title(f"Optimal control (Mixture {args.approximation})")
        plt.savefig(os.path.join(save_dir, "optimal_control.png"))
        if args.show:
            plt.show()
        plt.close()

        plt.figure()
        hp.nb.plot(hp.vector2Function(x[soupy.STATE], Vh[soupy.STATE]))
        plt.title("State at optimum")
        plt.savefig(os.path.join(save_dir, "state.png"))
        if args.show:
            plt.show()
        plt.close()

        plt.figure()
        hp.nb.plot(u_target_function)
        plt.title("Target state")
        plt.savefig(os.path.join(save_dir, "target_state.png"))
        if args.show:
            plt.show()
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
        plt.savefig(os.path.join(save_dir, "convergence.png"))
        if args.show:
            plt.show()
        plt.close()

        print(f"\nResults saved to: {save_dir}/")
