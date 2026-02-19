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
Poisson control using Taylor approximations for risk-averse optimization.

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
    python driver_poisson_taylor.py -a constant

    # Taylor linear with variance weight beta=1.0
    python driver_poisson_taylor.py -a linear --beta 1.0

    # Taylor quadratic with 5 dominant Hessian modes
    python driver_poisson_taylor.py -a quadratic --beta 1.0 --n-tr 5

    # Taylor linear with MC correction (8 samples)
    python driver_poisson_taylor.py -a linear --correction --num-mc 8

    # Taylor quadratic with MC correction
    python driver_poisson_taylor.py -a quadratic --beta 1.0 --n-tr 5 --correction --num-mc 8

    # Taylor constant + MC = SAA
    python driver_poisson_taylor.py -a constant --correction --num-mc 16

    # With larger prior variance (gamma=1, delta=5)
    python driver_poisson_taylor.py -a quadratic --beta 1.0 --prior-gamma 1.0 --prior-delta 5.0

Arguments:
    -a, --approximation : Taylor order {constant, linear, quadratic}
    -b, --beta          : Variance weight for risk measure (default: 1.0)
    --n-tr              : Number of dominant Hessian modes for quadratic (default: 5)
    --correction        : Enable Monte Carlo correction
    --num-mc            : Number of MC samples for correction (default: 8)
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
    TaylorConstantControlCostFunctional,
    TaylorLinearControlCostFunctional,
    TaylorQuadraticControlCostFunctional,
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
    parser = argparse.ArgumentParser(description="Poisson control with Taylor approximations")
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
        print("Poisson Optimal Control with Taylor Approximation")
        print("=" * 70)
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

    result = scipy.optimize.minimize(
        scipy_cost.function(),
        z0_np,
        method='L-BFGS-B',
        jac=scipy_cost.jac(),
        options={'maxiter': args.maxiter, 'disp': False}
    )

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
    x[soupy.PARAMETER].axpy(1.0, prior.mean)  # Set parameter to prior mean
    control_model.solveFwd(x[soupy.STATE], x)

    # Print summary
    if rank == 0:
        print("-" * 70)
        print("Optimization Summary")
        print("-" * 70)
        print(f"  Converged:       {result.success}")
        print(f"  Iterations:      {result.nit}")
        print(f"  Function evals:  {result.nfev}")
        print(f"  Final cost:      {result.fun:.6e}")
        print(f"  ||grad||:        {scipy_cost.grad_norm_history[-1]:.6e}")
        print("-" * 70)

        # Print Taylor statistics
        if args.approximation == "constant":
            Q0 = cost_functional.lin_mean
            print(f"Taylor Constant Statistics")
            print(f"  Q(m̄):            {Q0:.6e}")
            if args.correction and args.num_mc > 0:
                saa_mean = np.mean(cost_functional.Q_mc)
                saa_var = np.var(cost_functional.Q_mc)
                print(f"  SAA Mean:        {saa_mean:.6e}")
                print(f"  SAA Variance:    {saa_var:.6e}")

        elif args.approximation == "linear":
            legacy = cost_functional._legacy
            taylor_mean = legacy.lin_mean
            taylor_var = legacy.lin_var - taylor_mean ** 2
            print(f"Taylor Linear Statistics")
            print(f"  Taylor Mean:     {taylor_mean:.6e}")
            print(f"  Taylor Variance: {taylor_var:.6e}")
            if args.correction and args.num_mc > 0:
                mean_diff = np.mean(legacy.lin_diff_mean)
                var_diff = np.mean(legacy.lin_diff_var)
                corrected_mean = taylor_mean + mean_diff
                corrected_var = taylor_var + var_diff
                print(f"  Mean correction: {mean_diff:.6e}")
                print(f"  Var correction:  {var_diff:.6e}")
                print(f"  Corrected Mean:  {corrected_mean:.6e}")
                print(f"  Corrected Var:   {corrected_var:.6e}")

        elif args.approximation == "quadratic":
            legacy = cost_functional._legacy
            quad_mean = legacy.quad_mean
            quad_var = legacy.quad_var
            print(f"Taylor Quadratic Statistics")
            print(f"  Quad Mean:       {quad_mean:.6e}")
            print(f"  Quad Variance:   {quad_var:.6e}")
            print(f"  Eigenvalues:     {legacy.d[:5]}")
            if args.correction and args.num_mc > 0:
                mean_diff = legacy.mean_diff
                var_diff = legacy.var_diff
                print(f"  Mean correction: {mean_diff:.6e}")
                print(f"  Var correction:  {var_diff:.6e}")

        print("=" * 70)

        # Save plots
        plt.figure()
        hp.nb.plot(hp.vector2Function(x[soupy.CONTROL], Vh[soupy.CONTROL]))
        plt.title(f"Optimal control (Taylor {args.approximation})")
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

        print(f"\nResults saved to: {save_dir}/")
