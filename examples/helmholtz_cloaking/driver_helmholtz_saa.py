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
Helmholtz cloaking optimization using Sample Average Approximation (SAA).

This driver minimizes the scattered field from an obstacle surrounded by a
cloak, using SAA to handle the uncertainty in the material properties.

Usage examples:
    # Basic run with 16 samples
    python driver_helmholtz_saa.py --num-samples 16

    # With variance weight
    python driver_helmholtz_saa.py --num-samples 32 --beta 1.0

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
    MeanVarRiskMeasureSAA,
    meanVarRiskMeasureSAASettings,
    RiskMeasureControlCostFunctional,
    L2Penalization,
    STATE, PARAMETER, ADJOINT, CONTROL,
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
    parser = argparse.ArgumentParser(description="Helmholtz cloaking with SAA")
    parser.add_argument('-n', '--num-samples', type=int, default=16,
                        help="Number of SAA samples")
    parser.add_argument('-b', '--beta', type=float, default=0.0,
                        help="Variance weight for risk measure")
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
    save_dir = f"results_saa_n{args.num_samples}"
    os.makedirs(save_dir, exist_ok=True)

    # MPI setup
    comm_mesh = MPI.COMM_SELF
    comm_sampler = MPI.COMM_WORLD
    rank = comm_sampler.Get_rank()

    # Print header
    if rank == 0:
        print("=" * 70)
        print("Helmholtz Cloaking Optimal Design with SAA")
        print("=" * 70)
        print(f"  Wavenumber k0:   {args.wavenumber}")
        print(f"  SAA samples:     {args.num_samples}")
        print(f"  Beta (var wt):   {args.beta}")
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

    try:
        mesh, Vh, control_model, prior, penalty, problem = setup_helmholtz_cloaking_problem(
            wavenumber=args.wavenumber,
            prior_gamma=args.prior_gamma,
            prior_delta=args.prior_delta,
            penalty_alpha=args.penalty,
            comm=comm_mesh
        )
    except Exception as e:
        if rank == 0:
            print(f"Error setting up problem: {e}")
            print("Using fallback simple Helmholtz problem...")
        # Fallback to simpler problem if setup fails
        raise

    if rank == 0:
        print(f"  Mesh: {mesh.num_cells()} cells, {mesh.num_vertices()} vertices")
        print(f"  State DOFs: {Vh[STATE].dim()}")
        print(f"  Control DOFs: {Vh[CONTROL].dim()}")
        sys.stdout.flush()

    # Save initial scattered field (real part) at mean parameter and zero control.
    if rank == 0:
        x0 = control_model.generate_vector()
        x0[PARAMETER].axpy(1.0, prior.mean)
        control_model.solveFwd(x0[STATE], x0)
        u_func = hp.vector2Function(x0[STATE], Vh[STATE])
        u1, u2 = u_func.split(deepcopy=True)
        mesh = Vh[STATE].mesh()
        coords = mesh.coordinates()
        cells = mesh.cells()
        u1_vals = u1.compute_vertex_values(mesh)
        u2_vals = u2.compute_vertex_values(mesh)
        plt.figure()
        c = plt.tripcolor(coords[:, 0], coords[:, 1], cells, u1_vals, shading="gouraud")
        plt.gca().set_aspect("equal")
        plt.colorbar(c)
        plt.title("Initial scattered field (real part)")
        plt.savefig(os.path.join(save_dir, "state_initial_real.png"), dpi=150)
        if args.show:
            plt.show()
        plt.close()

        plt.figure()
        c = plt.tripcolor(coords[:, 0], coords[:, 1], cells, u2_vals, shading="gouraud")
        plt.gca().set_aspect("equal")
        plt.colorbar(c)
        plt.title("Initial scattered field (imag part)")
        plt.savefig(os.path.join(save_dir, "state_initial_imag.png"), dpi=150)
        if args.show:
            plt.show()
        plt.close()

        k0 = problem["wavenumber"]
        inc_dir = np.array(problem["incident_dir"], dtype=float)
        inc_norm = np.linalg.norm(inc_dir)
        if inc_norm == 0.0:
            inc_dir = np.array([1.0, 0.0])
        else:
            inc_dir = inc_dir / inc_norm
        phase = k0 * (inc_dir[0] * coords[:, 0] + inc_dir[1] * coords[:, 1])
        inc1 = np.cos(phase)
        inc2 = np.sin(phase)
        total1 = u1_vals + inc1
        total2 = u2_vals + inc2

        plt.figure()
        c = plt.tripcolor(coords[:, 0], coords[:, 1], cells, total1, shading="gouraud")
        plt.gca().set_aspect("equal")
        plt.colorbar(c)
        plt.title("Initial total field (real part)")
        plt.savefig(os.path.join(save_dir, "state_initial_total_real.png"), dpi=150)
        if args.show:
            plt.show()
        plt.close()

        plt.figure()
        c = plt.tripcolor(coords[:, 0], coords[:, 1], cells, total2, shading="gouraud")
        plt.gca().set_aspect("equal")
        plt.colorbar(c)
        plt.title("Initial total field (imag part)")
        plt.savefig(os.path.join(save_dir, "state_initial_total_imag.png"), dpi=150)
        if args.show:
            plt.show()
        plt.close()

    # Create SAA risk measure
    if rank == 0:
        print(f"\nCreating SAA risk measure with {args.num_samples} samples...")
        sys.stdout.flush()

    risk_measure_settings = meanVarRiskMeasureSAASettings()
    risk_measure_settings['sample_size'] = args.num_samples
    risk_measure_settings['beta'] = args.beta
    risk_measure = MeanVarRiskMeasureSAA(control_model, prior, risk_measure_settings)

    # Create cost functional
    cost_functional = RiskMeasureControlCostFunctional(risk_measure, penalty)

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

        # Get statistics from risk measure
        mean_val = risk_measure.q_bar
        var_val = risk_measure.q2_bar - risk_measure.q_bar**2
        print(f"\nSAA Statistics (at optimum)")
        print(f"  Mean Q:          {mean_val:.6e}")
        print(f"  Var Q:           {var_val:.6e}")
        print(f"  Std Q:           {np.sqrt(max(0, var_val)):.6e}")
        print("=" * 70)

        # Plot and save final scattered/total fields at mean parameter.
        x_opt = control_model.generate_vector()
        x_opt[PARAMETER].axpy(1.0, prior.mean)
        x_opt[CONTROL].axpy(1.0, z)
        control_model.solveFwd(x_opt[STATE], x_opt)
        u_func = hp.vector2Function(x_opt[STATE], Vh[STATE])
        u1, u2 = u_func.split(deepcopy=True)
        mesh = Vh[STATE].mesh()
        coords = mesh.coordinates()
        cells = mesh.cells()
        u1_vals = u1.compute_vertex_values(mesh)
        u2_vals = u2.compute_vertex_values(mesh)

        k0 = problem["wavenumber"]
        inc_dir = np.array(problem["incident_dir"], dtype=float)
        inc_norm = np.linalg.norm(inc_dir)
        if inc_norm == 0.0:
            inc_dir = np.array([1.0, 0.0])
        else:
            inc_dir = inc_dir / inc_norm
        phase = k0 * (inc_dir[0] * coords[:, 0] + inc_dir[1] * coords[:, 1])
        inc1_vals = np.cos(phase)
        inc2_vals = np.sin(phase)
        total1_vals = u1_vals + inc1_vals
        total2_vals = u2_vals + inc2_vals

        plt.figure()
        c = plt.tripcolor(coords[:, 0], coords[:, 1], cells, u1_vals, shading="gouraud")
        plt.gca().set_aspect("equal")
        plt.colorbar(c)
        plt.title("Final scattered field (real part)")
        plt.savefig(os.path.join(save_dir, "state_final_real.png"), dpi=150)
        if args.show:
            plt.show()
        plt.close()

        plt.figure()
        c = plt.tripcolor(coords[:, 0], coords[:, 1], cells, u2_vals, shading="gouraud")
        plt.gca().set_aspect("equal")
        plt.colorbar(c)
        plt.title("Final scattered field (imag part)")
        plt.savefig(os.path.join(save_dir, "state_final_imag.png"), dpi=150)
        if args.show:
            plt.show()
        plt.close()

        plt.figure()
        c = plt.tripcolor(coords[:, 0], coords[:, 1], cells, total1_vals, shading="gouraud")
        plt.gca().set_aspect("equal")
        plt.colorbar(c)
        plt.title("Final total field (real part)")
        plt.savefig(os.path.join(save_dir, "state_final_total_real.png"), dpi=150)
        if args.show:
            plt.show()
        plt.close()

        plt.figure()
        c = plt.tripcolor(coords[:, 0], coords[:, 1], cells, total2_vals, shading="gouraud")
        plt.gca().set_aspect("equal")
        plt.colorbar(c)
        plt.title("Final total field (imag part)")
        plt.savefig(os.path.join(save_dir, "state_final_total_imag.png"), dpi=150)
        if args.show:
            plt.show()
        plt.close()

        # Save fields to XDMF.
        Vh_scalar = dl.FunctionSpace(mesh, "CG", 1)
        u1_fun = dl.Function(Vh_scalar)
        u2_fun = dl.Function(Vh_scalar)
        u1_fun.vector().axpy(1.0, u1.vector())
        u2_fun.vector().axpy(1.0, u2.vector())

        inc1 = dl.Expression(
            "cos(k*(b0*x[0] + b1*x[1]))",
            k=k0, b0=inc_dir[0], b1=inc_dir[1], degree=2,
            mpi_comm=mesh.mpi_comm(),
        )
        inc2 = dl.Expression(
            "sin(k*(b0*x[0] + b1*x[1]))",
            k=k0, b0=inc_dir[0], b1=inc_dir[1], degree=2,
            mpi_comm=mesh.mpi_comm(),
        )
        inc1_fun = dl.interpolate(inc1, Vh_scalar)
        inc2_fun = dl.interpolate(inc2, Vh_scalar)
        total1_fun = dl.Function(Vh_scalar)
        total2_fun = dl.Function(Vh_scalar)
        total1_fun.vector().axpy(1.0, u1_fun.vector())
        total1_fun.vector().axpy(1.0, inc1_fun.vector())
        total2_fun.vector().axpy(1.0, u2_fun.vector())
        total2_fun.vector().axpy(1.0, inc2_fun.vector())

        u1_fun.rename("scattered_real", "")
        u2_fun.rename("scattered_imag", "")
        total1_fun.rename("total_real", "")
        total2_fun.rename("total_imag", "")
        xdmf_path = os.path.join(save_dir, "final_fields.xdmf")
        with dl.XDMFFile(mesh.mpi_comm(), xdmf_path) as xdmf:
            xdmf.parameters["functions_share_mesh"] = True
            xdmf.parameters["rewrite_function_mesh"] = False
            xdmf.write(u1_fun, 0.0)
            xdmf.write(u2_fun, 0.0)
            xdmf.write(total1_fun, 0.0)
            xdmf.write(total2_fun, 0.0)

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
        plt.title("Optimal design (control)")
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
