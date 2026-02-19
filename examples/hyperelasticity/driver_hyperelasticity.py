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

    def __init__(self, cost_functional, comm_rank=0):
        self.cost_functional = cost_functional
        self.comm_rank = comm_rank
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
            if self.comm_rank == 0 and self.iter_count % 10 == 0:
                print(f"  Iter {self.iter_count:4d}: cost = {self.cost_history[-1]:.6e}, ||grad|| = {grad_norm:.6e}")
            return self._g.get_local()
        return g


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--risk_measure', type=str, default="deterministic", choices=["deterministic", "mean_var"], help="Risk measure type")
    parser.add_argument('-n', '--sample_size', type=int, default=32, help="Sample size for risk measure computation")
    parser.add_argument('-b', '--beta', type=float, default=1.0, help="Variance weight for risk measure")
    parser.add_argument('-q', '--qoi_type', type=str, default="stiffness", choices=["all", "stiffness", "point"])
    parser.add_argument('-p', '--penalization', type=float, default=1e-1, help="Scaling of penalization")

    parser.add_argument('--maxiter', type=int, default=100, help="Maximum number of SD iterations")
    parser.add_argument('--show', default=False, action="store_true", help="Show figure")
    parser.add_argument('-v', '--verbose', default=False, action="store_true", help="Verbose output")
    args = parser.parse_args()

    save_dir = "results_%s" %(args.risk_measure)
    os.makedirs(save_dir, exist_ok=True)

    # Create mesh and setup model components
    comm_mesh = MPI.COMM_SELF
    comm_sampler = MPI.COMM_WORLD
    rank = comm_sampler.Get_rank()

    if rank == 0:
        print("=" * 70)
        print("Hyperelasticity Optimal Design Problem")
        print("=" * 70)
        print(f"  Risk measure:    {args.risk_measure}")
        if args.risk_measure == "mean_var":
            print(f"  Sample size:     {args.sample_size}")
            print(f"  Beta (var wt):   {args.beta}")
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
    if args.verbose:
        scipy_cost = soupy.ScipyCostWrapper(pde_cost, verbose=True)
    else:
        scipy_cost = QuietScipyCostWrapper(pde_cost, comm_rank=rank)
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
    if args.verbose:
        options['disp'] = True

    results = scipy.optimize.minimize(scipy_cost.function(), z0_np,
            method="L-BFGS-B", jac=scipy_cost.jac(),
            bounds=box_bounds, options=options)

    # Optimal design
    z_opt = results['x']

    # --------------------------------------------------------------------- #

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
        print("=" * 70)
        sys.stdout.flush()

    # ---------- postprocessing the plotting ----------- #

    plt.figure()
    ax = dl.plot(disp_fun_init, mode="displacement")
    plt.colorbar(ax)
    plt.title("Displacement with soft material")
    plt.savefig("%s/u_soft.png" %(save_dir))

    plt.figure()
    ax = dl.plot(hp.vector2Function(x[soupy.CONTROL], Vh[soupy.CONTROL]), vmin=0.0, vmax=1.0, extend='max')
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
        print(f"\nResults saved to: {save_dir}/")

    if args.show:
        plt.show()
