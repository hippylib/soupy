"""MPI regression check for cluster-parallel mixture Taylor models.

This script reuses the semilinear ADR setup with a serial mesh on each rank
(`MPI.COMM_SELF`) and compares:

- serial mixture models (`comm_sampler=MPI.COMM_SELF`)
- cluster-parallel mixture models (`comm_sampler=MPI.COMM_WORLD`)

Recommended usage:

    mpirun -n 3 python test_mixture_quadratic_parallel.py --direction kle

By default, `N_mix` is set to the MPI world size so that each cluster is owned
by exactly one rank.
"""

from __future__ import annotations

import argparse
import os
import sys

_soupy_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.environ.get("HIPPYLIB_PATH", ""))
sys.path.append(_soupy_root)

import dolfin as dl
import numpy as np
from mpi4py import MPI

import hippylib as hp
import soupy
from semilinear_adr_problem import (
    MeshParameters,
    PDEParameters,
    PriorParameters,
    SemilinearEllipticVarfHandler,
    setup_mesh,
    setup_prior,
    setup_qoi,
)
from soupy import (
    ControlModel,
    PDEVariationalControlProblem,
    VariationalControlQoI,
)
from soupy.approximations.taylor import (
    TaylorMixtureLinearCVaRControlCostFunctional,
    TaylorMixtureLinearControlCostFunctional,
    TaylorMixtureQuadraticCVaRControlCostFunctional,
    TaylorMixtureQuadraticControlCostFunctional,
)


dl.set_log_active(False)


class ControlledSemilinearADRVarfHandler:
    """Add the control as a source term: f becomes f + z."""

    def __init__(self, base_varf_handler):
        self.base_varf_handler = base_varf_handler

    def __call__(self, u, m, p, z):
        return self.base_varf_handler(u, m, p) - z * p * dl.dx


class SemilinearADRQoIFormHandler:
    """Wrap the ADR QoI form so it is compatible with a control variable."""

    def __init__(self, qoi_varf):
        self.qoi_varf = qoi_varf

    def __call__(self, u, m, z):
        del z
        return self.qoi_varf(u, m)


def setup_control_problem(nx, ny, comm_mesh, newton_max_it, newton_rtol, newton_atol):
    mesh_parameters = MeshParameters(nx=nx, ny=ny)
    pde_parameters = PDEParameters()
    prior_parameters = PriorParameters()
    qoi_type = "l2"

    mesh = setup_mesh(mesh_parameters, comm_mesh)
    Vh_state = dl.FunctionSpace(mesh, "CG", 1)
    Vh_parameter = dl.FunctionSpace(mesh, "CG", 1)
    Vh_control = dl.FunctionSpace(mesh, "CG", 1)
    Vh = [Vh_state, Vh_parameter, Vh_state, Vh_control]

    bc = dl.DirichletBC(Vh_state, dl.Constant(0.0), "on_boundary && near(x[0], 0.0)")
    bc0 = dl.DirichletBC(Vh_state, dl.Constant(0.0), "on_boundary && near(x[0], 0.0)")
    pde_varf = ControlledSemilinearADRVarfHandler(
        SemilinearEllipticVarfHandler(Vh, pde_parameters)
    )
    pde = PDEVariationalControlProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=False)
    pde.set_nonlinear_solver_parameters(
        {
            "newton_solver": {
                "linear_solver": "lu",
                "maximum_iterations": newton_max_it,
                "relative_tolerance": newton_rtol,
                "absolute_tolerance": newton_atol,
                "error_on_nonconvergence": True,
            }
        }
    )

    prior = setup_prior(Vh, prior_parameters)
    base_qoi = setup_qoi([Vh_state, Vh_parameter, Vh_state], qoi_type, mesh)
    qoi = VariationalControlQoI(Vh, SemilinearADRQoIFormHandler(base_qoi.qoi_varf))
    control_model = ControlModel(pde, qoi)

    return control_model, prior, Vh


def make_control_vector(cost_functional, Vh_control, control_mode, amplitude):
    z = cost_functional.generate_vector(soupy.CONTROL)
    z.zero()

    if control_mode == "zero":
        z.apply("")
        return z

    if control_mode == "wave":
        expr = dl.Expression(
            "a*sin(pi*x[0])*sin(pi*x[1])",
            a=amplitude,
            pi=np.pi,
            degree=4,
            mpi_comm=Vh_control.mesh().mpi_comm(),
        )
        z_fun = dl.interpolate(expr, Vh_control)
        z.axpy(1.0, z_fun.vector())
        z.apply("")
        return z

    raise ValueError(f"Unsupported control_mode: {control_mode}")


def evaluate_cost_and_grad(cost_functional, z):
    value = float(cost_functional.cost(z, order=1))
    g = cost_functional.generate_vector(soupy.CONTROL)
    grad_norm = float(cost_functional.grad(g))
    return value, g, grad_norm


def make_augmented_control(cost_functional, z, t_value):
    zt = cost_functional.generate_vector(soupy.CONTROL)
    zt.get_vector().zero()
    zt.get_vector().axpy(1.0, z)
    zt.set_scalar(float(t_value))
    return zt


def flatten_component_modes(component_d):
    if len(component_d) == 0:
        return np.zeros(0, dtype=float)
    return np.concatenate([np.asarray(modes, dtype=float) for modes in component_d])


def max_abs_diff(a, b):
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.size == 0:
        return 0.0
    return float(np.max(np.abs(a_arr - b_arr)))


def summarize_common_component_parallel(parallel_cost, comm, model_name):
    ownership = comm.gather(parallel_cost.owned_component_indices, root=0)
    if comm.Get_rank() == 0:
        print(f"[{model_name} MPI check] cluster_parallel_enabled={parallel_cost.cluster_parallel_enabled}")
        for owner_rank, indices in enumerate(ownership):
            print(f"  rank {owner_rank} owns clusters {list(indices)}")


def compare_case(serial_cost, parallel_cost, z, label, atol, rtol, comm):
    serial_value, g_serial, serial_grad_norm = evaluate_cost_and_grad(serial_cost, z)
    parallel_value, g_parallel, parallel_grad_norm = evaluate_cost_and_grad(parallel_cost, z)

    grad_serial = g_serial.get_local()
    grad_parallel = g_parallel.get_local()
    grad_diff = float(np.linalg.norm(grad_parallel - grad_serial))
    grad_ref = max(float(np.linalg.norm(grad_serial)), 1.0)

    component_mean_diff = float(
        np.max(
            np.abs(
                np.asarray(parallel_cost.component_quad_mean, dtype=float)
                - np.asarray(serial_cost.component_quad_mean, dtype=float)
            )
        )
    )
    component_var_diff = float(
        np.max(
            np.abs(
                np.asarray(parallel_cost.component_quad_var, dtype=float)
                - np.asarray(serial_cost.component_quad_var, dtype=float)
            )
        )
    )
    component_mode_diff = float(
        np.max(
            np.abs(
                flatten_component_modes(parallel_cost.component_d)
                - flatten_component_modes(serial_cost.component_d)
            )
        )
    )

    objective_diff = abs(parallel_value - serial_value)
    mean_diff = abs(parallel_cost.mixture_mean - serial_cost.mixture_mean)
    var_diff = abs(parallel_cost.mixture_var - serial_cost.mixture_var)
    grad_norm_diff = abs(parallel_grad_norm - serial_grad_norm)

    local_fail = int(
        objective_diff > atol + rtol * max(abs(serial_value), 1.0)
        or mean_diff > atol + rtol * max(abs(serial_cost.mixture_mean), 1.0)
        or var_diff > atol + rtol * max(abs(serial_cost.mixture_var), 1.0)
        or grad_norm_diff > atol + rtol * max(abs(serial_grad_norm), 1.0)
        or grad_diff > atol + rtol * grad_ref
        or component_mean_diff > atol + rtol * max(np.max(np.abs(serial_cost.component_quad_mean)), 1.0)
        or component_var_diff > atol + rtol * max(np.max(np.abs(serial_cost.component_quad_var)), 1.0)
        or component_mode_diff > atol + rtol * max(np.max(np.abs(flatten_component_modes(serial_cost.component_d))), 1.0)
    )
    global_fail = comm.allreduce(local_fail, op=MPI.MAX)

    summary = {
        "label": label,
        "serial_value": serial_value,
        "parallel_value": parallel_value,
        "objective_diff": objective_diff,
        "mean_diff": mean_diff,
        "var_diff": var_diff,
        "grad_norm_diff": grad_norm_diff,
        "grad_l2_diff": grad_diff,
        "component_mean_diff": component_mean_diff,
        "component_var_diff": component_var_diff,
        "component_mode_diff": component_mode_diff,
        "failed": bool(global_fail),
    }
    return summary


def compare_linear_case(serial_cost, parallel_cost, z, label, atol, rtol, comm):
    serial_value, g_serial, serial_grad_norm = evaluate_cost_and_grad(serial_cost, z)
    parallel_value, g_parallel, parallel_grad_norm = evaluate_cost_and_grad(parallel_cost, z)

    grad_diff = float(np.linalg.norm(g_parallel.get_local() - g_serial.get_local()))
    grad_ref = max(float(np.linalg.norm(g_serial.get_local())), 1.0)
    component_mean_diff = max_abs_diff(parallel_cost.component_means, serial_cost.component_means)
    component_std_diff = max_abs_diff(parallel_cost.component_stds, serial_cost.component_stds)
    objective_diff = abs(parallel_value - serial_value)
    mean_diff = abs(parallel_cost.mixture_mean - serial_cost.mixture_mean)
    var_diff = abs(parallel_cost.mixture_var - serial_cost.mixture_var)
    grad_norm_diff = abs(parallel_grad_norm - serial_grad_norm)

    local_fail = int(
        objective_diff > atol + rtol * max(abs(serial_value), 1.0)
        or mean_diff > atol + rtol * max(abs(serial_cost.mixture_mean), 1.0)
        or var_diff > atol + rtol * max(abs(serial_cost.mixture_var), 1.0)
        or grad_norm_diff > atol + rtol * max(abs(serial_grad_norm), 1.0)
        or grad_diff > atol + rtol * grad_ref
        or component_mean_diff > atol + rtol * max(np.max(np.abs(serial_cost.component_means)), 1.0)
        or component_std_diff > atol + rtol * max(np.max(np.abs(serial_cost.component_stds)), 1.0)
    )
    global_fail = comm.allreduce(local_fail, op=MPI.MAX)
    return {
        "label": label,
        "serial_value": serial_value,
        "parallel_value": parallel_value,
        "objective_diff": objective_diff,
        "mean_diff": mean_diff,
        "var_diff": var_diff,
        "grad_norm_diff": grad_norm_diff,
        "grad_l2_diff": grad_diff,
        "component_mean_diff": component_mean_diff,
        "component_std_diff": component_std_diff,
        "failed": bool(global_fail),
    }


def compare_linear_cvar_case(serial_cost, parallel_cost, z, label, atol, rtol, comm):
    serial_value, g_serial, serial_grad_norm = evaluate_cost_and_grad(serial_cost, z)
    parallel_value, g_parallel, parallel_grad_norm = evaluate_cost_and_grad(parallel_cost, z)

    grad_diff = float(np.linalg.norm(g_parallel.get_local() - g_serial.get_local()))
    grad_ref = max(float(np.linalg.norm(g_serial.get_local())), 1.0)
    component_mean_diff = max_abs_diff(parallel_cost.component_means, serial_cost.component_means)
    component_std_diff = max_abs_diff(parallel_cost.component_stds, serial_cost.component_stds)
    objective_diff = abs(parallel_value - serial_value)
    cvar_diff = abs(parallel_cost.cvar - serial_cost.cvar)
    var_diff = abs(parallel_cost.var - serial_cost.var)
    grad_norm_diff = abs(parallel_grad_norm - serial_grad_norm)

    local_fail = int(
        objective_diff > atol + rtol * max(abs(serial_value), 1.0)
        or cvar_diff > atol + rtol * max(abs(serial_cost.cvar), 1.0)
        or var_diff > atol + rtol * max(abs(serial_cost.var), 1.0)
        or grad_norm_diff > atol + rtol * max(abs(serial_grad_norm), 1.0)
        or grad_diff > atol + rtol * grad_ref
        or component_mean_diff > atol + rtol * max(np.max(np.abs(serial_cost.component_means)), 1.0)
        or component_std_diff > atol + rtol * max(np.max(np.abs(serial_cost.component_stds)), 1.0)
    )
    global_fail = comm.allreduce(local_fail, op=MPI.MAX)
    return {
        "label": label,
        "serial_value": serial_value,
        "parallel_value": parallel_value,
        "objective_diff": objective_diff,
        "cvar_diff": cvar_diff,
        "var_diff": var_diff,
        "grad_norm_diff": grad_norm_diff,
        "grad_l2_diff": grad_diff,
        "component_mean_diff": component_mean_diff,
        "component_std_diff": component_std_diff,
        "failed": bool(global_fail),
    }


def compare_quadratic_cvar_case(serial_cost, parallel_cost, zt, label, atol, rtol, comm):
    serial_value, g_serial, serial_grad_norm = evaluate_cost_and_grad(serial_cost, zt)
    parallel_value, g_parallel, parallel_grad_norm = evaluate_cost_and_grad(parallel_cost, zt)

    grad_vec_diff = float(
        np.linalg.norm(g_parallel.get_vector().get_local() - g_serial.get_vector().get_local())
    )
    grad_scalar_diff = abs(float(g_parallel.get_scalar()) - float(g_serial.get_scalar()))
    grad_ref = max(float(np.linalg.norm(g_serial.get_vector().get_local())), 1.0)
    component_q0_diff = max_abs_diff(parallel_cost.component_Q0, serial_cost.component_Q0)
    component_mode_diff = max_abs_diff(
        flatten_component_modes(parallel_cost.component_d),
        flatten_component_modes(serial_cost.component_d),
    )
    sample_diff = max_abs_diff(
        np.concatenate(parallel_cost.component_samples),
        np.concatenate(serial_cost.component_samples),
    )
    objective_diff = abs(parallel_value - serial_value)
    cvar_diff = abs(parallel_cost.cvar - serial_cost.cvar)
    t_opt_diff = abs(parallel_cost.t_opt - serial_cost.t_opt)
    grad_norm_diff = abs(parallel_grad_norm - serial_grad_norm)

    local_fail = int(
        objective_diff > atol + rtol * max(abs(serial_value), 1.0)
        or cvar_diff > atol + rtol * max(abs(serial_cost.cvar), 1.0)
        or t_opt_diff > atol + rtol * max(abs(serial_cost.t_opt), 1.0)
        or grad_norm_diff > atol + rtol * max(abs(serial_grad_norm), 1.0)
        or grad_vec_diff > atol + rtol * grad_ref
        or grad_scalar_diff > atol + rtol * max(abs(float(g_serial.get_scalar())), 1.0)
        or component_q0_diff > atol + rtol * max(np.max(np.abs(serial_cost.component_Q0)), 1.0)
        or component_mode_diff > atol + rtol * max(np.max(np.abs(flatten_component_modes(serial_cost.component_d))), 1.0)
        or sample_diff > atol + rtol * max(np.max(np.abs(np.concatenate(serial_cost.component_samples))), 1.0)
    )
    global_fail = comm.allreduce(local_fail, op=MPI.MAX)
    return {
        "label": label,
        "serial_value": serial_value,
        "parallel_value": parallel_value,
        "objective_diff": objective_diff,
        "cvar_diff": cvar_diff,
        "t_opt_diff": t_opt_diff,
        "grad_norm_diff": grad_norm_diff,
        "grad_l2_diff": grad_vec_diff,
        "grad_scalar_diff": grad_scalar_diff,
        "component_q0_diff": component_q0_diff,
        "component_mode_diff": component_mode_diff,
        "sample_diff": sample_diff,
        "failed": bool(global_fail),
    }


def print_summaries(model_name, summaries, rank):
    if rank != 0:
        return
    for item in summaries:
        details = [f"[{item['label']}]", f"J_serial={item['serial_value']:.12e}", f"J_parallel={item['parallel_value']:.12e}"]
        for key, value in item.items():
            if key in {"label", "serial_value", "parallel_value", "failed"}:
                continue
            details.append(f"{key}={value:.3e}")
        print(", ".join(details))
    print(f"[{model_name} MPI check] {'FAILED' if any(item['failed'] for item in summaries) else 'PASSED'}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=12, help="Number of mesh cells in x.")
    parser.add_argument("--ny", type=int, default=12, help="Number of mesh cells in y.")
    parser.add_argument("--n-tr", type=int, default=3, help="Number of quadratic Hessian modes.")
    parser.add_argument(
        "--n-mix",
        type=int,
        default=None,
        help="Number of mixture components. Defaults to MPI world size.",
    )
    parser.add_argument(
        "--direction",
        choices=("kle", "hep"),
        default="kle",
        help="Mixture direction used by mixture_quadratic.",
    )
    parser.add_argument("--beta", type=float, default=0.5, help="Variance weight.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument(
        "--control-amplitude",
        type=float,
        default=0.2,
        help="Amplitude for the nonzero test control.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-9,
        help="Absolute tolerance for serial/parallel comparisons.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-7,
        help="Relative tolerance for serial/parallel comparisons.",
    )
    parser.add_argument(
        "--allow-multi-clusters-per-rank",
        action="store_true",
        help="Allow running with world size different from N_mix.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable model verbosity.")
    parser.add_argument("--newton-max-it", type=int, default=30, help="Forward Newton iteration cap.")
    parser.add_argument("--newton-rtol", type=float, default=1e-10, help="Forward Newton relative tolerance.")
    parser.add_argument("--newton-atol", type=float, default=1e-12, help="Forward Newton absolute tolerance.")
    return parser.parse_args()


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_mix = size if args.n_mix is None else args.n_mix
    if size > n_mix:
        raise RuntimeError(f"MPI size {size} cannot exceed N_mix {n_mix}.")
    if (not args.allow_multi_clusters_per_rank) and size != n_mix:
        raise RuntimeError(
            "This regression check expects one cluster per rank. "
            f"Received MPI size {size} and N_mix {n_mix}. "
            "Either rerun with matching values or pass --allow-multi-clusters-per-rank."
        )

    control_model, prior, Vh = setup_control_problem(
        nx=args.nx,
        ny=args.ny,
        comm_mesh=MPI.COMM_SELF,
        newton_max_it=args.newton_max_it,
        newton_rtol=args.newton_rtol,
        newton_atol=args.newton_atol,
    )
    Vh_control = Vh[soupy.CONTROL]

    mv_settings = {
        "beta": args.beta,
        "N_mix": n_mix,
        "direction": args.direction,
        "N_tr": args.n_tr,
        "N_mc": 0,
        "seed": args.seed,
        "verbose": args.verbose,
    }
    cvar_settings = {
        "beta": 0.95,
        "N_mix": n_mix,
        "direction": args.direction,
        "N_tr": args.n_tr,
        "N_mc": max(16, 4 * n_mix),
        "seed": args.seed,
        "epsilon": 1e-4,
        "verbose": args.verbose,
    }

    z_zero = make_control_vector(
        TaylorMixtureLinearControlCostFunctional(control_model, prior, settings=mv_settings, comm_sampler=MPI.COMM_SELF),
        Vh_control,
        "zero",
        args.control_amplitude,
    )
    z_wave = make_control_vector(
        TaylorMixtureLinearControlCostFunctional(control_model, prior, settings=mv_settings, comm_sampler=MPI.COMM_SELF),
        Vh_control,
        "wave",
        args.control_amplitude,
    )

    failed = False

    linear_serial = TaylorMixtureLinearControlCostFunctional(control_model, prior, settings=mv_settings, comm_sampler=MPI.COMM_SELF)
    linear_parallel = TaylorMixtureLinearControlCostFunctional(control_model, prior, settings=mv_settings, comm_sampler=comm)
    if rank == 0:
        print(f"[mixture_linear MPI check] direction={args.direction}, N_mix={n_mix}, world_size={size}")
    summarize_common_component_parallel(linear_parallel, comm, "mixture_linear")
    linear_summaries = [
        compare_linear_case(linear_serial, linear_parallel, z_zero, "zero", args.atol, args.rtol, comm),
        compare_linear_case(linear_serial, linear_parallel, z_wave, "wave", args.atol, args.rtol, comm),
    ]
    print_summaries("mixture_linear", linear_summaries, rank)
    failed = failed or any(item["failed"] for item in linear_summaries)

    quad_serial = TaylorMixtureQuadraticControlCostFunctional(control_model, prior, settings=mv_settings, comm_sampler=MPI.COMM_SELF)
    quad_parallel = TaylorMixtureQuadraticControlCostFunctional(control_model, prior, settings=mv_settings, comm_sampler=comm)
    if rank == 0:
        print(f"[mixture_quadratic MPI check] direction={args.direction}, N_mix={n_mix}, world_size={size}")
    summarize_common_component_parallel(quad_parallel, comm, "mixture_quadratic")
    quad_summaries = [
        compare_case(quad_serial, quad_parallel, z_zero, "zero", args.atol, args.rtol, comm),
        compare_case(quad_serial, quad_parallel, z_wave, "wave", args.atol, args.rtol, comm),
    ]
    print_summaries("mixture_quadratic", quad_summaries, rank)
    failed = failed or any(item["failed"] for item in quad_summaries)

    linear_cvar_serial = TaylorMixtureLinearCVaRControlCostFunctional(control_model, prior, settings=cvar_settings, comm_sampler=MPI.COMM_SELF)
    linear_cvar_parallel = TaylorMixtureLinearCVaRControlCostFunctional(control_model, prior, settings=cvar_settings, comm_sampler=comm)
    if rank == 0:
        print(f"[mixture_linear_cvar MPI check] direction={args.direction}, N_mix={n_mix}, world_size={size}")
    summarize_common_component_parallel(linear_cvar_parallel, comm, "mixture_linear_cvar")
    linear_cvar_summaries = [
        compare_linear_cvar_case(linear_cvar_serial, linear_cvar_parallel, z_zero, "zero", args.atol, args.rtol, comm),
        compare_linear_cvar_case(linear_cvar_serial, linear_cvar_parallel, z_wave, "wave", args.atol, args.rtol, comm),
    ]
    print_summaries("mixture_linear_cvar", linear_cvar_summaries, rank)
    failed = failed or any(item["failed"] for item in linear_cvar_summaries)

    quad_cvar_serial = TaylorMixtureQuadraticCVaRControlCostFunctional(control_model, prior, settings=cvar_settings, comm_sampler=MPI.COMM_SELF)
    quad_cvar_parallel = TaylorMixtureQuadraticCVaRControlCostFunctional(control_model, prior, settings=cvar_settings, comm_sampler=comm)
    if rank == 0:
        print(f"[mixture_quadratic_cvar MPI check] direction={args.direction}, N_mix={n_mix}, world_size={size}")
    summarize_common_component_parallel(quad_cvar_parallel, comm, "mixture_quadratic_cvar")
    quad_cvar_zero = make_augmented_control(quad_cvar_serial, z_zero, 0.0)
    quad_cvar_wave = make_augmented_control(quad_cvar_serial, z_wave, 0.0)
    quad_cvar_summaries = [
        compare_quadratic_cvar_case(quad_cvar_serial, quad_cvar_parallel, quad_cvar_zero, "zero", args.atol, args.rtol, comm),
        compare_quadratic_cvar_case(quad_cvar_serial, quad_cvar_parallel, quad_cvar_wave, "wave", args.atol, args.rtol, comm),
    ]
    print_summaries("mixture_quadratic_cvar", quad_cvar_summaries, rank)
    failed = failed or any(item["failed"] for item in quad_cvar_summaries)

    if failed:
        raise RuntimeError("At least one serial vs cluster-parallel mixture model comparison failed.")


if __name__ == "__main__":
    main()
