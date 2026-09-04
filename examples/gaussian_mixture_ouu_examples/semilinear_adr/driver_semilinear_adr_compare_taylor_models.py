"""Compare Taylor and SAA models on semilinear ADR control.

This mirrors the workflow/style of
`examples/poisson/driver_poisson_compare_taylor_models.py`, adapted to the
semilinear ADR problem. The PDE has no known source term; the right hand
side is entirely the Gaussian-well combination sum_i z_i psi_i. The initial
control is the zero coefficient vector.

Models compared:
- linear
- quadratic
- mixture_linear_kle
- mixture_linear_hep
- mixture_quadratic_kle
- mixture_quadratic_hep
- saa_1
- saa_10
- saa_39
- saa_100
- saa_200
- saa_500
- saa_1000
- saa_10000

Reference truth:
- Mean-variance SAA objective with a large fixed sample size
- Evaluated only at the initial control and each model optimum

No MC correction is used in Taylor-based models.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List

# Configure macOS compiler BEFORE importing dolfin or soupy
_soupy_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, os.path.join(_soupy_root, "soupy", "utils"))
try:
    from macos_config import configure_macos_compiler, configure_dolfin_form_compiler

    configure_macos_compiler()
except ImportError:
    pass
sys.path.pop(0)

sys.path.append(os.environ.get("HIPPYLIB_PATH", ""))
sys.path.append(_soupy_root)

import dolfin as dl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import scipy.optimize
from mpi4py import MPI

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 14,
})

# Larger font override for the "optimal control vs optimal solution" field plots only.
LARGE_FONT_RC = {
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 16,
}

import hippylib as hp
import soupy
from semilinear_adr_problem import (
    ControlParameters,
    MeshParameters,
    PDEParameters,
    PriorParameters,
    ControlledSemilinearADRWellVarfHandler,
    control_coefficients_to_function,
    setup_control_function_space,
    setup_mesh,
    setup_prior,
    setup_qoi,
)
from soupy import (
    ControlModel,
    MeanVarRiskMeasureSAA,
    PDEVariationalControlProblem,
    RiskMeasureControlCostFunctional,
    VariationalControlQoI,
    meanVarRiskMeasureSAASettings,
)
from soupy.approximations.taylor import (
    TaylorLinearControlCostFunctional,
    TaylorMixtureLinearControlCostFunctional,
    TaylorMixtureQuadraticControlCostFunctional,
    TaylorQuadraticControlCostFunctional,
)


dl.set_log_active(False)
try:
    configure_dolfin_form_compiler(dl)
except Exception:
    pass


MODEL_ORDER = [
    "linear",
    "quadratic",
    "mixture_linear_kle",
    "mixture_linear_hep",
    "mixture_quadratic_kle",
    "mixture_quadratic_hep",
    "saa_1",
    "saa_10",
    "saa_39",
    "saa_100",
    "saa_200",
    "saa_500",
    "saa_1000",
    "saa_2000",
    "saa_5000",
    "saa_10000",
]

MODEL_COLORS = {
    "linear": "tab:blue",
    "quadratic": "tab:orange",
    "mixture_linear_kle": "tab:green",
    "mixture_linear_hep": "tab:olive",
    "mixture_quadratic_kle": "tab:red",
    "mixture_quadratic_hep": "tab:brown",
    "saa_1": "black",
    "saa_10": "#17becf",
    "saa_39": "#7f7f7f",
    "saa_100": "#9467bd",
    "saa_200": "#e377c2",
    "saa_500": "#8c564b",
    "saa_1000": "#bcbd22",
    "saa_2000": "#393b79",
    "saa_5000": "#e7969c",
    "saa_10000": "#1f9e89",
}


class TeeStream:
    """Duplicate writes to terminal stream and a log file stream."""

    def __init__(self, terminal_stream, file_stream):
        self._terminal = terminal_stream
        self._file = file_stream

    def write(self, data):
        self._terminal.write(data)
        self._file.write(data)

    def flush(self):
        self._terminal.flush()
        self._file.flush()


@dataclass
class IterRecord:
    iteration: int
    model_cost: float
    residual: float
    iter_time_sec: float
    rss_mb: float
    peak_rss_mb: float
    cost_rss_before_mb: float
    cost_rss_after_mb: float
    grad_rss_before_mb: float
    grad_rss_after_mb: float


class ControlledSemilinearADRVarfHandler:
    """Legacy field-control wrapper retained for backward compatibility."""

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


def _rss_from_proc_status_mb() -> float:
    """Read current RSS from /proc/self/status when available."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    match = re.search(r"(\d+)", line)
                    if match:
                        return float(match.group(1)) / 1024.0
    except OSError:
        pass
    return float("nan")


def get_current_rss_mb() -> float:
    """Best-effort current resident set size in MB."""
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 ** 2)
    except Exception:
        return _rss_from_proc_status_mb()


def get_peak_rss_mb() -> float:
    """Best-effort peak resident set size in MB."""
    try:
        import resource

        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return float(peak_kb) / (1024.0 ** 2)
        return float(peak_kb) / 1024.0
    except Exception:
        return float("nan")


def projected_gradient_inf_norm(x, grad, bounds):
    if grad is None:
        return float("nan")
    x = np.asarray(x, dtype=float)
    projected_grad = np.asarray(grad, dtype=float).copy()
    if bounds is not None:
        lb = np.asarray(bounds.lb, dtype=float)
        ub = np.asarray(bounds.ub, dtype=float)
        at_lb = np.isfinite(lb) & np.isclose(x, lb) & (projected_grad > 0.0)
        at_ub = np.isfinite(ub) & np.isclose(x, ub) & (projected_grad < 0.0)
        projected_grad[at_lb | at_ub] = 0.0
    return float(np.linalg.norm(projected_grad, ord=np.inf))


class ScipyObjectiveWithHistory:
    """Wrap a cost functional for scipy minimize and keep latest value/grad norm."""

    def __init__(self, cost_functional):
        self.cost_functional = cost_functional
        self._z = cost_functional.generate_vector(soupy.CONTROL)
        self._g = cost_functional.generate_vector(soupy.CONTROL)
        self.latest_cost = np.nan
        self.latest_grad_norm = np.nan
        self.latest_gradient = None
        self.latest_cost_rss_before_mb = np.nan
        self.latest_cost_rss_after_mb = np.nan
        self.latest_grad_rss_before_mb = np.nan
        self.latest_grad_rss_after_mb = np.nan
        self.n_func = 0
        self.n_grad = 0

    def function(self):
        def f(z_np):
            self._z.set_local(z_np)
            self._z.apply("")
            self.latest_cost_rss_before_mb = get_current_rss_mb()
            self.latest_cost = float(self.cost_functional.cost(self._z, order=0))
            self.latest_cost_rss_after_mb = get_current_rss_mb()
            self.n_func += 1
            return self.latest_cost

        return f

    def jac(self):
        def g(z_np):
            self._z.set_local(z_np)
            self._z.apply("")
            self.latest_grad_rss_before_mb = get_current_rss_mb()
            self.cost_functional.cost(self._z, order=1)
            self.latest_grad_norm = float(self.cost_functional.grad(self._g))
            self.latest_gradient = np.array(self._g.get_local(), copy=True)
            self.latest_grad_rss_after_mb = get_current_rss_mb()
            self.n_grad += 1
            return self.latest_gradient

        return g


def setup_problem(args, comm_mesh):
    mesh_parameters = MeshParameters()
    pde_parameters = PDEParameters()
    prior_parameters = PriorParameters()
    control_parameters = ControlParameters()
    qoi_type = "mismatch"

    if args.nx is not None:
        mesh_parameters.nx = args.nx
    if args.ny is not None:
        mesh_parameters.ny = args.ny

    mesh = setup_mesh(mesh_parameters, comm_mesh)
    Vh_state = dl.FunctionSpace(mesh, "CG", 1)
    Vh_parameter = dl.FunctionSpace(mesh, "CG", 1)
    Vh_control = setup_control_function_space(mesh, control_parameters)
    Vh = [Vh_state, Vh_parameter, Vh_state, Vh_control]

    bc = dl.DirichletBC(Vh_state, dl.Constant(0.0), "on_boundary")
    bc0 = dl.DirichletBC(Vh_state, dl.Constant(0.0), "on_boundary")
    pde_varf = ControlledSemilinearADRWellVarfHandler(Vh, pde_parameters, control_parameters)
    pde = PDEVariationalControlProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=False)
    pde.set_nonlinear_solver_parameters(
        {
            "newton_solver": {
                "linear_solver": "lu",
                "maximum_iterations": args.newton_max_it,
                "relative_tolerance": args.newton_rtol,
                "absolute_tolerance": args.newton_atol,
                "error_on_nonconvergence": True,
            }
        }
    )

    prior = setup_prior(Vh, prior_parameters)
    base_qoi = setup_qoi([Vh_state, Vh_parameter, Vh_state], qoi_type, mesh)
    qoi = VariationalControlQoI(Vh, SemilinearADRQoIFormHandler(base_qoi.qoi_varf))
    control_model = ControlModel(pde, qoi)

    return {
        "mesh_parameters": mesh_parameters,
        "pde_parameters": pde_parameters,
        "prior_parameters": prior_parameters,
        "control_parameters": control_parameters,
        "qoi_type": qoi_type,
        "mesh": mesh,
        "Vh": Vh,
        "control_model": control_model,
        "prior": prior,
    }


def make_saa_cost(control_model, prior, beta, sample_size, seed, comm_sampler):
    risk_settings = meanVarRiskMeasureSAASettings()
    risk_settings["beta"] = beta
    risk_settings["sample_size"] = sample_size
    risk_settings["seed"] = seed
    risk_measure = MeanVarRiskMeasureSAA(
        control_model,
        prior,
        settings=risk_settings,
        comm_sampler=comm_sampler,
    )
    return RiskMeasureControlCostFunctional(risk_measure, penalization=None)


def model_uses_world_parallel(model_name: str) -> bool:
    if model_name.startswith("mixture_"):
        return True
    if model_name.startswith("saa_"):
        return int(model_name.split("_", 1)[1]) >= 11
    return False


def make_taylor_cost(model_name, control_model, prior, args):
    if model_name.startswith("saa_"):
        sample_size = int(model_name.split("_", 1)[1])
        return make_saa_cost(
            control_model,
            prior,
            beta=args.beta,
            sample_size=sample_size,
            seed=args.saa_seed,
            comm_sampler=MPI.COMM_WORLD if sample_size >= 11 else MPI.COMM_SELF,
        )

    if model_name == "linear":
        settings = {
            "beta": args.beta,
            "correction": False,
            "N_mc": 0,
            "verbose": args.verbose,
        }
        return TaylorLinearControlCostFunctional(control_model, prior, None, settings)

    if model_name == "quadratic":
        settings = {
            "beta": args.beta,
            "N_tr": args.n_tr,
            "correction": False,
            "N_mc": 0,
            "verbose": args.verbose,
        }
        return TaylorQuadraticControlCostFunctional(control_model, prior, None, settings)

    if model_name == "mixture_linear_kle":
        settings = {
            "beta": args.beta,
            "N_mix": args.n_mix,
            "direction": "kle",
            "verbose": args.verbose,
        }
        return TaylorMixtureLinearControlCostFunctional(
            control_model,
            prior,
            None,
            settings,
            comm_sampler=MPI.COMM_WORLD,
        )

    if model_name == "mixture_linear_hep":
        settings = {
            "beta": args.beta,
            "N_mix": args.n_mix,
            "direction": "hep",
            "verbose": args.verbose,
        }
        return TaylorMixtureLinearControlCostFunctional(
            control_model,
            prior,
            None,
            settings,
            comm_sampler=MPI.COMM_WORLD,
        )

    if model_name == "mixture_quadratic_kle":
        settings = {
            "beta": args.beta,
            "N_mix": args.n_mix,
            "direction": "kle",
            "N_tr": args.n_tr,
            "N_mc": 0,
            "verbose": args.verbose,
        }
        return TaylorMixtureQuadraticControlCostFunctional(
            control_model,
            prior,
            None,
            settings,
            comm_sampler=MPI.COMM_WORLD,
        )

    if model_name == "mixture_quadratic_hep":
        settings = {
            "beta": args.beta,
            "N_mix": args.n_mix,
            "direction": "hep",
            "N_tr": args.n_tr,
            "N_mc": 0,
            "verbose": args.verbose,
        }
        return TaylorMixtureQuadraticControlCostFunctional(
            control_model,
            prior,
            None,
            settings,
            comm_sampler=MPI.COMM_WORLD,
        )

    raise ValueError(f"Unknown model name: {model_name}")


def np_to_control(cost_functional, z_np):
    z = cost_functional.generate_vector(soupy.CONTROL)
    z.set_local(z_np)
    z.apply("")
    return z


def zero_control_np(cost_functional):
    z0 = cost_functional.generate_vector(soupy.CONTROL)
    z0.zero()
    z0.apply("")
    return z0.get_local()


def evaluate_cost(cost_functional, z_np):
    z = np_to_control(cost_functional, z_np)
    return float(cost_functional.cost(z, order=0))


def evaluate_true_cost_stats(cost_functional, z_np):
    z = np_to_control(cost_functional, z_np)
    total_cost = float(cost_functional.cost(z, order=0))
    risk = cost_functional.risk_measure
    mean = float(risk.q_bar)
    variance = float(risk.q2_bar - risk.q_bar ** 2)
    return total_cost, mean, variance


def make_bounds(x0, args):
    lb = np.full_like(x0, args.bound_lb, dtype=float)
    ub = np.full_like(x0, args.bound_ub, dtype=float)
    if len(x0) == args.control_dim + 1:
        lb[-1] = -np.inf
        ub[-1] = np.inf
    return scipy.optimize.Bounds(lb, ub)


def optimize_with_tracking(model_name, approx_cost, args, rank, maxiter, root_only=False, bounds=None):
    payload = None

    if (not root_only) or rank == 0:
        x0 = zero_control_np(approx_cost)
        wrapper = ScipyObjectiveWithHistory(approx_cost)
        iter_records: List[IterRecord] = []
        callback_last_time = None

        def callback(xk):
            nonlocal callback_last_time
            now = time.perf_counter()
            iter_time = np.nan if callback_last_time is None else now - callback_last_time
            callback_last_time = now
            iter_records.append(
                IterRecord(
                    iteration=len(iter_records) + 1,
                    model_cost=float(wrapper.latest_cost),
                    residual=projected_gradient_inf_norm(xk, wrapper.latest_gradient, bounds),
                    iter_time_sec=iter_time,
                    rss_mb=get_current_rss_mb(),
                    peak_rss_mb=get_peak_rss_mb(),
                    cost_rss_before_mb=float(wrapper.latest_cost_rss_before_mb),
                    cost_rss_after_mb=float(wrapper.latest_cost_rss_after_mb),
                    grad_rss_before_mb=float(wrapper.latest_grad_rss_before_mb),
                    grad_rss_after_mb=float(wrapper.latest_grad_rss_after_mb),
                )
            )

        t0 = time.perf_counter()
        result = scipy.optimize.minimize(
            wrapper.function(),
            x0,
            method="L-BFGS-B",
            jac=wrapper.jac(),
            callback=callback,
            bounds=bounds,
            options={"maxiter": maxiter, "disp": False, "ftol": 1e-12, "gtol": 1e-4, "maxls": 20},
        )
        total_time = time.perf_counter() - t0
        iter_count = len(iter_records) if iter_records else int(result.nit)
        avg_iter_time_sec = total_time / max(iter_count, 1)

        for r in iter_records:
            if rank == 0 and r.iteration % args.print_every == 0:
                print(
                    f"  [{model_name:20s}] iter {r.iteration:4d}: "
                    f"J_model={r.model_cost:.6e}, ||proj g||_inf={r.residual:.3e}, "
                    f"RSS={r.rss_mb:.1f} MB, PeakRSS={r.peak_rss_mb:.1f} MB, "
                    f"cost_mem={r.cost_rss_before_mb:.1f}->{r.cost_rss_after_mb:.1f} MB, "
                    f"grad_mem={r.grad_rss_before_mb:.1f}->{r.grad_rss_after_mb:.1f} MB"
                )
                sys.stdout.flush()

        z_opt_np = np.array(result.x, copy=True)
        z_opt_vec = np_to_control(approx_cost, z_opt_np)
        approx_opt = float(approx_cost.cost(z_opt_vec, order=0))

        if rank == 0:
            serial_note = " [root-only serial]" if root_only else ""
            print(
                f"  [{model_name:20s}] done{serial_note}: success={result.success}, nit={iter_count}, "
                f"avg_iter_time={avg_iter_time_sec:.2f}s"
            )
            sys.stdout.flush()

        payload = {
            "success": bool(result.success),
            "nfev": int(result.nfev),
            "njev": int(result.njev),
            "message": str(result.message),
            "iter_records": iter_records,
            "iter_count": iter_count,
            "avg_iter_time_sec": avg_iter_time_sec,
            "total_time_sec": total_time,
            "z0_np": np.array(x0, copy=True),
            "z_opt_np": z_opt_np,
            "approx_opt": approx_opt,
            "n_func": wrapper.n_func,
            "n_grad": wrapper.n_grad,
        }

    if root_only:
        payload = MPI.COMM_WORLD.bcast(payload if rank == 0 else None, root=0)

    return payload


def save_iteration_csv(path, records: List[IterRecord]):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "iteration",
                "model_cost",
                "projected_grad_inf_norm",
                "iter_time_sec",
                "rss_mb",
                "peak_rss_mb",
                "cost_rss_before_mb",
                "cost_rss_after_mb",
                "grad_rss_before_mb",
                "grad_rss_after_mb",
            ]
        )
        for r in records:
            writer.writerow(
                [
                    r.iteration,
                    r.model_cost,
                    r.residual,
                    r.iter_time_sec,
                    r.rss_mb,
                    r.peak_rss_mb,
                    r.cost_rss_before_mb,
                    r.cost_rss_after_mb,
                    r.grad_rss_before_mb,
                    r.grad_rss_after_mb,
                ]
            )


def save_summary_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "success",
                "message",
                "nit",
                "nfev",
                "njev",
                "avg_iter_time_sec",
                "initial_model_objective",
                "initial_true_objective",
                "initial_rel_error",
                "opt_model_objective",
                "opt_true_objective",
                "opt_rel_error",
            ]
        )
        for row in rows:
            writer.writerow(row)


def relative_error(model_value, true_value):
    return float(abs(model_value - true_value) / max(abs(true_value), 1e-14))


def evaluate_truth_for_result(result, truth_cost):
    result["true_init"], result["true_init_mean"], result["true_init_var"] = evaluate_true_cost_stats(
        truth_cost, result["z0_np"]
    )
    result["init_rel_err"] = relative_error(result["approx_init"], result["true_init"])
    result["true_opt"], result["true_opt_mean"], result["true_opt_var"] = evaluate_true_cost_stats(
        truth_cost, result["z_opt_np"]
    )
    result["opt_rel_err"] = relative_error(result["approx_opt"], result["true_opt"])


def plot_curves(results, save_dir):
    # Objective and gradient-residual curves share the same per-model legend, so they are
    # drawn side by side in one figure with a single shared legend on the right.
    fig, (ax_obj, ax_res) = plt.subplots(1, 2, figsize=(15, 5))
    for model in MODEL_ORDER:
        rec = results[model]["iter_records"]
        if not rec:
            continue
        x = [r.iteration for r in rec]
        y_obj = [max(abs(r.model_cost), 1e-16) for r in rec]
        y_res = [max(r.residual, 1e-16) for r in rec]
        ax_obj.semilogy(x, y_obj, marker="o", linewidth=1.5, markersize=3, color=MODEL_COLORS[model], label=model)
        ax_res.semilogy(x, y_res, marker="o", linewidth=1.5, markersize=3, color=MODEL_COLORS[model], label=model)
    ax_obj.set_xlabel("Iteration")
    ax_obj.set_ylabel("Model Objective")
    ax_obj.set_title("Objective Value per Iteration")
    ax_obj.grid(True, which="both", alpha=0.3)
    ax_res.set_xlabel("Iteration")
    ax_res.set_ylabel("Projected Gradient Inf Norm")
    ax_res.set_title("Projected Gradient Inf Norm per Iteration")
    ax_res.grid(True, which="both", alpha=0.3)
    handles, labels = ax_obj.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "objective_and_residual_per_iteration.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    plt.figure(figsize=(11, 5))
    x = np.arange(len(MODEL_ORDER))
    time_values = [results[m]["avg_iter_time_sec"] for m in MODEL_ORDER]
    plt.bar(x, time_values, color=[MODEL_COLORS[m] for m in MODEL_ORDER])
    plt.xticks(x, MODEL_ORDER, rotation=25, ha="right")
    plt.ylabel("Average Time per Iteration (s)")
    plt.title("Average Iteration Time Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "avg_iteration_time_comparison.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(11, 5))
    iter_values = [results[m]["iter_count"] for m in MODEL_ORDER]
    plt.bar(x, iter_values, color=[MODEL_COLORS[m] for m in MODEL_ORDER])
    plt.xticks(x, MODEL_ORDER, rotation=25, ha="right")
    plt.ylabel("Total Iteration Count")
    plt.title("Iteration Count Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "iteration_count_comparison.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    y = [results[m]["opt_rel_err"] for m in MODEL_ORDER]
    plt.bar(x, y, color=[MODEL_COLORS[m] for m in MODEL_ORDER])
    plt.yscale("log")
    plt.xticks(x, MODEL_ORDER, rotation=25, ha="right")
    plt.ylabel("|J_model(z*) - J_true(z*)| / |J_true(z*)|")
    plt.title("Final Objective Relative Error")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "final_objective_rel_error.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    y = [results[m]["init_rel_err"] for m in MODEL_ORDER]
    plt.bar(x, y, color=[MODEL_COLORS[m] for m in MODEL_ORDER])
    plt.yscale("log")
    plt.xticks(x, MODEL_ORDER, rotation=25, ha="right")
    plt.ylabel("|J_model(z0) - J_true(z0)| / |J_true(z0)|")
    plt.title("Initial Objective Relative Error")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "initial_objective_rel_error.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    y = [results[m]["true_opt"] for m in MODEL_ORDER]
    plt.bar(x, y, color=[MODEL_COLORS[m] for m in MODEL_ORDER])
    plt.xticks(x, MODEL_ORDER, rotation=25, ha="right")
    plt.ylabel("J_true(z*)")
    plt.title("Final True Objective Value")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "final_true_objective_value.png"), dpi=180)
    plt.close()


def vector_to_function(function_space, vector):
    function = dl.Function(function_space)
    function.vector().zero()
    function.vector().axpy(1.0, vector)
    return function


def scalarize_for_plot(function, scalar_space):
    if function.function_space().num_sub_spaces() > 0:
        return dl.project(dl.sqrt(dl.inner(function, function)), scalar_space)
    return function


def plot_on_axes(function, ax):
    plt.sca(ax)
    return dl.plot(function)


def make_target_state_function(function_space):
    mesh = function_space.mesh()
    target_expr = dl.Expression(
        "sin(2*pi*x[0])*sin(2*pi*x[1])",
        pi=np.pi,
        degree=4,
        mpi_comm=mesh.mpi_comm(),
    )
    return dl.interpolate(target_expr, function_space)


def control_coefficients_to_grid(control_np, control_parameters):
    coeffs = np.asarray(control_np, dtype=float)
    n_wells = int(control_parameters.n_wells_per_side)
    if coeffs.size != n_wells * n_wells:
        raise ValueError(
            f"Expected {n_wells * n_wells} control coefficients, received {coeffs.size}."
        )

    well_grid = np.linspace(
        control_parameters.loc_lower,
        control_parameters.loc_upper,
        n_wells,
    )
    coeff_grid = np.zeros((n_wells, n_wells))
    count = 0
    for i in range(n_wells):
        for j in range(n_wells):
            coeff_grid[j, i] = coeffs[count]
            count += 1
    return well_grid, coeff_grid


def plot_control_coefficients_on_axes(control_np, control_parameters, ax):
    well_grid, coeff_grid = control_coefficients_to_grid(control_np, control_parameters)
    z_max = float(np.max(np.abs(coeff_grid)))
    if z_max == 0.0:
        z_max = 1.0
    width = (control_parameters.loc_upper - control_parameters.loc_lower) / control_parameters.n_wells_per_side
    extent = [
        well_grid[0] - 0.5 * width,
        well_grid[-1] + 0.5 * width,
        well_grid[0] - 0.5 * width,
        well_grid[-1] + 0.5 * width,
    ]
    artist = ax.imshow(
        coeff_grid,
        origin="lower",
        extent=extent,
        cmap=plt.get_cmap("coolwarm"),
        norm=Normalize(vmin=-z_max, vmax=z_max),
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return artist



def save_parameter_sample_plots(prior, Vh, save_dir, sample_count=3, seed=11):
    V_parameter = Vh[soupy.PARAMETER]
    V_parameter_scalar = dl.FunctionSpace(V_parameter.mesh(), "CG", 1)
    noise = dl.Vector(V_parameter.mesh().mpi_comm())
    prior.init_vector(noise, "noise")
    rng = hp.Random(seed=seed)

    fig, axes = plt.subplots(1, sample_count, figsize=(4.2 * sample_count, 3.6))
    axes = np.atleast_1d(axes)
    parameter_dofs = []
    for i, ax in enumerate(axes):
        m = prior.mean.copy()
        rng.normal(1.0, noise)
        prior.sample(noise, m)
        parameter_dofs.append(np.array(m.get_local(), copy=True))
        m_fun = scalarize_for_plot(vector_to_function(V_parameter, m), V_parameter_scalar)
        artist = plot_on_axes(m_fun, ax)
        ax.set_title(f"parameter sample {i + 1}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "parameter_samples.png"), dpi=180)
    plt.close(fig)
    np.savez(os.path.join(save_dir, "parameter_samples_data.npz"), seed=seed, parameter_dofs=np.array(parameter_dofs))

def solve_state_at_control(control_model, prior, z_np):
    x = control_model.generate_vector("ALL")
    x[soupy.PARAMETER].zero()
    x[soupy.PARAMETER].axpy(1.0, prior.mean)
    x[soupy.CONTROL].set_local(z_np)
    x[soupy.CONTROL].apply("")
    control_model.solveFwd(x[soupy.STATE], x)
    return x[soupy.CONTROL].copy(), x[soupy.STATE].copy()




def solve_state_at_parameter_control(control_model, parameter_vec, control_np):
    x = control_model.generate_vector("ALL")
    x[soupy.PARAMETER].zero()
    x[soupy.PARAMETER].axpy(1.0, parameter_vec)
    x[soupy.CONTROL].set_local(np.asarray(control_np))
    x[soupy.CONTROL].apply("")
    control_model.solveFwd(x[soupy.STATE], x)
    qoi = float(control_model.cost(x))
    return x[soupy.STATE].copy(), qoi


def save_saa_parameter_solution_sample_plots(results, saa_model_name, control_model, prior, Vh, save_dir, sample_count=3, seed=11):
    if saa_model_name not in results:
        return
    V_parameter = Vh[soupy.PARAMETER]
    V_state = Vh[soupy.STATE]
    V_parameter_scalar = dl.FunctionSpace(V_parameter.mesh(), "CG", 1)
    V_state_scalar = dl.FunctionSpace(V_state.mesh(), "CG", 1)
    control_np = results[saa_model_name].get("control_opt_np", results[saa_model_name].get("z_opt_np"))
    noise = dl.Vector(V_parameter.mesh().mpi_comm())
    prior.init_vector(noise, "noise")
    rng = hp.Random(seed=seed)

    fig, axes = plt.subplots(sample_count, 2, figsize=(9, 3.6 * sample_count))
    axes = np.atleast_2d(axes)
    qoi_values = []
    for i in range(sample_count):
        m = prior.mean.copy()
        rng.normal(1.0, noise)
        prior.sample(noise, m)
        state_vec, qoi = solve_state_at_parameter_control(control_model, m, control_np)
        qoi_values.append(float(qoi))
        m_fun = scalarize_for_plot(vector_to_function(V_parameter, m), V_parameter_scalar)
        state_fun = scalarize_for_plot(vector_to_function(V_state, state_vec), V_state_scalar)
        for ax, fun, title in zip(
            axes[i],
            [m_fun, state_fun],
            [f"{saa_model_name} sample {i + 1} parameter", f"solution at sample, QoI={qoi:.3e}"],
        ):
            artist = plot_on_axes(fun, ax)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(title)
            plt.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{saa_model_name}_parameter_solution_samples.png"), dpi=180)
    plt.close(fig)
    np.savez(
        os.path.join(save_dir, f"{saa_model_name}_parameter_solution_samples_data.npz"),
        seed=seed,
        control_np=np.array(control_np, copy=True),
        qoi_values=np.array(qoi_values),
    )


def save_solution_vs_linear_initial_plots(results, linear_initial_control_np, control_model, prior, Vh, save_dir):
    V_state = Vh[soupy.STATE]
    V_state_scalar = dl.FunctionSpace(V_state.mesh(), "CG", 1)
    _, init_state_vec = solve_state_at_control(control_model, prior, linear_initial_control_np)
    init_state_fun = scalarize_for_plot(vector_to_function(V_state, init_state_vec), V_state_scalar)

    fig, axes = plt.subplots(len(MODEL_ORDER), 2, figsize=(10.5, 3.6 * len(MODEL_ORDER)))
    axes = np.atleast_2d(axes)
    control_arrays = {}
    for i, model_name in enumerate(MODEL_ORDER):
        control_np = results[model_name].get("control_opt_np", results[model_name].get("z_opt_np"))
        control_arrays[model_name] = np.array(control_np, copy=True)
        _, opt_state_vec = solve_state_at_control(control_model, prior, control_np)
        opt_state_fun = scalarize_for_plot(vector_to_function(V_state, opt_state_vec), V_state_scalar)
        for ax, fun, title in zip(
            axes[i],
            [opt_state_fun, init_state_fun],
            [f"{model_name} solution at z*", "linear Taylor initial solution"],
        ):
            artist = plot_on_axes(fun, ax)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            # pad=16: the "linear Taylor initial solution" field is ~1e-15 (numerical
            # noise around a zero control), so matplotlib draws a small "1e-15" scale
            # annotation above the colorbar at the same height as a default title,
            # causing them to visually cross. Extra pad lifts the title clear of it.
            ax.set_title(title, pad=16)
            plt.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "optimal_vs_linear_initial_solutions.png"), dpi=180)
    plt.close(fig)
    np.savez(
        os.path.join(save_dir, "optimal_vs_linear_initial_solutions_data.npz"),
        model_order=np.array(MODEL_ORDER),
        linear_initial_control_np=np.array(linear_initial_control_np, copy=True),
        **{f"control_np__{name}": arr for name, arr in control_arrays.items()},
    )


def save_optimal_field_plots(results, control_model, prior, Vh, control_parameters, save_dir):
    with plt.rc_context(LARGE_FONT_RC):
        V_state = Vh[soupy.STATE]
        V_state_scalar = dl.FunctionSpace(V_state.mesh(), "CG", 1)
        target_fun = make_target_state_function(V_state_scalar)

        overview_payload = []

        for model_name in MODEL_ORDER:
            z_model_opt = results[model_name]["z_opt_np"]
            control_model_vec, state_model_vec = solve_state_at_control(control_model, prior, z_model_opt)
            control_np = np.array(control_model_vec.get_local(), copy=True)

            control_model_fun = control_coefficients_to_function(V_state_scalar, control_model_vec, control_parameters)
            state_model_fun = scalarize_for_plot(vector_to_function(V_state, state_model_vec), V_state_scalar)

            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            artist = plot_on_axes(control_model_fun, axes[0, 0])
            axes[0, 0].set_title(f"{model_name} source(z*)")
            axes[0, 0].set_xlabel("x")
            axes[0, 0].set_ylabel("y")
            plt.colorbar(artist, ax=axes[0, 0], fraction=0.046, pad=0.04)
            artist = plot_control_coefficients_on_axes(control_np, control_parameters, axes[0, 1])
            axes[0, 1].set_title(f"{model_name} well coefficients z*")
            plt.colorbar(artist, ax=axes[0, 1], fraction=0.046, pad=0.04)
            artist = plot_on_axes(state_model_fun, axes[1, 0])
            axes[1, 0].set_title(f"{model_name} u(z*)")
            axes[1, 0].set_xlabel("x")
            axes[1, 0].set_ylabel("y")
            plt.colorbar(artist, ax=axes[1, 0], fraction=0.046, pad=0.04)
            artist = plot_on_axes(target_fun, axes[1, 1])
            axes[1, 1].set_title("target state")
            axes[1, 1].set_xlabel("x")
            axes[1, 1].set_ylabel("y")
            plt.colorbar(artist, ax=axes[1, 1], fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, f"{model_name}_optimal_fields.png"), dpi=180)
            plt.close(fig)
            np.savez(os.path.join(save_dir, f"{model_name}_optimal_fields_data.npz"), control_np=control_np)
            overview_payload.append((model_name, control_np, control_model_fun, state_model_fun))

        n_models = len(overview_payload)
        fig, axes = plt.subplots(n_models, 4, figsize=(18, 3.6 * n_models))
        if n_models == 1:
            axes = np.array([axes])
        col_titles = ["optimal control source", "well coefficients z*", "state u(z*)", "target state"]
        for j, title in enumerate(col_titles):
            axes[0, j].set_title(title)
        for i, (model_name, control_np, control_model_fun, state_model_fun) in enumerate(overview_payload):
            plot_on_axes(control_model_fun, axes[i, 0])
            plot_control_coefficients_on_axes(control_np, control_parameters, axes[i, 1])
            plot_on_axes(state_model_fun, axes[i, 2])
            plot_on_axes(target_fun, axes[i, 3])
            for ax in axes[i, :]:
                ax.set_xticks([])
                ax.set_yticks([])
            axes[i, 0].set_ylabel(model_name, rotation=90, fontsize=16)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "optimal_fields_overview.png"), dpi=180)
        plt.close(fig)
    np.savez(
        os.path.join(save_dir, "optimal_fields_overview_data.npz"),
        model_order=np.array([m for m, _, _, _ in overview_payload]),
        **{f"control_np__{m}": c for m, c, _, _ in overview_payload},
    )


def save_optimal_pde_solution_plot(results, control_model, prior, Vh, save_dir):
    """Plot PDE solutions at each model's optimal control next to the target state."""
    V_state = Vh[soupy.STATE]
    V_state_scalar = dl.FunctionSpace(V_state.mesh(), "CG", 1)
    target_fun = make_target_state_function(V_state_scalar)

    solution_payload = []
    for model_name in MODEL_ORDER:
        _, state_model_vec = solve_state_at_control(
            control_model,
            prior,
            results[model_name]["z_opt_np"],
        )
        state_model_fun = scalarize_for_plot(vector_to_function(V_state, state_model_vec), V_state_scalar)
        solution_payload.append((model_name, state_model_fun))

    fig, axes = plt.subplots(len(solution_payload), 2, figsize=(9, 3.6 * len(solution_payload)))
    axes = np.atleast_2d(axes)

    for i, (model_name, state_model_fun) in enumerate(solution_payload):
        for ax, fun, title in zip(axes[i], [state_model_fun, target_fun], [f"{model_name} u(z*)", "target state"]):
            artist = plot_on_axes(fun, ax)
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            plt.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "optimal_pde_solutions.png"), dpi=180)
    plt.close(fig)
    np.savez(
        os.path.join(save_dir, "optimal_pde_solutions_data.npz"),
        model_order=np.array([m for m, _ in solution_payload]),
        **{f"control_np__{model_name}": results[model_name]["z_opt_np"] for model_name in MODEL_ORDER},
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare Taylor models against mean-variance SAA truth on semilinear ADR control"
    )
    parser.add_argument("--beta", type=float, default=1.0, help="Variance weight beta")
    parser.add_argument("--n-tr", type=int, default=50, help="Number of Hessian modes for quadratic models")
    parser.add_argument("--n-mix", type=int, default=11, help="Number of mixture components")
    parser.add_argument("--truth-saa-samples", type=int, default=100000, help="Sample size for ground-truth SAA evaluation")
    parser.add_argument("--saa-seed", type=int, default=1, help="Seed for SAA sampling")
    parser.add_argument("--maxiter", type=int, default=600, help="Max iterations for each Taylor model")
    parser.add_argument("--maxiter-saa", type=int, default=600, help="Max iterations for optimized SAA models")
    parser.add_argument("--nx", type=int, default=64, help="Mesh cells in x")
    parser.add_argument("--ny", type=int, default=64, help="Mesh cells in y")
    parser.add_argument("--newton-max-it", type=int, default=50)
    parser.add_argument("--newton-rtol", type=float, default=1e-8)
    parser.add_argument("--newton-atol", type=float, default=1e-10)
    parser.add_argument("--print-every", type=int, default=1, help="Print every N iterations")
    parser.add_argument("--save-dir", type=str, default="results_semilinear_adr_compare_taylor_models", help="Output directory")
    parser.add_argument(
        "--log-file",
        type=str,
        default="terminal_output.txt",
        help="Log file name (or path) for terminal outputs",
    )
    parser.add_argument("--bound-lb", type=float, default=-4.0, help="Lower bound for each Gaussian-well control coefficient")
    parser.add_argument("--bound-ub", type=float, default=4.0, help="Upper bound for each Gaussian-well control coefficient")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Verbose model output")
    args = parser.parse_args()

    rank = MPI.COMM_WORLD.Get_rank()
    comm_mesh = MPI.COMM_SELF

    os.makedirs(args.save_dir, exist_ok=True)

    log_path = args.log_file
    if not os.path.isabs(log_path):
        log_path = os.path.join(args.save_dir, log_path)

    log_file = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if rank == 0:
        log_file = open(log_path, "w", buffering=1)
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)

    try:
        if rank == 0:
            print("=" * 90)
            print("Semilinear ADR Model Comparison:")
            print("  linear / quadratic / mixture_linear_kle / mixture_linear_hep")
            print("  mixture_quadratic_kle / mixture_quadratic_hep")
            print("  saa_1 / saa_10 / saa_39 / saa_100 / saa_200 / saa_500 / saa_1000 / saa_2000 / saa_5000 / saa_10000")
            print("  serial: linear / quadratic / saa_1 / saa_10")
            print("  parallel on all ranks: mixture_* / saa_39 / saa_100 / saa_200 / saa_500 / saa_1000 / saa_2000 / saa_5000 / saa_10000")
            print("=" * 90)
            print(f"Ground-truth SAA samples: {args.truth_saa_samples}")
            print(f"beta={args.beta}, n_tr={args.n_tr}, n_mix={args.n_mix}")
            print(f"box constraint on control: {args.bound_lb} <= z <= {args.bound_ub}")
            print("control source term: rhs = sum_i z_i psi_i (no known source f)")
            print("initial control: zero well coefficients")
            print(f"terminal log file: {log_path}")
            print("=" * 90)
            sys.stdout.flush()

        problem = setup_problem(args, comm_mesh)
        Vh = problem["Vh"]
        control_model = problem["control_model"]
        prior = problem["prior"]
        control_parameters = problem["control_parameters"]
        control0_np = control_model.generate_vector(soupy.CONTROL).get_local()
        args.control_dim = int(control0_np.size)

        results: Dict[str, Dict] = {}
        for model_name in MODEL_ORDER:
            approx_cost = None
            z0_np = None
            approx_init = None
            run_parallel_model = model_uses_world_parallel(model_name)

            if run_parallel_model:
                approx_cost = make_taylor_cost(model_name, control_model, prior, args)
                z0_np = zero_control_np(approx_cost)
                approx_init = evaluate_cost(approx_cost, z0_np)
            elif rank == 0:
                approx_cost = make_taylor_cost(model_name, control_model, prior, args)
                z0_np = zero_control_np(approx_cost)
                approx_init = evaluate_cost(approx_cost, z0_np)

            z0_np = MPI.COMM_WORLD.bcast(z0_np if rank == 0 else None, root=0)
            approx_init = MPI.COMM_WORLD.bcast(approx_init if rank == 0 else None, root=0)
            truth_cost = make_saa_cost(
                control_model,
                prior,
                beta=args.beta,
                sample_size=args.truth_saa_samples,
                seed=args.saa_seed,
                comm_sampler=MPI.COMM_WORLD,
            )
            true_init, true_init_mean, true_init_var = evaluate_true_cost_stats(truth_cost, z0_np)
            del truth_cost
            init_rel_err = relative_error(approx_init, true_init)

            if rank == 0:
                print(f"\nOptimizing {model_name} with L-BFGS-B ...")
                print(
                    f"  [{model_name:20s}] initial: "
                    f"J_model(z0)={approx_init:.6e}, "
                    f"J_true(z0)={true_init:.6e}, "
                    f"mean_true(z0)={true_init_mean:.6e}, "
                    f"var_true(z0)={true_init_var:.6e}, "
                    f"rel_err(z0)={init_rel_err:.3e}"
                )
                sys.stdout.flush()

            bounds = make_bounds(z0_np, args) if (run_parallel_model or rank == 0) else None
            res = optimize_with_tracking(
                model_name,
                approx_cost,
                args,
                rank,
                maxiter=args.maxiter_saa if model_name.startswith("saa_") else args.maxiter,
                root_only=not run_parallel_model,
                bounds=bounds,
            )
            res["approx_init"] = float(approx_init)
            res["true_init"] = float(true_init)
            res["true_init_mean"] = float(true_init_mean)
            res["true_init_var"] = float(true_init_var)
            res["init_rel_err"] = float(init_rel_err)
            truth_cost = make_saa_cost(
                control_model,
                prior,
                beta=args.beta,
                sample_size=args.truth_saa_samples,
                seed=args.saa_seed,
                comm_sampler=MPI.COMM_WORLD,
            )
            evaluate_truth_for_result(res, truth_cost)
            del truth_cost
            results[model_name] = res
            if approx_cost is not None:
                del approx_cost

            if rank == 0:
                print(
                    f"  [{model_name:20s}] optimal: "
                    f"J_model(z*)={res['approx_opt']:.6e}, "
                    f"J_true(z*)={res['true_opt']:.6e}, "
                    f"mean_true(z*)={res['true_opt_mean']:.6e}, "
                    f"var_true(z*)={res['true_opt_var']:.6e}, "
                    f"rel_err(z*)={res['opt_rel_err']:.3e}"
                )
                sys.stdout.flush()

        if rank == 0:
            for model_name in MODEL_ORDER:
                save_iteration_csv(
                    os.path.join(args.save_dir, f"{model_name}_iteration_metrics.csv"),
                    results[model_name]["iter_records"],
                )

            summary_rows = []
            for model_name in MODEL_ORDER:
                rr = results[model_name]
                summary_rows.append(
                    [
                        model_name,
                        rr["success"],
                        rr["message"],
                        rr["iter_count"],
                        rr["nfev"],
                        rr["njev"],
                        rr["avg_iter_time_sec"],
                        rr["approx_init"],
                        rr["true_init"],
                        rr["init_rel_err"],
                        rr["approx_opt"],
                        rr["true_opt"],
                        rr["opt_rel_err"],
                    ]
                )

            save_summary_csv(os.path.join(args.save_dir, "summary.csv"), summary_rows)

            json_payload = {
                "config": vars(args),
                "problem": {
                    "mesh": {
                        "nx": int(problem["mesh_parameters"].nx),
                        "ny": int(problem["mesh_parameters"].ny),
                    },
                    "qoi_type": problem["qoi_type"],
                    "control_source": "sum_i z_i psi_i",
                    "control_bounds": {"lb": float(args.bound_lb), "ub": float(args.bound_ub)},
                    "penalization": None,
                },
                "truth_evaluation": {
                    "sample_size": int(args.truth_saa_samples),
                    "seed": int(args.saa_seed),
                },
                "models": {
                    name: {
                        "success": bool(results[name]["success"]),
                        "message": str(results[name]["message"]),
                        "nit": int(results[name]["iter_count"]),
                        "nfev": int(results[name]["nfev"]),
                        "njev": int(results[name]["njev"]),
                        "iter_count": int(results[name]["iter_count"]),
                        "avg_iter_time_sec": float(results[name]["avg_iter_time_sec"]),
                        "initial_model_objective": float(results[name]["approx_init"]),
                        "initial_true_objective": float(results[name]["true_init"]),
                        "initial_rel_error": float(results[name]["init_rel_err"]),
                        "opt_model_objective": float(results[name]["approx_opt"]),
                        "opt_true_objective": float(results[name]["true_opt"]),
                        "opt_rel_error": float(results[name]["opt_rel_err"]),
                    }
                    for name in MODEL_ORDER
                },
            }
            with open(os.path.join(args.save_dir, "summary.json"), "w") as f:
                json.dump(json_payload, f, indent=2)

            plot_curves(results, args.save_dir)
            save_optimal_field_plots(results, control_model, prior, Vh, control_parameters, args.save_dir)
            save_optimal_pde_solution_plot(results, control_model, prior, Vh, args.save_dir)
            save_saa_parameter_solution_sample_plots(results, "saa_10000", control_model, prior, Vh, args.save_dir)
            save_solution_vs_linear_initial_plots(results, control0_np, control_model, prior, Vh, args.save_dir)

            timing_rows = []
            for model_name in MODEL_ORDER:
                rr = results[model_name]
                timing_rows.append([model_name, rr["iter_count"], rr["avg_iter_time_sec"]])
            with open(os.path.join(args.save_dir, "timing_comparison.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["model", "total_iterations", "avg_iter_time_sec"])
                writer.writerows(timing_rows)

            print("\n" + "-" * 90)
            print("Summary (ground-truth SAA evaluated at z0 and z*)")
            print("-" * 90)
            for model_name in MODEL_ORDER:
                rr = results[model_name]
                print(
                    f"{model_name:20s} | nit={rr['iter_count']:3d} | "
                    f"avg_iter_time={rr['avg_iter_time_sec']:8.2f}s | "
                    f"J_model(z*)={rr['approx_opt']:.6e} | "
                    f"init rel_err={rr['init_rel_err']:.3e} | "
                    f"opt rel_err={rr['opt_rel_err']:.3e} | "
                    f"J_true(z*)={rr['true_opt']:.6e}"
                )
            print("-" * 90)
            print(f"All outputs written to: {args.save_dir}")
            print(f"Terminal outputs saved to: {log_path}")
    finally:
        if rank == 0 and log_file is not None:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()


if __name__ == "__main__":
    main()
