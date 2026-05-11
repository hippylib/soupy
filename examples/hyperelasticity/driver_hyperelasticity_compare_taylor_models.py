"""Compare Taylor and SAA models on hyperelasticity control.

Models compared:
- linear
- quadratic
- mixture_linear_kle
- mixture_linear_hep
- mixture_quadratic_kle
- mixture_quadratic_hep
- saa_10
- saa_100
- saa_200
- saa_500
- saa_1000
- saa_10000

Reference truth:
- Mean-variance SAA objective with a large fixed sample size
- Evaluated only at the initial control and each model optimum

Metrics:
1) Optimization time comparison:
      - average time per iteration
      - total iteration count
2) Initial/final objective approximation relative error:
      |J_model(z) - J_truth(z)| / |J_truth(z)|
3) Initial/final true objective values
4) Final optimal control and corresponding state plots

No MC correction is used in Taylor-based models.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List

# Configure macOS compiler BEFORE importing dolfin or soupy
_SOUPY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, os.path.join(_SOUPY_ROOT, "soupy", "utils"))
try:
    from macos_config import configure_macos_compiler, configure_dolfin_form_compiler

    configure_macos_compiler()
except ImportError:
    configure_dolfin_form_compiler = None
sys.path.pop(0)

sys.path.append(os.environ.get("HIPPYLIB_PATH", ""))
sys.path.append(_SOUPY_ROOT)

import dolfin as dl
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from mpi4py import MPI

import soupy
from soupy import MeanVarRiskMeasureSAA, RiskMeasureControlCostFunctional, meanVarRiskMeasureSAASettings
from soupy.approximations.taylor import (
    TaylorLinearControlCostFunctional,
    TaylorMixtureLinearControlCostFunctional,
    TaylorMixtureQuadraticControlCostFunctional,
    TaylorQuadraticControlCostFunctional,
)

from setupHyperelasticityProblem import hyperelasticity_problem_settings, setup_hyperelasticity_problem


logging.getLogger("FFC").setLevel(logging.WARNING)
logging.getLogger("UFL").setLevel(logging.WARNING)
dl.set_log_active(False)
if configure_dolfin_form_compiler is not None:
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
    "saa_10",
    "saa_100",
    "saa_200",
    "saa_500",
    "saa_1000",
    "saa_10000",
]
MODEL_COLORS = {
    "linear": "tab:blue",
    "quadratic": "tab:orange",
    "mixture_linear_kle": "tab:green",
    "mixture_linear_hep": "tab:olive",
    "mixture_quadratic_kle": "tab:red",
    "mixture_quadratic_hep": "tab:brown",
    "saa_10": "#17becf",
    "saa_100": "#7f7f7f",
    "saa_200": "#9467bd",
    "saa_500": "#e377c2",
    "saa_1000": "#8c564b",
    "saa_10000": "#bcbd22",
}


class TeeStream:
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
        # Linux reports KB, macOS reports bytes.
        if sys.platform == "darwin":
            return float(peak_kb) / (1024.0 ** 2)
        return float(peak_kb) / 1024.0
    except Exception:
        return float("nan")


class ScipyObjectiveWithHistory:
    def __init__(self, cost_functional):
        self.cost_functional = cost_functional
        self._z = cost_functional.generate_vector(soupy.CONTROL)
        self._g = cost_functional.generate_vector(soupy.CONTROL)
        self.latest_cost = np.nan
        self.latest_grad_norm = np.nan
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
            self.latest_grad_rss_after_mb = get_current_rss_mb()
            self.n_grad += 1
            return self._g.get_local()

        return g


def setup_problem(args, comm_mesh):
    settings = hyperelasticity_problem_settings()
    settings["qoi_type"] = args.qoi_type
    settings["geometry"]["lx"] = args.lx
    settings["geometry"]["ly"] = args.ly
    settings["geometry"]["lz"] = args.lz
    settings["geometry"]["dim"] = args.geometry_dim
    settings["mesh"]["nx"] = args.nx
    settings["mesh"]["ny"] = args.ny
    settings["mesh"]["nz"] = args.nz
    mesh, Vh, _, control_model, prior = setup_hyperelasticity_problem(settings, comm_mesh)
    penalty = soupy.L2Penalization(Vh, args.penalty)
    return mesh, Vh, control_model, prior, penalty


def make_saa_cost(control_model, prior, penalty, beta, sample_size, seed, comm_sampler):
    settings = meanVarRiskMeasureSAASettings()
    settings["beta"] = beta
    settings["sample_size"] = sample_size
    settings["seed"] = seed
    risk = MeanVarRiskMeasureSAA(control_model, prior, settings=settings, comm_sampler=comm_sampler)
    return RiskMeasureControlCostFunctional(risk, penalty)


def model_uses_world_parallel(model_name: str) -> bool:
    if model_name.startswith("mixture_"):
        return True
    if model_name.startswith("saa_"):
        return int(model_name.split("_", 1)[1]) >= 100
    return False


def make_taylor_cost(model_name, control_model, prior, penalty, args):
    if model_name.startswith("saa_"):
        sample_size = int(model_name.split("_", 1)[1])
        return make_saa_cost(
            control_model,
            prior,
            penalty,
            beta=args.beta,
            sample_size=sample_size,
            seed=args.saa_seed,
            comm_sampler=MPI.COMM_WORLD if sample_size >= 100 else MPI.COMM_SELF,
        )

    if model_name == "linear":
        settings = {
            "beta": args.beta,
            "correction": False,
            "N_mc": 0,
            "verbose": args.verbose,
        }
        return TaylorLinearControlCostFunctional(control_model, prior, penalty, settings)

    if model_name == "quadratic":
        settings = {
            "beta": args.beta,
            "N_tr": args.n_tr,
            "correction": False,
            "N_mc": 0,
            "verbose": args.verbose,
        }
        return TaylorQuadraticControlCostFunctional(control_model, prior, penalty, settings)

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
            penalty,
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
            penalty,
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
            penalty,
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
            penalty,
            settings,
            comm_sampler=MPI.COMM_WORLD,
        )

    raise ValueError(f"Unknown model name: {model_name}")


def np_to_control(cost_functional, z_np):
    z = cost_functional.generate_vector(soupy.CONTROL)
    z.set_local(z_np)
    z.apply("")
    return z


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


def optimize_with_tracking(model_name, approx_cost, args, rank, maxiter, bounds=None, root_only=False):
    payload = None

    if (not root_only) or rank == 0:
        z0 = approx_cost.generate_vector(soupy.CONTROL)
        x0 = z0.get_local()
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
                    residual=float(wrapper.latest_grad_norm),
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
            options={"maxiter": maxiter, "disp": False},
        )
        total_time = time.perf_counter() - t0
        iter_count = len(iter_records) if iter_records else int(result.nit)
        avg_iter_time_sec = total_time / max(iter_count, 1)

        for r in iter_records:
            if rank == 0 and r.iteration % args.print_every == 0:
                print(
                    f"  [{model_name:20s}] iter {r.iteration:4d}: "
                    f"J_model={r.model_cost:.6e}, ||g||={r.residual:.3e}, "
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
                "residual",
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
    plt.figure(figsize=(8, 5))
    for model in MODEL_ORDER:
        rec = results[model]["iter_records"]
        if not rec:
            continue
        x = [r.iteration for r in rec]
        y = [max(abs(r.model_cost), 1e-16) for r in rec]
        plt.semilogy(x, y, marker="o", linewidth=1.5, markersize=3, color=MODEL_COLORS[model], label=model)
    plt.xlabel("Iteration")
    plt.ylabel("Model Objective")
    plt.title("Objective Value per Iteration")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "objective_per_iteration.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for model in MODEL_ORDER:
        rec = results[model]["iter_records"]
        if not rec:
            continue
        x = [r.iteration for r in rec]
        y = [max(r.residual, 1e-16) for r in rec]
        plt.semilogy(x, y, marker="o", linewidth=1.5, markersize=3, color=MODEL_COLORS[model], label=model)
    plt.xlabel("Iteration")
    plt.ylabel("Residual ||grad||")
    plt.title("Residual per Iteration")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "residual_per_iteration.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    x = np.arange(len(MODEL_ORDER))
    time_values = [results[m]["avg_iter_time_sec"] for m in MODEL_ORDER]
    plt.bar(x, time_values, color=[MODEL_COLORS[m] for m in MODEL_ORDER])
    plt.xticks(x, MODEL_ORDER, rotation=20, ha="right")
    plt.ylabel("Average Time per Iteration (s)")
    plt.title("Average Iteration Time Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "avg_iteration_time_comparison.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    x = np.arange(len(MODEL_ORDER))
    iter_values = [results[m]["iter_count"] for m in MODEL_ORDER]
    plt.bar(x, iter_values, color=[MODEL_COLORS[m] for m in MODEL_ORDER])
    plt.xticks(x, MODEL_ORDER, rotation=20, ha="right")
    plt.ylabel("Total Iteration Count")
    plt.title("Iteration Count Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "iteration_count_comparison.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    x = np.arange(len(MODEL_ORDER))
    y = [results[m]["opt_rel_err"] for m in MODEL_ORDER]
    plt.bar(x, y, color=[MODEL_COLORS[m] for m in MODEL_ORDER])
    plt.yscale("log")
    plt.xticks(x, MODEL_ORDER, rotation=15)
    plt.ylabel("|J_model(z*) - J_true(z*)| / |J_true(z*)|")
    plt.title("Final Objective Relative Error")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "final_objective_rel_error.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    x = np.arange(len(MODEL_ORDER))
    y = [results[m]["init_rel_err"] for m in MODEL_ORDER]
    plt.bar(x, y, color=[MODEL_COLORS[m] for m in MODEL_ORDER])
    plt.yscale("log")
    plt.xticks(x, MODEL_ORDER, rotation=15)
    plt.ylabel("|J_model(z0) - J_true(z0)| / |J_true(z0)|")
    plt.title("Initial Objective Relative Error")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "initial_objective_rel_error.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    x = np.arange(len(MODEL_ORDER))
    y = [results[m]["true_opt"] for m in MODEL_ORDER]
    plt.bar(x, y, color=[MODEL_COLORS[m] for m in MODEL_ORDER])
    plt.xticks(x, MODEL_ORDER, rotation=15)
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


def solve_state_at_control(control_model, prior, z_np):
    x = control_model.generate_vector("ALL")
    x[soupy.PARAMETER].zero()
    x[soupy.PARAMETER].axpy(1.0, prior.mean)
    x[soupy.CONTROL].set_local(z_np)
    x[soupy.CONTROL].apply("")
    control_model.solveFwd(x[soupy.STATE], x)
    return x[soupy.CONTROL].copy(), x[soupy.STATE].copy()


def save_optimal_field_plots(results, control_model, prior, Vh, save_dir):
    V_control = Vh[soupy.CONTROL]
    V_state = Vh[soupy.STATE]
    V_state_scalar = dl.FunctionSpace(V_state.mesh(), "CG", 1)

    overview_payload = []

    for model_name in MODEL_ORDER:
        z_model_opt = results[model_name]["z_opt_np"]
        control_model_vec, state_model_vec = solve_state_at_control(control_model, prior, z_model_opt)

        control_model_fun = vector_to_function(V_control, control_model_vec)
        state_model_fun = scalarize_for_plot(vector_to_function(V_state, state_model_vec), V_state_scalar)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, fun, title in zip(
            axes,
            [control_model_fun, state_model_fun],
            [f"{model_name} z*", f"{model_name} |u(z*)|"],
        ):
            artist = plot_on_axes(fun, ax)
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            plt.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{model_name}_optimal_fields.png"), dpi=180)
        plt.close(fig)
        overview_payload.append((model_name, control_model_fun, state_model_fun))

    n_models = len(overview_payload)
    fig, axes = plt.subplots(n_models, 2, figsize=(10, 3.6 * n_models))
    if n_models == 1:
        axes = np.array([axes])
    col_titles = ["optimal control z*", "state |u(z*)|"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title)
    for i, (model_name, control_model_fun, state_model_fun) in enumerate(overview_payload):
        fields = [control_model_fun, state_model_fun]
        for j, fun in enumerate(fields):
            ax = axes[i, j]
            plot_on_axes(fun, ax)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[i, 0].set_ylabel(model_name, rotation=90, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "optimal_fields_overview.png"), dpi=180)
    plt.close(fig)


def save_optimal_pde_solution_plot(results, control_model, prior, Vh, save_dir):
    """Plot PDE solutions at each model's optimal control."""
    V_state = Vh[soupy.STATE]
    V_state_scalar = dl.FunctionSpace(V_state.mesh(), "CG", 1)

    solution_payload = []
    for model_name in MODEL_ORDER:
        _, state_model_vec = solve_state_at_control(
            control_model,
            prior,
            results[model_name]["z_opt_np"],
        )
        state_model_fun = scalarize_for_plot(vector_to_function(V_state, state_model_vec), V_state_scalar)
        solution_payload.append((model_name, state_model_fun))

    n_models = len(solution_payload)
    ncols = 3
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax in axes.ravel():
        ax.axis("off")

    for ax, (model_name, state_model_fun) in zip(axes.ravel(), solution_payload):
        ax.axis("on")
        artist = plot_on_axes(state_model_fun, ax)
        ax.set_title(f"{model_name} |u(z*)|")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "optimal_pde_solutions.png"), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare Taylor models against SAA truth on hyperelasticity")
    parser.add_argument("--beta", type=float, default=1.0, help="Variance weight beta")
    parser.add_argument("--n-tr", type=int, default=10, help="Number of Hessian modes for quadratic models")
    parser.add_argument("--n-mix", type=int, default=39, help="Number of mixture components")
    parser.add_argument(
        "--truth-saa-samples",
        "--saa-samples",
        dest="truth_saa_samples",
        type=int,
        default=100000,
        help="Sample size for ground-truth SAA evaluation",
    )
    parser.add_argument("--saa-seed", type=int, default=1, help="Seed for SAA sampling")
    parser.add_argument("--qoi-type", type=str, default="virtual_work",
                        choices=["all", "stiffness", "point", "virtual_work"])
    parser.add_argument("--penalty", type=float, default=1e-2, help="Control penalty")
    parser.add_argument("--maxiter", type=int, default=60, help="Max iterations for each Taylor model")
    parser.add_argument("--maxiter-saa", type=int, default=60, help="Max iterations for optimized SAA models")
    parser.add_argument("--nx", type=int, default=32, help="Mesh cells in x")
    parser.add_argument("--ny", type=int, default=8, help="Mesh cells in y")
    parser.add_argument("--nz", type=int, default=8, help="Mesh cells in z for 3D settings")
    parser.add_argument("--lx", type=float, default=2.0, help="Beam length in x")
    parser.add_argument("--ly", type=float, default=0.5, help="Beam length in y")
    parser.add_argument("--lz", type=float, default=0.25, help="Beam length in z for 3D settings")
    parser.add_argument("--geometry-dim", type=int, default=2, choices=[2, 3], help="Geometry dimension")
    parser.add_argument("--print-every", type=int, default=1, help="Print every N iterations")
    parser.add_argument("--save-dir", type=str, default="results_compare_taylor_models", help="Output directory")
    parser.add_argument(
        "--log-file",
        type=str,
        default="terminal_output.txt",
        help="Log file name (or path) for terminal outputs",
    )
    parser.add_argument("--bound-lb", type=float, default=0.0, help="Lower bound for L-BFGS-B controls")
    parser.add_argument("--bound-ub", type=float, default=1.0, help="Upper bound for L-BFGS-B controls")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Verbose model output")
    args = parser.parse_args()

    rank = MPI.COMM_WORLD.Get_rank()
    comm_mesh = MPI.COMM_SELF
    os.makedirs(args.save_dir, exist_ok=True)

    log_path = args.log_file
    if not os.path.isabs(log_path):
        log_path = os.path.join(args.save_dir, log_path)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = None
    if rank == 0:
        log_file = open(log_path, "w", buffering=1)
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)

    try:
        if rank == 0:
            print("=" * 78)
            print("Hyperelasticity Model Comparison:")
            print("  linear / quadratic / mixture_linear_kle / mixture_linear_hep")
            print(
                "  mixture_quadratic_kle / mixture_quadratic_hep / "
                "saa_10 / saa_100 / saa_200 / saa_500 / saa_1000 / saa_10000"
            )
            print("  serial: linear / quadratic / saa_10")
            print("  parallel on all ranks: mixture_* / saa_100 / saa_200 / saa_500 / saa_1000 / saa_10000")
            print("=" * 78)
            print(f"Ground-truth SAA samples: {args.truth_saa_samples}")
            print(f"beta={args.beta}, n_tr={args.n_tr}, n_mix={args.n_mix}")
            print(f"mesh={args.nx}x{args.ny}" + (f"x{args.nz}" if args.geometry_dim == 3 else ""))
            print(f"terminal log file: {log_path}")
            print("=" * 78)
            sys.stdout.flush()

        _, Vh, control_model, prior, penalty = setup_problem(args, comm_mesh)
        bounds = scipy.optimize.Bounds(lb=args.bound_lb, ub=args.bound_ub)

        # Compare all optimization models. Ground truth is built and released model-by-model
        # to avoid retaining the full SAA sample cache across the entire run.
        results: Dict[str, Dict] = {}
        for model_name in MODEL_ORDER:
            approx_cost = None
            z0_np = None
            approx_init = None
            run_parallel_model = model_uses_world_parallel(model_name)

            if run_parallel_model:
                approx_cost = make_taylor_cost(model_name, control_model, prior, penalty, args)
                z0_np = approx_cost.generate_vector(soupy.CONTROL).get_local()
                approx_init = evaluate_cost(approx_cost, z0_np)
            elif rank == 0:
                approx_cost = make_taylor_cost(model_name, control_model, prior, penalty, args)
                z0_np = approx_cost.generate_vector(soupy.CONTROL).get_local()
                approx_init = evaluate_cost(approx_cost, z0_np)

            z0_np = MPI.COMM_WORLD.bcast(z0_np if rank == 0 else None, root=0)
            approx_init = MPI.COMM_WORLD.bcast(approx_init if rank == 0 else None, root=0)

            truth_cost = make_saa_cost(
                control_model,
                prior,
                penalty,
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

            res = optimize_with_tracking(
                model_name,
                approx_cost,
                args,
                rank,
                maxiter=args.maxiter_saa if model_name.startswith("saa_") else args.maxiter,
                bounds=bounds,
                root_only=not run_parallel_model,
            )
            res["approx_init"] = float(approx_init)
            res["true_init"] = float(true_init)
            res["true_init_mean"] = float(true_init_mean)
            res["true_init_var"] = float(true_init_var)
            res["init_rel_err"] = float(init_rel_err)

            truth_cost = make_saa_cost(
                control_model,
                prior,
                penalty,
                beta=args.beta,
                sample_size=args.truth_saa_samples,
                seed=args.saa_seed,
                comm_sampler=MPI.COMM_WORLD,
            )
            res["true_opt"], res["true_opt_mean"], res["true_opt_var"] = evaluate_true_cost_stats(
                truth_cost, res["z_opt_np"]
            )
            res["opt_rel_err"] = relative_error(res["approx_opt"], res["true_opt"])
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
                save_iteration_csv(os.path.join(args.save_dir, f"{model_name}_iteration_metrics.csv"), results[model_name]["iter_records"])

            summary_rows = []
            for model_name in MODEL_ORDER:
                rr = results[model_name]
                summary_rows.append([
                    model_name,
                    rr["success"],
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
                ])
            save_summary_csv(os.path.join(args.save_dir, "summary.csv"), summary_rows)

            payload = {
                "config": vars(args),
                "truth_evaluation": {
                    "sample_size": int(args.truth_saa_samples),
                    "seed": int(args.saa_seed),
                },
                "models": {
                    name: {
                        "success": bool(results[name]["success"]),
                        "nit": int(results[name]["iter_count"]),
                        "nfev": int(results[name]["nfev"]),
                        "njev": int(results[name]["njev"]),
                        "iter_count": int(results[name]["iter_count"]),
                        "avg_iter_time_sec": float(results[name]["avg_iter_time_sec"]),
                        "initial_model_objective": float(results[name]["approx_init"]),
                        "initial_true_objective": float(results[name]["true_init"]),
                        "initial_true_mean": float(results[name]["true_init_mean"]),
                        "initial_true_variance": float(results[name]["true_init_var"]),
                        "initial_rel_error": float(results[name]["init_rel_err"]),
                        "opt_model_objective": float(results[name]["approx_opt"]),
                        "opt_true_objective": float(results[name]["true_opt"]),
                        "opt_true_mean": float(results[name]["true_opt_mean"]),
                        "opt_true_variance": float(results[name]["true_opt_var"]),
                        "opt_rel_error": float(results[name]["opt_rel_err"]),
                    }
                    for name in MODEL_ORDER
                },
            }
            with open(os.path.join(args.save_dir, "summary.json"), "w") as f:
                json.dump(payload, f, indent=2)

            plot_curves(results, args.save_dir)
            save_optimal_field_plots(results, control_model, prior, Vh, args.save_dir)
            save_optimal_pde_solution_plot(results, control_model, prior, Vh, args.save_dir)

            timing_rows = []
            for model_name in MODEL_ORDER:
                rr = results[model_name]
                timing_rows.append([model_name, rr["iter_count"], rr["avg_iter_time_sec"]])
            with open(os.path.join(args.save_dir, "timing_comparison.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["model", "total_iterations", "avg_iter_time_sec"])
                writer.writerows(timing_rows)

            print("\n" + "-" * 78)
            print("Summary (ground-truth SAA evaluated at z0 and z*)")
            print("-" * 78)
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
            print("-" * 78)
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
