"""Compare Taylor and SAA models on semilinear ADR control.

This mirrors the workflow/style of
`examples/poisson/driver_poisson_compare_taylor_models.py`, adapted to the
semilinear ADR problem. The control enters as an additive source term, so the
PDE right hand side is changed from f to f + z. The initial control is zero
everywhere.

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
import numpy as np
import scipy.optimize
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


class ScipyObjectiveWithHistory:
    """Wrap a cost functional for scipy minimize and keep latest value/grad norm."""

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
    mesh_parameters = MeshParameters()
    pde_parameters = PDEParameters()
    prior_parameters = PriorParameters()
    qoi_type = "l2"

    if args.nx is not None:
        mesh_parameters.nx = args.nx
    if args.ny is not None:
        mesh_parameters.ny = args.ny

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
        return int(model_name.split("_", 1)[1]) >= 100
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
            comm_sampler=MPI.COMM_WORLD if sample_size >= 100 else MPI.COMM_SELF,
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


def optimize_with_tracking(model_name, approx_cost, args, rank, maxiter, root_only=False):
    payload = None

    if (not root_only) or rank == 0:
        x0 = zero_control_np(approx_cost)
        wrapper = ScipyObjectiveWithHistory(approx_cost)
        iter_records: List[IterRecord] = []
        callback_last_time = None

        def callback(xk):
            del xk
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
    plt.figure(figsize=(9, 5))
    for model in MODEL_ORDER:
        rec = results[model]["iter_records"]
        if not rec:
            continue
        x = [r.iteration for r in rec]
        y = [max(abs(r.model_cost), 1e-16) for r in rec]
        plt.semilogy(
            x,
            y,
            marker="o",
            linewidth=1.5,
            markersize=3,
            color=MODEL_COLORS[model],
            label=model,
        )
    plt.xlabel("Iteration")
    plt.ylabel("Model Objective")
    plt.title("Objective Value per Iteration")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "objective_per_iteration.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    for model in MODEL_ORDER:
        rec = results[model]["iter_records"]
        if not rec:
            continue
        x = [r.iteration for r in rec]
        y = [max(r.residual, 1e-16) for r in rec]
        plt.semilogy(
            x,
            y,
            marker="o",
            linewidth=1.5,
            markersize=3,
            color=MODEL_COLORS[model],
            label=model,
        )
    plt.xlabel("Iteration")
    plt.ylabel("Residual ||grad||")
    plt.title("Residual per Iteration")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "residual_per_iteration.png"), dpi=180)
    plt.close()

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
            [f"{model_name} z*", f"{model_name} u(z*)"],
        ):
            plt.sca(ax)
            artist = dl.plot(fun)
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
    col_titles = ["optimal control z*", "state u(z*)"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title)
    for i, (model_name, control_model_fun, state_model_fun) in enumerate(overview_payload):
        fields = [control_model_fun, state_model_fun]
        for j, fun in enumerate(fields):
            ax = axes[i, j]
            plt.sca(ax)
            dl.plot(fun)
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
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax in axes.ravel():
        ax.axis("off")

    for ax, (model_name, state_model_fun) in zip(axes.ravel(), solution_payload):
        ax.axis("on")
        plt.sca(ax)
        artist = dl.plot(state_model_fun)
        ax.set_title(f"{model_name} u(z*)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "optimal_pde_solutions.png"), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Taylor models against mean-variance SAA truth on semilinear ADR control"
    )
    parser.add_argument("--beta", type=float, default=0.1, help="Variance weight beta")
    parser.add_argument("--n-tr", type=int, default=10, help="Number of Hessian modes for quadratic models")
    parser.add_argument("--n-mix", type=int, default=39, help="Number of mixture components")
    parser.add_argument("--truth-saa-samples", type=int, default=100000, help="Sample size for ground-truth SAA evaluation")
    parser.add_argument("--saa-seed", type=int, default=1, help="Seed for SAA sampling")
    parser.add_argument("--maxiter", type=int, default=60, help="Max iterations for each Taylor model")
    parser.add_argument("--maxiter-saa", type=int, default=60, help="Max iterations for optimized SAA models")
    parser.add_argument("--nx", type=int, default=16, help="Mesh cells in x")
    parser.add_argument("--ny", type=int, default=16, help="Mesh cells in y")
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
            print("  saa_1 / saa_10 / saa_39 / saa_100 / saa_200 / saa_500 / saa_1000 / saa_10000")
            print("  serial: linear / quadratic / saa_1 / saa_10 / saa_39")
            print("  parallel on all ranks: mixture_* / saa_100 / saa_200 / saa_500 / saa_1000 / saa_10000")
            print("=" * 90)
            print(f"Ground-truth SAA samples: {args.truth_saa_samples}")
            print(f"beta={args.beta}, n_tr={args.n_tr}, n_mix={args.n_mix}")
            print("control source term: f -> f + z")
            print("initial control: zero everywhere")
            print(f"terminal log file: {log_path}")
            print("=" * 90)
            sys.stdout.flush()

        problem = setup_problem(args, comm_mesh)
        Vh = problem["Vh"]
        control_model = problem["control_model"]
        prior = problem["prior"]

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

            res = optimize_with_tracking(
                model_name,
                approx_cost,
                args,
                rank,
                maxiter=args.maxiter_saa if model_name.startswith("saa_") else args.maxiter,
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
                    "control_source": "f + z",
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
