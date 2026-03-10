"""Compare Taylor and mixture-Taylor models on hyperelasticity control."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
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
]
MODEL_COLORS = {
    "linear": "tab:blue",
    "quadratic": "tab:orange",
    "mixture_linear_kle": "tab:green",
    "mixture_linear_hep": "tab:olive",
    "mixture_quadratic_kle": "tab:red",
    "mixture_quadratic_hep": "tab:brown",
    "saa": "tab:purple",
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
    true_cost: float
    abs_error: float
    rel_error: float
    residual: float
    iter_time_sec: float


class ScipyObjectiveWithHistory:
    def __init__(self, cost_functional):
        self.cost_functional = cost_functional
        self._z = cost_functional.generate_vector(soupy.CONTROL)
        self._g = cost_functional.generate_vector(soupy.CONTROL)
        self.latest_cost = np.nan
        self.latest_grad_norm = np.nan
        self.n_func = 0
        self.n_grad = 0

    def function(self):
        def f(z_np):
            self._z.set_local(z_np)
            self._z.apply("")
            self.latest_cost = float(self.cost_functional.cost(self._z, order=0))
            self.n_func += 1
            return self.latest_cost

        return f

    def jac(self):
        def g(z_np):
            self._z.set_local(z_np)
            self._z.apply("")
            self.cost_functional.cost(self._z, order=1)
            self.latest_grad_norm = float(self.cost_functional.grad(self._g))
            self.n_grad += 1
            return self._g.get_local()

        return g


def setup_problem(args, comm_mesh):
    settings = hyperelasticity_problem_settings()
    settings["qoi_type"] = args.qoi_type
    mesh, Vh, _, control_model, prior = setup_hyperelasticity_problem(settings, comm_mesh)
    penalty = soupy.L2Penalization(Vh, args.penalty)
    return mesh, Vh, control_model, prior, penalty


def make_saa_cost(control_model, prior, penalty, beta, sample_size, seed):
    settings = meanVarRiskMeasureSAASettings()
    settings["beta"] = beta
    settings["sample_size"] = sample_size
    settings["seed"] = seed
    risk = MeanVarRiskMeasureSAA(control_model, prior, settings=settings, comm_sampler=MPI.COMM_SELF)
    return RiskMeasureControlCostFunctional(risk, penalty)


def make_taylor_cost(model_name, control_model, prior, penalty, args):
    if model_name == "linear":
        return TaylorLinearControlCostFunctional(
            control_model,
            prior,
            penalty,
            {"beta": args.beta, "correction": False, "N_mc": 0, "verbose": args.verbose},
        )
    if model_name == "quadratic":
        return TaylorQuadraticControlCostFunctional(
            control_model,
            prior,
            penalty,
            {"beta": args.beta, "N_tr": args.n_tr, "correction": False, "N_mc": 0, "verbose": args.verbose},
        )
    if model_name == "mixture_linear_kle":
        return TaylorMixtureLinearControlCostFunctional(
            control_model,
            prior,
            penalty,
            {"beta": args.beta, "N_mix": args.n_mix, "direction": "kle", "verbose": args.verbose},
        )
    if model_name == "mixture_linear_hep":
        return TaylorMixtureLinearControlCostFunctional(
            control_model,
            prior,
            penalty,
            {"beta": args.beta, "N_mix": args.n_mix, "direction": "hep", "verbose": args.verbose},
        )
    if model_name == "mixture_quadratic_kle":
        return TaylorMixtureQuadraticControlCostFunctional(
            control_model,
            prior,
            penalty,
            {
                "beta": args.beta,
                "N_mix": args.n_mix,
                "direction": "kle",
                "N_tr": args.n_tr,
                "N_mc": 0,
                "verbose": args.verbose,
            },
        )
    if model_name == "mixture_quadratic_hep":
        return TaylorMixtureQuadraticControlCostFunctional(
            control_model,
            prior,
            penalty,
            {
                "beta": args.beta,
                "N_mix": args.n_mix,
                "direction": "hep",
                "N_tr": args.n_tr,
                "N_mc": 0,
                "verbose": args.verbose,
            },
        )
    raise ValueError(f"Unknown model name: {model_name}")


def np_to_control(cost_functional, z_np):
    z = cost_functional.generate_vector(soupy.CONTROL)
    z.set_local(z_np)
    z.apply("")
    return z


def optimize_with_tracking(model_name, approx_cost, true_cost, args, rank, maxiter, bounds):
    wrapper = ScipyObjectiveWithHistory(approx_cost)
    z0 = approx_cost.generate_vector(soupy.CONTROL)
    x0 = z0.get_local()
    iterate_history: List[np.ndarray] = []

    def callback(xk):
        iterate_history.append(np.array(xk, copy=True))

    if rank == 0:
        print(f"\nOptimizing {model_name} with L-BFGS-B ...")
        sys.stdout.flush()

    t0 = time.perf_counter()
    result = scipy.optimize.minimize(
        wrapper.function(),
        x0,
        method="L-BFGS-B",
        jac=wrapper.jac(),
        callback=callback,
        bounds=bounds,
        options={"maxiter": maxiter, "disp": True},
    )
    total_time = time.perf_counter() - t0
    iter_count = len(iterate_history) if iterate_history else int(result.nit)
    avg_iter_time_sec = total_time / max(iter_count, 1)

    iter_records: List[IterRecord] = []
    for k, xk in enumerate(iterate_history, start=1):
        z_vec = np_to_control(approx_cost, xk)
        model_cost = float(approx_cost.cost(z_vec, order=1))
        g_vec = approx_cost.generate_vector(soupy.CONTROL)
        residual = float(approx_cost.grad(g_vec))
        true_cost_val = float(true_cost.cost(z_vec, order=0))
        abs_err = abs(model_cost - true_cost_val)
        rel_err = abs_err / max(abs(true_cost_val), 1e-14)
        iter_records.append(IterRecord(k, model_cost, true_cost_val, abs_err, rel_err, residual, np.nan))
        if rank == 0 and k % args.print_every == 0:
            print(
                f"  [{model_name:20s}] iter {k:4d}: "
                f"J_model={model_cost:.6e}, J_true={true_cost_val:.6e}, "
                f"rel_err={rel_err:.3e}, ||g||={residual:.3e}"
            )

    z_opt_np = np.array(result.x, copy=True)
    z_opt_vec = np_to_control(approx_cost, z_opt_np)
    approx_opt = float(approx_cost.cost(z_opt_vec, order=0))
    true_opt = float(true_cost.cost(z_opt_vec, order=0))
    opt_rel_err = abs(approx_opt - true_opt) / max(abs(true_opt), 1e-14)

    if rank == 0:
        print(
            f"  [{model_name:20s}] done: success={result.success}, nit={iter_count}, "
            f"avg_iter_time={avg_iter_time_sec:.2f}s, opt rel_err={opt_rel_err:.3e}"
        )

    return {
        "result": result,
        "iter_records": iter_records,
        "iter_count": iter_count,
        "avg_iter_time_sec": avg_iter_time_sec,
        "z_opt_np": z_opt_np,
        "approx_opt": approx_opt,
        "true_opt": true_opt,
        "opt_rel_err": opt_rel_err,
    }


def rrmse(z, z_ref):
    return float(np.sqrt(np.mean((z - z_ref) ** 2)) / max(np.sqrt(np.mean(z_ref ** 2)), 1e-14))


def save_iteration_csv(path, records):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "model_cost", "true_cost", "abs_error", "rel_error", "residual", "iter_time_sec"])
        for r in records:
            writer.writerow([r.iteration, r.model_cost, r.true_cost, r.abs_error, r.rel_error, r.residual, r.iter_time_sec])


def save_summary_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "success", "nit", "nfev", "njev", "avg_iter_time_sec", "opt_model_objective", "opt_true_objective", "opt_rel_error", "rrmse_vs_saa_opt"])
        writer.writerows(rows)


def plot_curves(results, saa_ref, save_dir):
    plt.figure(figsize=(8, 5))
    for model in MODEL_ORDER:
        rec = results[model]["iter_records"]
        if not rec:
            continue
        x = [r.iteration for r in rec]
        y = [max(r.rel_error, 1e-16) for r in rec]
        plt.semilogy(x, y, marker="o", linewidth=1.5, markersize=3, color=MODEL_COLORS[model], label=model)
    plt.xlabel("Iteration")
    plt.ylabel("Relative Approximation Error")
    plt.title("Objective Relative Approximation Error per Iteration")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "approx_error_per_iteration.png"), dpi=180)
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

    names = ["saa"] + MODEL_ORDER
    tvals = [saa_ref["avg_iter_time_sec"]] + [results[m]["avg_iter_time_sec"] for m in MODEL_ORDER]
    plt.figure(figsize=(10, 5))
    x = np.arange(len(names))
    plt.bar(x, tvals, color=[MODEL_COLORS[m] for m in names])
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylabel("Average Time per Iteration (s)")
    plt.title("Average Iteration Time Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "avg_iteration_time_comparison.png"), dpi=180)
    plt.close()

    iters = [saa_ref["iter_count"]] + [results[m]["iter_count"] for m in MODEL_ORDER]
    plt.figure(figsize=(10, 5))
    x = np.arange(len(names))
    plt.bar(x, iters, color=[MODEL_COLORS[m] for m in names])
    plt.xticks(x, names, rotation=20, ha="right")
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
    plt.title("Optimal-Control Objective Relative Error")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "optimal_objective_rel_error.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    y = [results[m]["rrmse_vs_saa_opt"] for m in MODEL_ORDER]
    plt.bar(x, y, color=[MODEL_COLORS[m] for m in MODEL_ORDER])
    plt.yscale("log")
    plt.xticks(x, MODEL_ORDER, rotation=15)
    plt.ylabel("RRMSE vs SAA z*")
    plt.title("Optimal Control Relative RMSE")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "optimal_control_rrmse.png"), dpi=180)
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


def save_optimal_field_plots(results, z_saa_opt, control_model, prior, Vh, save_dir):
    control_saa_vec, state_saa_vec = solve_state_at_control(control_model, prior, z_saa_opt)

    V_control = Vh[soupy.CONTROL]
    V_state = Vh[soupy.STATE]
    V_state_scalar = dl.FunctionSpace(V_state.mesh(), "CG", 1)

    control_saa_fun = vector_to_function(V_control, control_saa_vec)
    state_saa_fun = scalarize_for_plot(vector_to_function(V_state, state_saa_vec), V_state_scalar)
    overview_payload = []

    for model_name in MODEL_ORDER:
        z_model_opt = results[model_name]["z_opt_np"]
        control_model_vec, state_model_vec = solve_state_at_control(control_model, prior, z_model_opt)

        control_model_fun = vector_to_function(V_control, control_model_vec)
        control_abs_err_fun = vector_to_function(V_control, control_model_vec.copy())
        control_abs_err_fun.vector().axpy(-1.0, control_saa_vec)
        control_abs_err_fun.vector().set_local(np.abs(control_abs_err_fun.vector().get_local()))
        control_abs_err_fun.vector().apply("")

        state_model_fun = scalarize_for_plot(vector_to_function(V_state, state_model_vec), V_state_scalar)
        state_abs_err_fun = dl.Function(V_state_scalar)
        state_abs_err_fun.vector().zero()
        state_abs_err_fun.vector().axpy(1.0, state_model_fun.vector())
        state_abs_err_fun.vector().axpy(-1.0, state_saa_fun.vector())
        state_abs_err_fun.vector().set_local(np.abs(state_abs_err_fun.vector().get_local()))
        state_abs_err_fun.vector().apply("")

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, fun, title in zip(
            axes,
            [control_model_fun, control_saa_fun, control_abs_err_fun],
            [f"{model_name} z*", "SAA z*", "|z* - z*_SAA|"],
        ):
            artist = dl.plot(fun, ax=ax)
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            plt.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{model_name}_optimal_control_vs_saa.png"), dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, fun, title in zip(
            axes,
            [state_model_fun, state_saa_fun, state_abs_err_fun],
            [f"{model_name} u(z*)", "SAA u(z*_SAA)", "|u(z*) - u(z*_SAA)|"],
        ):
            artist = dl.plot(fun, ax=ax)
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            plt.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{model_name}_optimal_solution_vs_saa.png"), dpi=180)
        plt.close(fig)
        overview_payload.append((model_name, control_model_fun, control_abs_err_fun, state_model_fun, state_abs_err_fun))

    n_models = len(overview_payload)
    fig, axes = plt.subplots(n_models, 6, figsize=(24, 3.6 * n_models))
    if n_models == 1:
        axes = np.array([axes])
    col_titles = ["model z*", "SAA z*", "|z* - z*_SAA|", "model u(z*)", "SAA u(z*_SAA)", "|u(z*) - u(z*_SAA)|"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title)
    for i, (model_name, control_model_fun, control_abs_err_fun, state_model_fun, state_abs_err_fun) in enumerate(overview_payload):
        fields = [control_model_fun, control_saa_fun, control_abs_err_fun, state_model_fun, state_saa_fun, state_abs_err_fun]
        for j, fun in enumerate(fields):
            ax = axes[i, j]
            dl.plot(fun, ax=ax)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[i, 0].set_ylabel(model_name, rotation=90, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "optimal_fields_overview.png"), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare Taylor models against SAA truth on hyperelasticity")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--n-tr", type=int, default=50)
    parser.add_argument("--n-mix", type=int, default=11)
    parser.add_argument("--saa-samples", type=int, default=1000)
    parser.add_argument("--saa-seed", type=int, default=1)
    parser.add_argument("--qoi-type", type=str, default="stiffness", choices=["all", "stiffness", "point"])
    parser.add_argument("--penalty", type=float, default=1e-2)
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument("--maxiter-saa", type=int, default=60)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--save-dir", type=str, default="results_compare_taylor_models")
    parser.add_argument("--log-file", type=str, default="terminal_output.txt")
    parser.add_argument("--bound-lb", type=float, default=0.0)
    parser.add_argument("--bound-ub", type=float, default=1.0)
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
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
            print("  mixture_quadratic_kle / mixture_quadratic_hep")
            print("=" * 78)
            print(f"SAA truth samples: {args.saa_samples}")
            print(f"beta={args.beta}, n_tr={args.n_tr}, n_mix={args.n_mix}")
            print(f"terminal log file: {log_path}")
            print("=" * 78)

        _, _, control_model, prior, penalty = setup_problem(args, comm_mesh)
        bounds = scipy.optimize.Bounds(lb=args.bound_lb, ub=args.bound_ub)

        true_cost = make_saa_cost(control_model, prior, penalty, args.beta, args.saa_samples, args.saa_seed)

        if rank == 0:
            print("\nComputing SAA reference optimum with L-BFGS-B ...")
        saa_ref = optimize_with_tracking("saa", true_cost, true_cost, args, rank, maxiter=args.maxiter_saa, bounds=bounds)
        z_saa_opt = saa_ref["z_opt_np"]

        results: Dict[str, Dict] = {}
        for model_name in MODEL_ORDER:
            approx_cost = make_taylor_cost(model_name, control_model, prior, penalty, args)
            res = optimize_with_tracking(model_name, approx_cost, true_cost, args, rank, maxiter=args.maxiter, bounds=bounds)
            res["rrmse_vs_saa_opt"] = rrmse(res["z_opt_np"], z_saa_opt)
            results[model_name] = res

        if rank == 0:
            for model_name in MODEL_ORDER:
                save_iteration_csv(os.path.join(args.save_dir, f"{model_name}_iteration_metrics.csv"), results[model_name]["iter_records"])
            save_iteration_csv(os.path.join(args.save_dir, "saa_iteration_metrics.csv"), saa_ref["iter_records"])

            summary_rows = []
            for model_name in MODEL_ORDER:
                rr = results[model_name]
                summary_rows.append([
                    model_name,
                    rr["result"].success,
                    rr["iter_count"],
                    rr["result"].nfev,
                    rr["result"].njev,
                    rr["avg_iter_time_sec"],
                    rr["approx_opt"],
                    rr["true_opt"],
                    rr["opt_rel_err"],
                    rr["rrmse_vs_saa_opt"],
                ])
            save_summary_csv(os.path.join(args.save_dir, "summary.csv"), summary_rows)

            payload = {
                "config": vars(args),
                "saa_reference": {
                    "success": bool(saa_ref["result"].success),
                    "nit": int(saa_ref["iter_count"]),
                    "nfev": int(saa_ref["result"].nfev),
                    "njev": int(saa_ref["result"].njev),
                    "iter_count": int(saa_ref["iter_count"]),
                    "avg_iter_time_sec": float(saa_ref["avg_iter_time_sec"]),
                    "objective_at_opt": float(saa_ref["true_opt"]),
                },
                "models": {
                    name: {
                        "success": bool(results[name]["result"].success),
                        "nit": int(results[name]["iter_count"]),
                        "nfev": int(results[name]["result"].nfev),
                        "njev": int(results[name]["result"].njev),
                        "iter_count": int(results[name]["iter_count"]),
                        "avg_iter_time_sec": float(results[name]["avg_iter_time_sec"]),
                        "opt_model_objective": float(results[name]["approx_opt"]),
                        "opt_true_objective": float(results[name]["true_opt"]),
                        "opt_rel_error": float(results[name]["opt_rel_err"]),
                        "rrmse_vs_saa_opt": float(results[name]["rrmse_vs_saa_opt"]),
                    }
                    for name in MODEL_ORDER
                },
            }
            with open(os.path.join(args.save_dir, "summary.json"), "w") as f:
                json.dump(payload, f, indent=2)

            plot_curves(results, saa_ref, args.save_dir)
            save_optimal_field_plots(results, z_saa_opt, control_model, prior, Vh, args.save_dir)

            print("\n" + "-" * 78)
            print("Summary (vs SAA reference optimum)")
            print("-" * 78)
            print(f"{'saa':20s} | nit={saa_ref['iter_count']:3d} | avg_iter_time={saa_ref['avg_iter_time_sec']:8.2f}s")
            for model_name in MODEL_ORDER:
                rr = results[model_name]
                print(
                    f"{model_name:20s} | nit={rr['iter_count']:3d} | "
                    f"avg_iter_time={rr['avg_iter_time_sec']:8.2f}s | "
                    f"opt rel_err={rr['opt_rel_err']:.3e} | "
                    f"RRMSE(z*)={rr['rrmse_vs_saa_opt']:.3e}"
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
