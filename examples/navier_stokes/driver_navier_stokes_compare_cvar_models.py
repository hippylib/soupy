"""Compare Taylor and SAA CVaR models on Navier-Stokes control."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import scipy.optimize
from mpi4py import MPI

_SOUPY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.environ.get("HIPPYLIB_PATH", ""))
sys.path.append(_SOUPY_ROOT)

import soupy
from navier_stokes_compare_utils import save_optimal_field_plots, save_optimal_pde_solution_plot, setup_problem
from navier_stokes_driver_common import TeeStream, get_current_rss_mb, get_peak_rss_mb, relative_error
from soupy import RiskMeasureControlCostFunctional, SuperquantileRiskMeasureSAA, superquantileRiskMeasureSAASettings
from soupy.approximations.taylor import (
    TaylorLinearCVaRControlCostFunctional,
    TaylorMixtureLinearCVaRControlCostFunctional,
    TaylorMixtureQuadraticCVaRControlCostFunctional,
    TaylorQuadraticCVaRControlCostFunctional,
)

MODEL_ORDER = [
    "linear",
    "quadratic",
    "mixture_linear_kle",
    "mixture_linear_hep",
    "mixture_quadratic_kle",
    "mixture_quadratic_hep",
    "saa_1",
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
    "saa_1": "black",
    "saa_10": "#17becf",
    "saa_100": "#7f7f7f",
    "saa_200": "#9467bd",
    "saa_500": "#e377c2",
    "saa_1000": "#8c564b",
    "saa_10000": "#bcbd22",
}
EXPLICIT_CVAR_CONTINUATION_LEVELS = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]


@dataclass
class IterRecord:
    iteration: int
    model_cost: float
    residual: float
    epsilon: float
    iter_time_sec: float
    rss_mb: float
    peak_rss_mb: float
    cost_rss_before_mb: float
    cost_rss_after_mb: float
    grad_rss_before_mb: float
    grad_rss_after_mb: float


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

    def current_smoothing_epsilon(self):
        if hasattr(self.cost_functional, "risk_measure") and hasattr(self.cost_functional.risk_measure, "smoothplus") and hasattr(self.cost_functional.risk_measure.smoothplus, "epsilon"):
            return float(self.cost_functional.risk_measure.smoothplus.epsilon)
        if hasattr(self.cost_functional, "_legacy") and hasattr(self.cost_functional._legacy, "epsilon"):
            return float(self.cost_functional._legacy.epsilon)
        if hasattr(self.cost_functional, "epsilon"):
            return float(self.cost_functional.epsilon)
        return float("nan")

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


def is_augmented_vector(vector) -> bool:
    return hasattr(vector, "get_vector") and hasattr(vector, "get_scalar")


def set_control_part(vector, control_np, scalar=0.0):
    if is_augmented_vector(vector):
        vector.get_vector().set_local(control_np)
        vector.get_vector().apply("")
        vector.set_scalar(float(scalar))
    else:
        vector.set_local(control_np)
        vector.apply("")


def split_control(vector):
    if is_augmented_vector(vector):
        return np.array(vector.get_vector().get_local(), copy=True), float(vector.get_scalar())
    return np.array(vector.get_local(), copy=True), None


def np_to_control(cost_functional, z_np):
    z = cost_functional.generate_vector(soupy.CONTROL)
    z.set_local(z_np)
    z.apply("")
    return z


def exact_cvar_from_samples(samples, beta):
    quantile = float(np.percentile(samples, beta * 100.0))
    cvar = quantile + float(np.mean(np.maximum(samples - quantile, 0.0))) / (1.0 - beta)
    return quantile, cvar


def initial_model_point_at_control(cost_functional, control_np, scalar=0.0):
    point = cost_functional.generate_vector(soupy.CONTROL)
    set_control_part(point, control_np, scalar=scalar)
    value = float(cost_functional.cost(point, order=0))
    t_init = float(point.get_scalar()) if is_augmented_vector(point) else None
    return point.get_local(), value, t_init


def get_model_var_warm_start(cost_functional):
    var_value = getattr(cost_functional, "var", None)
    if var_value is None:
        return None
    return float(var_value)


def zero_control_np(control_model):
    z = control_model.generate_vector(soupy.CONTROL)
    z.zero()
    z.apply("")
    return np.array(z.get_local(), copy=True)


def make_cvar_saa_cost(control_model, prior, penalty, beta, sample_size, seed, comm_sampler, epsilon=1e-4):
    settings = superquantileRiskMeasureSAASettings()
    settings["beta"] = beta
    settings["sample_size"] = sample_size
    settings["seed"] = seed
    settings["epsilon"] = float(epsilon)
    risk = SuperquantileRiskMeasureSAA(control_model, prior, settings=settings, comm_sampler=comm_sampler)
    return RiskMeasureControlCostFunctional(risk, penalty)


def model_uses_world_parallel(model_name: str) -> bool:
    if model_name.startswith("mixture_"):
        return True
    if model_name.startswith("saa_"):
        return int(model_name.split("_", 1)[1]) >= 100
    return False


def is_saa_model(model_name: str) -> bool:
    return model_name.startswith("saa_")


def should_use_linear_warm_start(model_name: str) -> bool:
    return model_name != "linear" and not is_saa_model(model_name)


def is_explicit_continuation_model(model_name: str) -> bool:
    return model_name in {"quadratic", "mixture_quadratic_kle", "mixture_quadratic_hep"} or is_saa_model(model_name)


def make_cvar_cost(model_name, control_model, prior, penalty, args, epsilon_override=None):
    if is_saa_model(model_name):
        sample_size = int(model_name.split("_", 1)[1])
        return make_cvar_saa_cost(control_model, prior, penalty, args.cvar_beta, sample_size, args.saa_seed, MPI.COMM_WORLD if sample_size >= 100 else MPI.COMM_SELF, epsilon=1e-4 if epsilon_override is None else float(epsilon_override))
    if model_name == "linear":
        return TaylorLinearCVaRControlCostFunctional(control_model, prior, penalty, {"beta": args.cvar_beta, "verbose": args.verbose})
    if model_name == "quadratic":
        return TaylorQuadraticCVaRControlCostFunctional(control_model, prior, penalty, {"beta": args.cvar_beta, "N_tr": args.n_tr, "N_mc": args.quadratic_cvar_n_mc, "epsilon": 1e-4 if epsilon_override is None else float(epsilon_override), "verbose": args.verbose})
    if model_name == "mixture_linear_kle":
        return TaylorMixtureLinearCVaRControlCostFunctional(control_model, prior, penalty, {"beta": args.cvar_beta, "N_mix": args.n_mix, "direction": "kle", "verbose": args.verbose}, comm_sampler=MPI.COMM_WORLD)
    if model_name == "mixture_linear_hep":
        return TaylorMixtureLinearCVaRControlCostFunctional(control_model, prior, penalty, {"beta": args.cvar_beta, "N_mix": args.n_mix, "direction": "hep", "verbose": args.verbose}, comm_sampler=MPI.COMM_WORLD)
    if model_name == "mixture_quadratic_kle":
        return TaylorMixtureQuadraticCVaRControlCostFunctional(control_model, prior, penalty, {"beta": args.cvar_beta, "N_mix": args.n_mix, "direction": "kle", "N_tr": args.n_tr, "N_mc": args.quadratic_cvar_n_mc, "epsilon": 1e-4 if epsilon_override is None else float(epsilon_override), "verbose": args.verbose}, comm_sampler=MPI.COMM_WORLD)
    if model_name == "mixture_quadratic_hep":
        return TaylorMixtureQuadraticCVaRControlCostFunctional(control_model, prior, penalty, {"beta": args.cvar_beta, "N_mix": args.n_mix, "direction": "hep", "N_tr": args.n_tr, "N_mc": args.quadratic_cvar_n_mc, "epsilon": 1e-4 if epsilon_override is None else float(epsilon_override), "verbose": args.verbose}, comm_sampler=MPI.COMM_WORLD)
    raise ValueError(f"Unknown model name: {model_name}")


def evaluate_true_cvar_stats(cost_functional, control_np, beta):
    point = cost_functional.generate_vector(soupy.CONTROL)
    set_control_part(point, control_np, scalar=0.0)
    cost_functional.cost(point, order=0)
    risk = cost_functional.risk_measure
    samples = risk.gather_samples()
    t_opt, cvar_qoi = exact_cvar_from_samples(samples, beta)
    penalty = 0.0 if cost_functional.penalization is None else float(cost_functional.penalization.cost(point))
    total_cost = float(cvar_qoi + penalty)
    return total_cost, float(np.mean(samples)), float(np.var(samples)), float(cvar_qoi), float(t_opt)


def optimize_with_tracking(model_name, approx_cost, x0, bounds, args, rank, maxiter, root_only=True):
    payload = None
    if (not root_only) or rank == 0:
        wrapper = ScipyObjectiveWithHistory(approx_cost)
        iter_records: List[IterRecord] = []
        callback_last_time = None

        def callback(_xk):
            nonlocal callback_last_time
            now = time.perf_counter()
            iter_time = np.nan if callback_last_time is None else now - callback_last_time
            callback_last_time = now
            iter_records.append(IterRecord(len(iter_records) + 1, float(wrapper.latest_cost), float(wrapper.latest_grad_norm), float(wrapper.current_smoothing_epsilon()), iter_time, get_current_rss_mb(), get_peak_rss_mb(), float(wrapper.latest_cost_rss_before_mb), float(wrapper.latest_cost_rss_after_mb), float(wrapper.latest_grad_rss_before_mb), float(wrapper.latest_grad_rss_after_mb)))

        t0 = time.perf_counter()
        result = scipy.optimize.minimize(wrapper.function(), x0, method="L-BFGS-B", jac=wrapper.jac(), callback=callback, bounds=bounds, options={"maxiter": maxiter, "disp": False, "ftol": 1e-20, "gtol": 1e-4, "maxls": 50})
        total_time = time.perf_counter() - t0
        iter_count = len(iter_records) if iter_records else int(result.nit)
        avg_iter_time_sec = total_time / max(iter_count, 1)
        z_opt_vec = np_to_control(approx_cost, result.x)
        control_opt_np, t_opt = split_control(z_opt_vec)
        approx_opt = float(approx_cost.cost(z_opt_vec, order=0))
        payload = {"success": bool(result.success), "status": int(result.status), "message": str(result.message), "nfev": int(result.nfev), "njev": int(result.njev), "iter_records": iter_records, "iter_count": iter_count, "avg_iter_time_sec": avg_iter_time_sec, "total_time_sec": total_time, "x0_np": np.array(x0, copy=True), "z_opt_np": np.array(result.x, copy=True), "control_opt_np": control_opt_np, "t_opt": t_opt, "approx_opt": approx_opt}
    if root_only:
        payload = MPI.COMM_WORLD.bcast(payload if rank == 0 else None, root=0)
    return payload


def optimize_with_explicit_continuation(model_name, control_model, prior, penalty, x0_np, args, rank, maxiter):
    initial_x0_np = np.array(x0_np, copy=True)
    current_x0_np = np.array(x0_np, copy=True)
    total_iter_count = 0
    total_time_sec = 0.0
    total_nfev = 0
    total_njev = 0
    combined_iter_records: List[IterRecord] = []
    stage_payloads = []
    final_payload = None
    root_only = not model_uses_world_parallel(model_name)

    for stage_idx, epsilon in enumerate(EXPLICIT_CVAR_CONTINUATION_LEVELS, start=1):
        approx_cost = make_cvar_cost(model_name, control_model, prior, penalty, args, epsilon_override=epsilon)
        inner_bounds = make_bounds(current_x0_np, args)
        inner_payload = optimize_with_tracking(f"{model_name}[eps={epsilon:.0e}]", approx_cost, current_x0_np, inner_bounds, args, rank, maxiter=maxiter, root_only=root_only)
        del approx_cost
        for record in inner_payload["iter_records"]:
            combined_iter_records.append(IterRecord(record.iteration + total_iter_count, record.model_cost, record.residual, record.epsilon, record.iter_time_sec, record.rss_mb, record.peak_rss_mb, record.cost_rss_before_mb, record.cost_rss_after_mb, record.grad_rss_before_mb, record.grad_rss_after_mb))
        total_iter_count += int(inner_payload["iter_count"])
        total_time_sec += float(inner_payload["total_time_sec"])
        total_nfev += int(inner_payload["nfev"])
        total_njev += int(inner_payload["njev"])
        current_x0_np = np.array(inner_payload["z_opt_np"], copy=True)
        stage_payloads.append({"stage": stage_idx, "epsilon": float(epsilon), "success": bool(inner_payload["success"]), "status": int(inner_payload["status"]), "message": str(inner_payload["message"]), "iter_count": int(inner_payload["iter_count"]), "approx_opt": float(inner_payload["approx_opt"])})
        final_payload = inner_payload

    final_payload = dict(final_payload)
    final_payload["success"] = all(payload["success"] for payload in stage_payloads)
    final_payload["x0_np"] = np.array(initial_x0_np, copy=True)
    final_payload["z_opt_np"] = np.array(current_x0_np, copy=True)
    final_payload["iter_records"] = combined_iter_records
    final_payload["iter_count"] = total_iter_count
    final_payload["avg_iter_time_sec"] = total_time_sec / max(total_iter_count, 1)
    final_payload["total_time_sec"] = total_time_sec
    final_payload["nfev"] = total_nfev
    final_payload["njev"] = total_njev
    final_payload["continuation_stages"] = stage_payloads
    return final_payload


def make_bounds(x0, args):
    lb = np.full_like(x0, float(args.bound_lb), dtype=float)
    ub = np.full_like(x0, float(args.bound_ub), dtype=float)
    if x0.shape[0] > args.control_dim:
        lb[-1] = -np.inf
        ub[-1] = np.inf
    return scipy.optimize.Bounds(lb=lb, ub=ub)


def save_iteration_csv(path, records: List[IterRecord]):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "model_cost", "residual", "epsilon", "iter_time_sec", "rss_mb", "peak_rss_mb", "cost_rss_before_mb", "cost_rss_after_mb", "grad_rss_before_mb", "grad_rss_after_mb"])
        for r in records:
            writer.writerow([r.iteration, r.model_cost, r.residual, r.epsilon, r.iter_time_sec, r.rss_mb, r.peak_rss_mb, r.cost_rss_before_mb, r.cost_rss_after_mb, r.grad_rss_before_mb, r.grad_rss_after_mb])


def save_summary_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "success", "message", "nit", "nfev", "njev", "avg_iter_time_sec", "initial_model_objective", "initial_true_objective", "initial_rel_error", "opt_model_objective", "opt_true_objective", "opt_rel_error", "opt_model_t", "opt_true_var", "opt_t_var_rel_error"])
        for row in rows:
            writer.writerow(row)


def optional_relative_error(model_value, true_value):
    if model_value is None or true_value is None:
        return None
    return relative_error(float(model_value), float(true_value))


def plot_curves(results, save_dir):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    for model in MODEL_ORDER:
        rec = results[model]["iter_records"]
        if not rec:
            continue
        plt.semilogy([r.iteration for r in rec], [max(abs(r.model_cost), 1e-16) for r in rec], marker="o", linewidth=1.5, markersize=3, color=MODEL_COLORS[model], label=model)
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(save_dir, "objective_per_iteration.png"), dpi=180); plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare Taylor models against CVaR SAA truth on Navier-Stokes")
    parser.add_argument("--cvar-beta", "--beta", dest="cvar_beta", type=float, default=0.95)
    parser.add_argument("--n-tr", type=int, default=50)
    parser.add_argument("--n-mix", type=int, default=11)
    parser.add_argument("--quadratic-cvar-n-mc", type=int, default=5000)
    parser.add_argument("--truth-saa-samples", "--saa-samples", dest="truth_saa_samples", type=int, default=100000)
    parser.add_argument("--saa-seed", type=int, default=1)
    parser.add_argument("--qoi-type", type=str, default="velocity_tracking", choices=["velocity_tracking"])
    parser.add_argument("--penalty", type=float, default=1.0)
    parser.add_argument("--maxiter", type=int, default=240)
    parser.add_argument("--maxiter-saa", type=int, default=240)
    parser.add_argument("--mesh-base-directory", type=str, default="./")
    parser.add_argument("--mesh-resolution", type=str, default="medium")
    parser.add_argument("--mesh-format", type=str, default="xdmf")
    parser.add_argument("--nu", type=float, default=5e-3)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=5.0)
    parser.add_argument("--mean-velocity", type=float, default=1.0)
    parser.add_argument("--continuation", action="store_true", help="Use PDE viscosity continuation")
    parser.add_argument("--no-continuation", dest="continuation", action="store_false")
    parser.add_argument("--stabilization", action="store_true")
    parser.add_argument("--no-stabilization", dest="stabilization", action="store_false")
    parser.add_argument("--nitche", action="store_true")
    parser.add_argument("--no-nitche", dest="nitche", action="store_false")
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--save-dir", type=str, default="results_compare_cvar_models")
    parser.add_argument("--log-file", type=str, default="terminal_output.txt")
    parser.add_argument("--bound-lb", type=float, default=-2.0)
    parser.add_argument("--bound-ub", type=float, default=2.0)
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.set_defaults(continuation=True, stabilization=True, nitche=True)
    args = parser.parse_args()

    rank = MPI.COMM_WORLD.Get_rank()
    comm_mesh = MPI.COMM_SELF
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = args.log_file if os.path.isabs(args.log_file) else os.path.join(args.save_dir, args.log_file)

    log_file = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if rank == 0:
        log_file = open(log_path, "w", buffering=1)
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)

    try:
        problem = setup_problem(args, comm_mesh)
        Vh = problem["Vh"]
        control_model = problem["control_model"]
        prior = problem["prior"]
        penalty = problem["penalty"]
        control0_np = zero_control_np(control_model)
        base_control_np = control0_np.copy()
        base_t = 0.0
        linear_base_t = 0.0
        args.control_dim = len(control0_np)
        linear_init_control_np = control0_np.copy()
        linear_init_t = 0.0

        results: Dict[str, Dict] = {}
        for model_name in MODEL_ORDER:
            approx_cost = None
            x0_np = None
            approx_init = None
            init_t = None
            use_linear_warm_start = should_use_linear_warm_start(model_name)
            run_parallel_model = model_uses_world_parallel(model_name)
            epsilon_override = EXPLICIT_CVAR_CONTINUATION_LEVELS[0] if is_explicit_continuation_model(model_name) else None

            if run_parallel_model:
                approx_cost = make_cvar_cost(model_name, control_model, prior, penalty, args, epsilon_override=epsilon_override)
                init_control_np = linear_init_control_np if use_linear_warm_start else base_control_np
                init_control_np = MPI.COMM_WORLD.bcast(init_control_np if rank == 0 else None, root=0)
                init_scalar = linear_init_t if use_linear_warm_start else (linear_base_t if model_name == "linear" else base_t)
                x0_np, approx_init, init_t = initial_model_point_at_control(approx_cost, init_control_np, scalar=init_scalar)
            elif rank == 0:
                approx_cost = make_cvar_cost(model_name, control_model, prior, penalty, args, epsilon_override=epsilon_override)
                init_control_np = linear_init_control_np if use_linear_warm_start else base_control_np
                init_scalar = linear_init_t if use_linear_warm_start else (linear_base_t if model_name == "linear" else base_t)
                x0_np, approx_init, init_t = initial_model_point_at_control(approx_cost, init_control_np, scalar=init_scalar)
            else:
                init_control_np = None

            x0_np = MPI.COMM_WORLD.bcast(x0_np if rank == 0 else None, root=0)
            approx_init = MPI.COMM_WORLD.bcast(approx_init if rank == 0 else None, root=0)
            init_t = MPI.COMM_WORLD.bcast(init_t if rank == 0 else None, root=0)
            init_control_np = MPI.COMM_WORLD.bcast(init_control_np if rank == 0 else None, root=0)

            truth_cost = make_cvar_saa_cost(control_model, prior, penalty, args.cvar_beta, args.truth_saa_samples, args.saa_seed, MPI.COMM_WORLD)
            true_init, true_init_mean, true_init_var, true_init_cvar, _ = evaluate_true_cvar_stats(truth_cost, init_control_np, args.cvar_beta)
            del truth_cost
            init_rel_err = relative_error(approx_init, true_init)

            bounds = make_bounds(x0_np, args) if (run_parallel_model or rank == 0) else None
            if is_explicit_continuation_model(model_name):
                res = optimize_with_explicit_continuation(model_name, control_model, prior, penalty, x0_np, args, rank, maxiter=args.maxiter_saa if is_saa_model(model_name) else args.maxiter)
            else:
                res = optimize_with_tracking(model_name, approx_cost, x0_np, bounds, args, rank, maxiter=args.maxiter_saa if is_saa_model(model_name) else args.maxiter, root_only=not run_parallel_model)

            res["control_init_np"] = np.array(init_control_np, copy=True)
            res["t_init"] = None if init_t is None else float(init_t)
            res["approx_init"] = float(approx_init)
            res["true_init"] = float(true_init)
            res["true_init_mean"] = float(true_init_mean)
            res["true_init_var"] = float(true_init_var)
            res["true_init_cvar"] = float(true_init_cvar)
            res["init_rel_err"] = float(init_rel_err)

            truth_cost = make_cvar_saa_cost(control_model, prior, penalty, args.cvar_beta, args.truth_saa_samples, args.saa_seed, MPI.COMM_WORLD)
            true_opt, true_opt_mean, true_opt_var, true_opt_cvar, t_true_opt = evaluate_true_cvar_stats(truth_cost, res["control_opt_np"], args.cvar_beta)
            del truth_cost
            res["true_opt"] = float(true_opt)
            res["true_opt_mean"] = float(true_opt_mean)
            res["true_opt_var"] = float(true_opt_var)
            res["true_opt_cvar"] = float(true_opt_cvar)
            res["t_true_opt"] = float(t_true_opt)
            res["opt_rel_err"] = relative_error(res["approx_opt"], res["true_opt"])
            res["opt_t_var_rel_err"] = optional_relative_error(res["t_opt"], res["t_true_opt"])
            results[model_name] = res

            if model_name == "linear":
                linear_init_control_np = np.array(res["control_opt_np"], copy=True)
                linear_var = get_model_var_warm_start(approx_cost)
                linear_init_t = 0.0 if linear_var is None else float(linear_var)
            if approx_cost is not None:
                del approx_cost

        if rank == 0:
            for model_name in MODEL_ORDER:
                save_iteration_csv(os.path.join(args.save_dir, f"{model_name}_iteration_metrics.csv"), results[model_name]["iter_records"])
            summary_rows = []
            for model_name in MODEL_ORDER:
                rr = results[model_name]
                summary_rows.append([model_name, rr["success"], rr["message"], rr["iter_count"], rr["nfev"], rr["njev"], rr["avg_iter_time_sec"], rr["approx_init"], rr["true_init"], rr["init_rel_err"], rr["approx_opt"], rr["true_opt"], rr["opt_rel_err"], rr["t_opt"], rr["t_true_opt"], rr["opt_t_var_rel_err"]])
            save_summary_csv(os.path.join(args.save_dir, "summary.csv"), summary_rows)
            with open(os.path.join(args.save_dir, "summary.json"), "w") as f:
                json.dump({"config": vars(args), "problem_settings": problem["settings"]}, f, indent=2)
            plot_curves(results, args.save_dir)
            save_optimal_field_plots(results, MODEL_ORDER, control_model, prior, Vh, args.save_dir)
            save_optimal_pde_solution_plot(results, MODEL_ORDER, control_model, prior, Vh, args.save_dir)
    finally:
        if rank == 0 and log_file is not None:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()


if __name__ == "__main__":
    main()
