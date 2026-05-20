"""Compare static approximation errors on Navier-Stokes with z = 0."""

from __future__ import annotations
import numpy as np
import scipy.optimize
import soupy
import argparse
import atexit
import gc
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List

_SOUPY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.environ.get("HIPPYLIB_PATH", ""))
sys.path.append(_SOUPY_ROOT)

from mpi4py import MPI

from navier_stokes_compare_utils import setup_problem
from navier_stokes_driver_common import (
    TeeStream,
    compute_ground_truth_stats,
    compute_mc_trial_statistics,
    make_zero_control,
    parse_int_list,
    plot_decay,
    print_result_line,
    relative_error,
    save_rows_csv,
    summarize_mc_single_trial,
    summarize_mc_trials,
)
from soupy.approximations.taylor import (
    TaylorMixtureLinearCVaRControlCostFunctional,
    TaylorMixtureLinearControlCostFunctional,
    TaylorMixtureQuadraticCVaRControlCostFunctional,
    TaylorMixtureQuadraticControlCostFunctional,
)

MODEL_ORDER = ["mc_single", "mc", "mixture_linear_kle", "mixture_linear_hep", "mixture_quadratic_kle", "mixture_quadratic_hep"]
MIXTURE_MODEL_ORDER = ["mixture_linear_kle", "mixture_linear_hep", "mixture_quadratic_kle", "mixture_quadratic_hep"]
MODEL_LABELS = {
    "mc_single": "MC (single trial)",
    "mc": "MC (200 trial)",
    "mixture_linear_kle": "Mixture linear KLE",
    "mixture_linear_hep": "Mixture linear HEP",
    "mixture_quadratic_kle": "Mixture quadratic KLE",
    "mixture_quadratic_hep": "Mixture quadratic HEP",
}
MODEL_COLORS = {
    "mc_single": "dimgray",
    "mc": "black",
    "mixture_linear_kle": "tab:green",
    "mixture_linear_hep": "tab:olive",
    "mixture_quadratic_kle": "tab:red",
    "mixture_quadratic_hep": "tab:brown",
}
MODEL_MARKERS = {"mc_single": "x", "mc": "o", "mixture_linear_kle": "s", "mixture_linear_hep": "^", "mixture_quadratic_kle": "D", "mixture_quadratic_hep": "v"}


def make_meanvar_mixture_cost(model_name: str, control_model, prior, args, beta: float):
    if model_name == "mixture_linear_kle":
        return TaylorMixtureLinearControlCostFunctional(control_model, prior, None, {"beta": beta, "N_mix": args.current_n_mix, "direction": "kle", "verbose": args.verbose})
    if model_name == "mixture_linear_hep":
        return TaylorMixtureLinearControlCostFunctional(control_model, prior, None, {"beta": beta, "N_mix": args.current_n_mix, "direction": "hep", "verbose": args.verbose})
    if model_name == "mixture_quadratic_kle":
        return TaylorMixtureQuadraticControlCostFunctional(control_model, prior, None, {"beta": beta, "N_mix": args.current_n_mix, "direction": "kle", "N_tr": args.n_tr, "N_mc": 0, "verbose": args.verbose})
    if model_name == "mixture_quadratic_hep":
        return TaylorMixtureQuadraticControlCostFunctional(control_model, prior, None, {"beta": beta, "N_mix": args.current_n_mix, "direction": "hep", "N_tr": args.n_tr, "N_mc": 0, "verbose": args.verbose})
    raise ValueError(f"Unknown mean-variance model: {model_name}")


def make_cvar_mixture_cost(model_name: str, control_model, prior, args):
    if model_name == "mixture_linear_kle":
        return TaylorMixtureLinearCVaRControlCostFunctional(control_model, prior, None, {"beta": args.cvar_beta, "N_mix": args.current_n_mix, "direction": "kle", "verbose": args.verbose})
    if model_name == "mixture_linear_hep":
        return TaylorMixtureLinearCVaRControlCostFunctional(control_model, prior, None, {"beta": args.cvar_beta, "N_mix": args.current_n_mix, "direction": "hep", "verbose": args.verbose})
    if model_name == "mixture_quadratic_kle":
        return TaylorMixtureQuadraticCVaRControlCostFunctional(control_model, prior, None, {"beta": args.cvar_beta, "N_mix": args.current_n_mix, "direction": "kle", "N_tr": args.n_tr, "N_mc": args.quadratic_cvar_n_mc, "verbose": args.verbose})
    if model_name == "mixture_quadratic_hep":
        return TaylorMixtureQuadraticCVaRControlCostFunctional(control_model, prior, None, {"beta": args.cvar_beta, "N_mix": args.current_n_mix, "direction": "hep", "N_tr": args.n_tr, "N_mc": args.quadratic_cvar_n_mc, "verbose": args.verbose})
    raise ValueError(f"Unknown CVaR model: {model_name}")


def evaluate_linear_cvar(cost_functional, z0) -> float:
    return float(cost_functional.cost(z0, order=0))


def evaluate_quadratic_cvar(cost_functional, z0, beta: float) -> float:

    zt0 = cost_functional.generate_vector(soupy.CONTROL)
    zt0.get_vector().zero()
    zt0.get_vector().axpy(1.0, z0)
    zt0.get_vector().apply("")
    zt0.set_scalar(0.0)
    cost_functional.cost(zt0, order=0)

    component_samples = list(cost_functional.component_samples)
    component_weights = list(cost_functional.component_weights)
    smoothplus = cost_functional._legacy.smoothplus
    scale = 1.0 / (1.0 - beta)
    stacked = np.concatenate(component_samples)
    t_init = float(np.percentile(stacked, beta * 100.0))

    def objective_t(t_arr):
        t = float(np.atleast_1d(t_arr)[0])
        value = t
        for weight, samples_i in zip(component_weights, component_samples):
            value += weight * np.mean(smoothplus(samples_i - t)) * scale
        return float(value)

    t_opt = float(scipy.optimize.fmin(objective_t, np.array([t_init]), disp=False, xtol=1e-10, ftol=1e-10)[0])
    zt_opt = cost_functional.generate_vector(soupy.CONTROL)
    zt_opt.get_vector().zero()
    zt_opt.get_vector().axpy(1.0, z0)
    zt_opt.get_vector().apply("")
    zt_opt.set_scalar(t_opt)
    return float(cost_functional.cost(zt_opt, order=0))


def evaluate_mixture_model(model_name: str, control_model, prior, z0, args, gt_stats: Dict[str, float]) -> Dict[str, float]:
    start = time.time()
    mean_cost = make_meanvar_mixture_cost(model_name, control_model, prior, args, beta=0.0)
    mean_val = float(mean_cost.cost(z0, order=0))
    del mean_cost
    gc.collect()

    var_cost = make_meanvar_mixture_cost(model_name, control_model, prior, args, beta=1.0)
    mean_plus_var = float(var_cost.cost(z0, order=0))
    std_val = float(max(mean_plus_var - mean_val, 0.0) ** 0.5)
    del var_cost
    gc.collect()

    cvar_cost = make_cvar_mixture_cost(model_name, control_model, prior, args)
    cvar_val = evaluate_quadratic_cvar(cvar_cost, z0, args.cvar_beta) if "quadratic" in model_name else evaluate_linear_cvar(cvar_cost, z0)
    del cvar_cost
    gc.collect()

    elapsed = time.time() - start
    return {
        "model": model_name,
        "resolution_type": "n_mix",
        "resolution_value": int(args.current_n_mix),
        "mean": mean_val,
        "std": std_val,
        "cvar": cvar_val,
        "mean_rel_error": relative_error(mean_val, gt_stats["mean"]),
        "std_rel_error": relative_error(std_val, gt_stats["std"]),
        "cvar_rel_error": relative_error(cvar_val, gt_stats["cvar"]),
        "runtime_sec": float(elapsed),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare static approximation errors for Navier-Stokes at z = 0")
    parser.add_argument("--mixture-sizes", type=str, default="1, 3, 5, 7, 9, 15, 21, 27, 33, 39")
    parser.add_argument("--mc-samples", type=str, default="1, 2, 5, 10, 20, 50, 100, 200, 500, 1000")
    parser.add_argument("--mc-trials", type=int, default=5)
    parser.add_argument("--ground-truth-samples", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=1)
    parser.add_argument("--cvar-beta", type=float, default=0.95)
    parser.add_argument("--n-tr", type=int, default=50)
    parser.add_argument("--quadratic-cvar-n-mc", type=int, default=5000)
    parser.add_argument("--qoi-type", type=str, default="velocity_tracking", choices=["velocity_tracking"])
    parser.add_argument("--penalty", type=float, default=1.0)
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
    parser.add_argument("--save-dir", type=str, default="results_navier_stokes_static_model_error")
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.set_defaults(continuation=True, stabilization=True, nitche=True)
    args = parser.parse_args()

    args.mixture_sizes = parse_int_list(args.mixture_sizes)
    args.mc_samples = parse_int_list(args.mc_samples)

    rank = MPI.COMM_WORLD.Get_rank()
    comm_mesh = MPI.COMM_SELF
    os.makedirs(args.save_dir, exist_ok=True)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if rank == 0:
        log_path = os.path.join(args.save_dir, "terminal_output.txt")
        log_file = open(log_path, "w", buffering=1)
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)

        def _cleanup_log():
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                log_file.close()

        atexit.register(_cleanup_log)

    problem = setup_problem(args, comm_mesh)
    control_model = problem["control_model"]
    prior = problem["prior"]
    z0 = make_zero_control(control_model)

    gt_start = time.time()
    gt_stats = compute_ground_truth_stats(control_model, prior, z0, args.ground_truth_samples, args.sample_seed, args.cvar_beta)
    gt_elapsed = time.time() - gt_start

    rows: List[Dict[str, float]] = []
    for sample_size in args.mc_samples:
        mc_start = time.time()
        trial_stats = compute_mc_trial_statistics(control_model, prior, z0, int(sample_size), args.mc_trials, args.sample_seed, args.cvar_beta)
        mc_elapsed = time.time() - mc_start
        stats_rmse = summarize_mc_trials(trial_stats, gt_stats)
        row = {"model": "mc", "resolution_type": "mc_samples", "resolution_value": int(sample_size), "mean": stats_rmse["mean"], "std": stats_rmse["std"], "cvar": stats_rmse["cvar"], "mean_rel_error": stats_rmse["mean_rel_error"], "std_rel_error": stats_rmse["std_rel_error"], "cvar_rel_error": stats_rmse["cvar_rel_error"], "runtime_sec": float(mc_elapsed)}
        rows.append(row)
        if rank == 0:
            print_result_line(row, MODEL_LABELS)

        stats_single = summarize_mc_single_trial(trial_stats, gt_stats)
        row_single = {"model": "mc_single", "resolution_type": "mc_samples", "resolution_value": int(sample_size), "mean": stats_single["mean"], "std": stats_single["std"], "cvar": stats_single["cvar"], "mean_rel_error": stats_single["mean_rel_error"], "std_rel_error": stats_single["std_rel_error"], "cvar_rel_error": stats_single["cvar_rel_error"], "runtime_sec": float(mc_elapsed)}
        rows.append(row_single)
        if rank == 0:
            print_result_line(row_single, MODEL_LABELS)

    for n_mix in args.mixture_sizes:
        args.current_n_mix = n_mix
        for model_name in MIXTURE_MODEL_ORDER:
            row = evaluate_mixture_model(model_name, control_model, prior, z0, args, gt_stats)
            rows.append(row)
            if rank == 0:
                print_result_line(row, MODEL_LABELS)

    if rank == 0:
        save_rows_csv(os.path.join(args.save_dir, "navier_stokes_static_model_errors.csv"), rows)
        with open(os.path.join(args.save_dir, "navier_stokes_static_ground_truth.json"), "w") as f:
            json.dump({"timestamp": datetime.now().isoformat(), "config": {**vars(args), "current_n_mix": None}, "problem_settings": problem["settings"], "ground_truth": gt_stats, "ground_truth_runtime_sec": float(gt_elapsed)}, f, indent=2)
        plot_decay(rows, args.save_dir, MODEL_ORDER, MODEL_LABELS, MODEL_COLORS, MODEL_MARKERS, "navier_stokes_static_model_error_decay.png")


if __name__ == "__main__":
    main()
