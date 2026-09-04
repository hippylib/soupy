from __future__ import annotations

import csv
import gc
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

import dolfin as dl
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from mpi4py import MPI

import hippylib as hp
import soupy
from soupy import MeanVarRiskMeasureSAA, RiskMeasureControlCostFunctional, meanVarRiskMeasureSAASettings, sample_superquantile

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 16,
})

LBFGSB_OPTIONS = {
    "maxiter": None,
    "disp": False,
    "ftol": 1e-12,
    "gtol": 1e-4,
    "maxls": 20,
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
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 ** 2)
    except Exception:
        return _rss_from_proc_status_mb()


def get_peak_rss_mb() -> float:
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
        return int(model_name.split("_", 1)[1]) >= 11
    return False


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
            options={**LBFGSB_OPTIONS, "maxiter": maxiter},
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
        writer.writerow([
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
        ])
        for r in records:
            writer.writerow([
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
            ])


def save_summary_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
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
        ])
        for row in rows:
            writer.writerow(row)


def relative_error(model_value, true_value):
    return float(abs(model_value - true_value) / max(abs(true_value), 1e-14))


def plot_curves(results, model_order, model_colors, save_dir):
    # Objective and gradient-residual curves share the same per-model legend, so they are
    # drawn side by side in one figure with a single shared legend on the right.
    fig, (ax_obj, ax_res) = plt.subplots(1, 2, figsize=(15, 5))
    for model in model_order:
        rec = results[model]["iter_records"]
        if not rec:
            continue
        x = [r.iteration for r in rec]
        y_obj = [max(abs(r.model_cost), 1e-16) for r in rec]
        y_res = [max(r.residual, 1e-16) for r in rec]
        ax_obj.semilogy(x, y_obj, marker="o", linewidth=1.5, markersize=3, color=model_colors[model], label=model)
        ax_res.semilogy(x, y_res, marker="o", linewidth=1.5, markersize=3, color=model_colors[model], label=model)
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

    plt.figure(figsize=(10, 5))
    x = np.arange(len(model_order))
    time_values = [results[m]["avg_iter_time_sec"] for m in model_order]
    plt.bar(x, time_values, color=[model_colors[m] for m in model_order])
    plt.xticks(x, model_order, rotation=20, ha="right")
    plt.ylabel("Average Time per Iteration (s)")
    plt.title("Average Iteration Time Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "avg_iteration_time_comparison.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    iter_values = [results[m]["iter_count"] for m in model_order]
    plt.bar(x, iter_values, color=[model_colors[m] for m in model_order])
    plt.xticks(x, model_order, rotation=20, ha="right")
    plt.ylabel("Total Iteration Count")
    plt.title("Iteration Count Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "iteration_count_comparison.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    y = [results[m]["opt_rel_err"] for m in model_order]
    plt.bar(x, y, color=[model_colors[m] for m in model_order])
    plt.yscale("log")
    plt.xticks(x, model_order, rotation=15)
    plt.ylabel("|J_model(z*) - J_true(z*)| / |J_true(z*)|")
    plt.title("Final Objective Relative Error")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "final_objective_rel_error.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    y = [results[m]["init_rel_err"] for m in model_order]
    plt.bar(x, y, color=[model_colors[m] for m in model_order])
    plt.yscale("log")
    plt.xticks(x, model_order, rotation=15)
    plt.ylabel("|J_model(z0) - J_true(z0)| / |J_true(z0)|")
    plt.title("Initial Objective Relative Error")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "initial_objective_rel_error.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    y = [results[m]["true_opt"] for m in model_order]
    plt.bar(x, y, color=[model_colors[m] for m in model_order])
    plt.xticks(x, model_order, rotation=15)
    plt.ylabel("J_true(z*)")
    plt.title("Final True Objective Value")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "final_true_objective_value.png"), dpi=180)
    plt.close()


def parse_int_list(text: str) -> List[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError(f"Could not parse any integers from: {text}")
    return values


def compute_stats_from_samples(samples: np.ndarray, beta: float) -> Dict[str, float]:
    return {
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples)),
        "cvar": float(sample_superquantile(samples, beta)),
    }


def make_zero_control(control_model):
    z = control_model.generate_vector(soupy.CONTROL)
    z.zero()
    z.apply("")
    return z


def sample_qoi(control_model, prior, z0, sample_size: int, seed: int) -> np.ndarray:
    noise = dl.Vector(control_model.problem.Vh[soupy.STATE].mesh().mpi_comm())
    prior.init_vector(noise, "noise")
    rng = hp.Random(seed=seed)

    q_samples = np.zeros(sample_size, dtype=float)
    u = control_model.generate_vector(soupy.STATE)
    p = control_model.generate_vector(soupy.ADJOINT)

    for i in range(sample_size):
        m = control_model.generate_vector(soupy.PARAMETER)
        rng.normal(1.0, noise)
        prior.sample(noise, m)
        x = [u, m, p, z0]
        control_model.solveFwd(u, x)
        q_samples[i] = float(control_model.cost(x))

    return q_samples


def compute_ground_truth_stats(control_model, prior, z0, gt_sample_size: int, seed: int, beta: float):
    q_samples = sample_qoi(control_model, prior, z0, gt_sample_size, seed)
    gt_stats = compute_stats_from_samples(q_samples, beta)
    del q_samples
    gc.collect()
    return gt_stats


def _mc_trial_seed(base_seed: int, sample_size: int, trial_idx: int) -> int:
    return int(base_seed + 1000003 * int(sample_size) + 9176 * int(trial_idx + 1))


def _relative_rmse(estimates: np.ndarray, truth: float) -> float:
    truth = float(truth)
    return float(np.sqrt(np.mean((estimates - truth) ** 2)) / max(abs(truth), 1e-14))


def compute_mc_trial_statistics(control_model, prior, z0, sample_size: int, n_trials: int, base_seed: int, beta: float) -> Dict[str, np.ndarray]:
    trial_means = np.zeros(n_trials, dtype=float)
    trial_stds = np.zeros(n_trials, dtype=float)
    trial_cvars = np.zeros(n_trials, dtype=float)

    for trial_idx in range(n_trials):
        q_samples = sample_qoi(control_model, prior, z0, sample_size=sample_size, seed=_mc_trial_seed(base_seed, sample_size, trial_idx))
        stats = compute_stats_from_samples(q_samples, beta)
        trial_means[trial_idx] = stats["mean"]
        trial_stds[trial_idx] = stats["std"]
        trial_cvars[trial_idx] = stats["cvar"]
        del q_samples

    gc.collect()
    return {"mean": trial_means, "std": trial_stds, "cvar": trial_cvars}


def summarize_mc_trials(trial_stats: Dict[str, np.ndarray], gt_stats: Dict[str, float]) -> Dict[str, float]:
    trial_means = np.asarray(trial_stats["mean"], dtype=float)
    trial_stds = np.asarray(trial_stats["std"], dtype=float)
    trial_cvars = np.asarray(trial_stats["cvar"], dtype=float)
    return {
        "mean": float(np.mean(trial_means)),
        "std": float(np.mean(trial_stds)),
        "cvar": float(np.mean(trial_cvars)),
        "mean_rel_error": _relative_rmse(trial_means, gt_stats["mean"]),
        "std_rel_error": _relative_rmse(trial_stds, gt_stats["std"]),
        "cvar_rel_error": _relative_rmse(trial_cvars, gt_stats["cvar"]),
    }


def summarize_mc_single_trial(trial_stats: Dict[str, np.ndarray], gt_stats: Dict[str, float]) -> Dict[str, float]:
    mean_val = float(trial_stats["mean"][0])
    std_val = float(trial_stats["std"][0])
    cvar_val = float(trial_stats["cvar"][0])
    return {
        "mean": mean_val,
        "std": std_val,
        "cvar": cvar_val,
        "mean_rel_error": relative_error(mean_val, gt_stats["mean"]),
        "std_rel_error": relative_error(std_val, gt_stats["std"]),
        "cvar_rel_error": relative_error(cvar_val, gt_stats["cvar"]),
    }


def save_rows_csv(path: str, rows: List[Dict[str, float]]):
    fieldnames = [
        "model",
        "resolution_type",
        "resolution_value",
        "mean",
        "std",
        "cvar",
        "mean_rel_error",
        "std_rel_error",
        "cvar_rel_error",
        "runtime_sec",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_rows_csv(path: str) -> List[Dict[str, float]]:
    """Inverse of save_rows_csv, used to replot from previously saved results."""
    int_fields = {"resolution_value"}
    float_fields = {"mean", "std", "cvar", "mean_rel_error", "std_rel_error", "cvar_rel_error", "runtime_sec"}
    rows: List[Dict[str, float]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = dict(raw_row)
            for key in int_fields:
                row[key] = int(row[key])
            for key in float_fields:
                row[key] = float(row[key])
            rows.append(row)
    return rows


def plot_decay(rows: Sequence[Dict[str, float]], save_dir: str, model_order, model_labels, model_colors, model_markers, out_name):
    quantities = [
        ("mean_rel_error", "Mean Relative Error"),
        ("std_rel_error", "Std Relative Error"),
        ("cvar_rel_error", "CVaR(0.95) Relative Error"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharex=False, sharey=False)
    for ax, (key, title) in zip(axes, quantities):
        for model_name in model_order:
            model_rows = [r for r in rows if r["model"] == model_name]
            model_rows.sort(key=lambda r: r["resolution_value"])
            x = [r["resolution_value"] for r in model_rows]
            y = [max(r[key], 1e-16) for r in model_rows]
            ax.loglog(
                x,
                y,
                marker=model_markers[model_name],
                color=model_colors[model_name],
                linewidth=1.8,
                markersize=6,
                label=model_labels[model_name],
            )
        ax.set_xlabel(r"$N_{\mathrm{mix}}$ / MC samples")
        ax.set_ylabel("Relative error")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, out_name), dpi=180, bbox_inches="tight")
    plt.close(fig)


def print_result_line(row: Dict[str, float], model_labels):
    print(
        f"[{model_labels[row['model']]:22s}] "
        f"{row['resolution_type']}={int(row['resolution_value']):6d} | "
        f"mean={row['mean']:.6e} (rel={row['mean_rel_error']:.3e}) | "
        f"std={row['std']:.6e} (rel={row['std_rel_error']:.3e}) | "
        f"cvar={row['cvar']:.6e} (rel={row['cvar_rel_error']:.3e}) | "
        f"time={row['runtime_sec']:.2f}s"
    )
