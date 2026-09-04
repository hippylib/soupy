"""Compare static approximation errors on semilinear ADR with a fixed target control.

This mirrors the workflow/style of
`examples/poisson/driver_poisson_compare_models_fixed_controls.py`, but for
the semilinear ADR problem at a fixed Gaussian-well control. We keep the
full coefficient-based control space and evaluate every model at the fixed
control vector z with zᵢ = sin(2π·xᵢ) * sin(2π·yᵢ) at each well center.

Compared models:
- MC
- mixture_linear_kle
- mixture_linear_hep
- mixture_quadratic_kle
- mixture_quadratic_hep

Metrics:
- QoI mean relative error
- QoI standard deviation relative error
- QoI CVaR(0.95) relative error

Ground truth:
- MC with 1e5 samples by default

Notes:
- Mean is obtained from the mean-variance model with beta = 0 and no penalty.
- Standard deviation is obtained from
      sqrt( J_{beta=1} - J_{beta=0} )
  since J_beta = E[Q] + beta Var[Q] here.
- CVaR uses the mixture CVaR classes as requested.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import gc
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Iterable, List, Sequence

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

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 16,
})

import hippylib as hp
import soupy
from semilinear_adr_problem import (
    ControlParameters,
    MeshParameters,
    PDEParameters,
    PriorParameters,
    ControlledSemilinearADRWellVarfHandler,
    control_well_centers,
    setup_control_function_space,
    setup_mesh,
    setup_prior,
    setup_qoi,
)
from soupy import (
    ControlModel,
    PDEVariationalControlProblem,
    VariationalControlQoI,
    sample_superquantile,
)
from soupy.approximations.taylor import (
    TaylorMixtureLinearCVaRControlCostFunctional,
    TaylorMixtureLinearControlCostFunctional,
    TaylorMixtureQuadraticCVaRControlCostFunctional,
    TaylorMixtureQuadraticControlCostFunctional,
)


dl.set_log_active(False)
try:
    configure_dolfin_form_compiler(dl)
except Exception:
    pass


MODEL_ORDER = [
    "mc_single",
    "mc",
    "mixture_linear_kle",
    "mixture_linear_hep",
    "mixture_quadratic_kle",
    "mixture_quadratic_hep",
]

MIXTURE_MODEL_ORDER = [
    "mixture_linear_kle",
    "mixture_linear_hep",
    "mixture_quadratic_kle",
    "mixture_quadratic_hep",
]

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

MODEL_MARKERS = {
    "mc_single": "x",
    "mc": "o",
    "mixture_linear_kle": "s",
    "mixture_linear_hep": "^",
    "mixture_quadratic_kle": "D",
    "mixture_quadratic_hep": "v",
}


class TeeStream:
    """Duplicate writes to terminal stream and a log file stream."""

    def __init__(self, terminal_stream, file_stream):
        self._terminal = terminal_stream
        self._file = file_stream

    def write(self, data):
        self._terminal.write(data)
        self._file.write(data)
        self._terminal.flush()
        self._file.flush()

    def flush(self):
        self._terminal.flush()
        self._file.flush()


class StaticSemilinearADRControlVarfHandler:
    """Wrap the ADR residual so it is compatible with a dummy control variable."""

    def __init__(self, base_varf_handler):
        self.base_varf_handler = base_varf_handler

    def __call__(self, u, m, p, z):
        del z
        return self.base_varf_handler(u, m, p)


class StaticSemilinearADRQoIFormHandler:
    """Wrap the ADR QoI form so it is compatible with a dummy control variable."""

    def __init__(self, qoi_varf):
        self.qoi_varf = qoi_varf

    def __call__(self, u, m, z):
        del z
        return self.qoi_varf(u, m)


def parse_int_list(text: str) -> List[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError(f"Could not parse any integers from: {text}")
    return values


def relative_error(approx: float, truth: float) -> float:
    return abs(float(approx) - float(truth)) / max(abs(float(truth)), 1e-14)


def compute_stats_from_samples(samples: np.ndarray, beta: float) -> Dict[str, float]:
    return {
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples)),
        "cvar": float(sample_superquantile(samples, beta)),
    }


def make_zero_control(control_model: ControlModel) -> dl.Vector:
    z = control_model.generate_vector(soupy.CONTROL)
    z.zero()
    z.apply("")
    return z


def make_target_control(control_model: ControlModel, control_parameters: ControlParameters) -> dl.Vector:
    """Fixed control with zᵢ = sin(2π·xᵢ) * sin(2π·yᵢ) at each well center."""
    centers = control_well_centers(control_parameters)
    coeffs = np.sin(2.0 * np.pi * centers[:, 0]) * np.sin(2.0 * np.pi * centers[:, 1])
    z = control_model.generate_vector(soupy.CONTROL)
    z.set_local(coeffs)
    z.apply("")
    return z


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
    qoi = VariationalControlQoI(
        Vh,
        StaticSemilinearADRQoIFormHandler(base_qoi.qoi_varf),
    )
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


def sample_qoi(control_model, prior, z0, sample_size: int, seed: int) -> np.ndarray:
    """Sample only scalar QoI values to keep memory usage modest."""
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
    """Compute ground-truth statistics from a single large MC sample set."""
    q_samples = sample_qoi(control_model, prior, z0, gt_sample_size, seed)
    gt_stats = compute_stats_from_samples(q_samples, beta)
    del q_samples
    gc.collect()
    return gt_stats


def _mc_trial_seed(base_seed: int, sample_size: int, trial_idx: int) -> int:
    """Deterministic per-trial seed used for independent MC resampling."""
    return int(base_seed + 1000003 * int(sample_size) + 9176 * int(trial_idx + 1))


def _relative_rmse(estimates: np.ndarray, truth: float) -> float:
    truth = float(truth)
    return float(np.sqrt(np.mean((estimates - truth) ** 2)) / max(abs(truth), 1e-14))


def compute_mc_trial_statistics(
    control_model,
    prior,
    z0,
    sample_size: int,
    n_trials: int,
    base_seed: int,
    beta: float,
) -> Dict[str, float]:
    """Collect per-trial MC statistics used for RMSE and single-trial reporting."""
    trial_means = np.zeros(n_trials, dtype=float)
    trial_stds = np.zeros(n_trials, dtype=float)
    trial_cvars = np.zeros(n_trials, dtype=float)

    for trial_idx in range(n_trials):
        q_samples = sample_qoi(
            control_model,
            prior,
            z0,
            sample_size=sample_size,
            seed=_mc_trial_seed(base_seed, sample_size, trial_idx),
        )
        stats = compute_stats_from_samples(q_samples, beta)
        trial_means[trial_idx] = stats["mean"]
        trial_stds[trial_idx] = stats["std"]
        trial_cvars[trial_idx] = stats["cvar"]
        del q_samples

    gc.collect()
    return {
        "mean": trial_means,
        "std": trial_stds,
        "cvar": trial_cvars,
    }


def summarize_mc_trials(trial_stats: Dict[str, np.ndarray], gt_stats: Dict[str, float]) -> Dict[str, float]:
    """Summarize MC trials using relative RMSE against ground truth."""
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
    """Report the first MC trial and its relative error trajectory."""
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


def make_meanvar_mixture_cost(model_name: str, control_model, prior, args, beta: float):
    if model_name == "mixture_linear_kle":
        settings = {"beta": beta, "N_mix": args.current_n_mix, "direction": "kle", "verbose": args.verbose}
        return TaylorMixtureLinearControlCostFunctional(control_model, prior, None, settings)

    if model_name == "mixture_linear_hep":
        settings = {"beta": beta, "N_mix": args.current_n_mix, "direction": "hep", "verbose": args.verbose}
        return TaylorMixtureLinearControlCostFunctional(control_model, prior, None, settings)

    if model_name == "mixture_quadratic_kle":
        settings = {
            "beta": beta,
            "N_mix": args.current_n_mix,
            "direction": "kle",
            "N_tr": args.n_tr,
            "N_mc": 0,
            "verbose": args.verbose,
        }
        return TaylorMixtureQuadraticControlCostFunctional(control_model, prior, None, settings)

    if model_name == "mixture_quadratic_hep":
        settings = {
            "beta": beta,
            "N_mix": args.current_n_mix,
            "direction": "hep",
            "N_tr": args.n_tr,
            "N_mc": 0,
            "verbose": args.verbose,
        }
        return TaylorMixtureQuadraticControlCostFunctional(control_model, prior, None, settings)

    raise ValueError(f"Unknown mean-variance model: {model_name}")


def make_cvar_mixture_cost(model_name: str, control_model, prior, args):
    if model_name == "mixture_linear_kle":
        settings = {
            "beta": args.cvar_beta,
            "N_mix": args.current_n_mix,
            "direction": "kle",
            "verbose": args.verbose,
        }
        return TaylorMixtureLinearCVaRControlCostFunctional(control_model, prior, None, settings)

    if model_name == "mixture_linear_hep":
        settings = {
            "beta": args.cvar_beta,
            "N_mix": args.current_n_mix,
            "direction": "hep",
            "verbose": args.verbose,
        }
        return TaylorMixtureLinearCVaRControlCostFunctional(control_model, prior, None, settings)

    if model_name == "mixture_quadratic_kle":
        settings = {
            "beta": args.cvar_beta,
            "N_mix": args.current_n_mix,
            "direction": "kle",
            "N_tr": args.n_tr,
            "N_mc": args.quadratic_cvar_n_mc,
            "verbose": args.verbose,
        }
        return TaylorMixtureQuadraticCVaRControlCostFunctional(control_model, prior, None, settings)

    if model_name == "mixture_quadratic_hep":
        settings = {
            "beta": args.cvar_beta,
            "N_mix": args.current_n_mix,
            "direction": "hep",
            "N_tr": args.n_tr,
            "N_mc": args.quadratic_cvar_n_mc,
            "verbose": args.verbose,
        }
        return TaylorMixtureQuadraticCVaRControlCostFunctional(control_model, prior, None, settings)

    raise ValueError(f"Unknown CVaR model: {model_name}")


def evaluate_linear_cvar(cost_functional, z) -> float:
    return float(cost_functional.cost(z, order=0))


def evaluate_quadratic_cvar(cost_functional, beta: float, z) -> float:
    zt0 = cost_functional.generate_vector(soupy.CONTROL)
    zt0.get_vector().zero()
    zt0.get_vector().axpy(1.0, z)
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

    t_opt = float(
        scipy.optimize.fmin(objective_t, np.array([t_init]), disp=False, xtol=1e-10, ftol=1e-10)[0]
    )

    zt_opt = cost_functional.generate_vector(soupy.CONTROL)
    zt_opt.get_vector().zero()
    zt_opt.get_vector().axpy(1.0, z)
    zt_opt.get_vector().apply("")
    zt_opt.set_scalar(t_opt)
    return float(cost_functional.cost(zt_opt, order=0))


def evaluate_mixture_model(model_name: str, control_model, prior, args, gt_stats: Dict[str, float], z0) -> Dict[str, float]:
    start = time.time()

    mean_cost = make_meanvar_mixture_cost(model_name, control_model, prior, args, beta=0.0)
    mean_val = float(mean_cost.cost(z0, order=0))

    var_cost = make_meanvar_mixture_cost(model_name, control_model, prior, args, beta=1.0)
    mean_plus_var = float(var_cost.cost(z0, order=0))
    std_val = float(np.sqrt(max(mean_plus_var - mean_val, 0.0)))

    cvar_cost = make_cvar_mixture_cost(model_name, control_model, prior, args)
    if "quadratic" in model_name:
        cvar_val = evaluate_quadratic_cvar(cvar_cost, args.cvar_beta, z0)
    else:
        cvar_val = evaluate_linear_cvar(cvar_cost, z0)

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


def plot_decay(rows: Sequence[Dict[str, float]], save_dir: str):
    quantities = [
        ("mean_rel_error", "Mean Relative Error"),
        ("std_rel_error", "Std Relative Error"),
        ("cvar_rel_error", "CVaR(0.95) Relative Error"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharex=False, sharey=False)

    for ax, (key, title) in zip(axes, quantities):
        for model_name in MODEL_ORDER:
            model_rows = [r for r in rows if r["model"] == model_name]
            model_rows.sort(key=lambda r: r["resolution_value"])
            x = [r["resolution_value"] for r in model_rows]
            y = [max(r[key], 1e-16) for r in model_rows]
            ax.loglog(
                x,
                y,
                marker=MODEL_MARKERS[model_name],
                color=MODEL_COLORS[model_name],
                linewidth=1.8,
                markersize=6,
                label=MODEL_LABELS[model_name],
            )

        ax.set_xlabel(r"$N_{\mathrm{mix}}$ / MC samples")
        ax.set_ylabel("Relative error")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "semilinear_adr_static_model_error_decay.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def print_result_line(row: Dict[str, float]):
    print(
        f"[{MODEL_LABELS[row['model']]:22s}] "
        f"{row['resolution_type']}={int(row['resolution_value']):6d} | "
        f"mean={row['mean']:.6e} (rel={row['mean_rel_error']:.3e}) | "
        f"std={row['std']:.6e} (rel={row['std_rel_error']:.3e}) | "
        f"cvar={row['cvar']:.6e} (rel={row['cvar_rel_error']:.3e}) | "
        f"time={row['runtime_sec']:.2f}s"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare static approximation errors for semilinear ADR at target control z"
    )
    parser.add_argument("--mixture-sizes", type=str, default="1, 3, 5, 7, 9, 15, 21, 27, 33, 39")
    parser.add_argument("--mc-samples", type=str, default="1, 2, 5, 10, 20, 50, 100, 200, 500, 1000")
    parser.add_argument("--mc-trials", type=int, default=20) # Here DC used 1000
    parser.add_argument("--ground-truth-samples", type=int, default=100000) # Here DC used 200000
    parser.add_argument("--sample-seed", type=int, default=1)
    parser.add_argument("--cvar-beta", type=float, default=0.95)
    parser.add_argument("--n-tr", type=int, default=50)
    parser.add_argument("--quadratic-cvar-n-mc", type=int, default=10000) # Here DC used 10^5
    parser.add_argument("--nx", type=int, default=64, help="Optional mesh override for debugging") # DC used 128
    parser.add_argument("--ny", type=int, default=64, help="Optional mesh override for debugging") # DC used 128
    parser.add_argument("--newton-max-it", type=int, default=50)
    parser.add_argument("--newton-rtol", type=float, default=1e-8)
    parser.add_argument("--newton-atol", type=float, default=1e-10)
    parser.add_argument("--save-dir", type=str, default="results_semilinear_adr_static_model_error")
    parser.add_argument("--replot", action="store_true", help="Skip all computation and regenerate the plot from the CSV already saved in --save-dir")
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    args = parser.parse_args()

    args.mixture_sizes = parse_int_list(args.mixture_sizes)
    args.mc_samples = parse_int_list(args.mc_samples)

    rank = MPI.COMM_WORLD.Get_rank()
    comm_mesh = MPI.COMM_SELF
    os.makedirs(args.save_dir, exist_ok=True)

    if args.replot:
        if rank == 0:
            rows = load_rows_csv(os.path.join(args.save_dir, "semilinear_adr_static_model_errors.csv"))
            plot_decay(rows, args.save_dir)
        return

    if rank == 0:
        log_path = os.path.join(args.save_dir, "terminal_output.txt")
        log_file = open(log_path, "w", buffering=1)
        sys.stdout = TeeStream(sys.stdout, log_file)
        sys.stderr = TeeStream(sys.stderr, log_file)

        def _cleanup_log():
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                log_file.close()

        atexit.register(_cleanup_log)

    problem = setup_problem(args, comm_mesh)
    control_model = problem["control_model"]
    prior = problem["prior"]
    z0 = make_target_control(control_model, problem["control_parameters"])

    if rank == 0:
        print("=" * 90)
        print("Semilinear ADR Static Approximation Error Comparison (mismatch QoI, z = target)")
        print("=" * 90)
        print(f"save_dir={args.save_dir}")
        print(
            f"mesh=({problem['mesh_parameters'].nx}, {problem['mesh_parameters'].ny}), "
            f"cvar_beta={args.cvar_beta}, n_tr={args.n_tr}, quadratic_cvar_n_mc={args.quadratic_cvar_n_mc}"
        )
        print(
            f"newton_max_it={args.newton_max_it}, "
            f"newton_rtol={args.newton_rtol:.1e}, newton_atol={args.newton_atol:.1e}, "
            "error_on_nonconvergence=True"
        )
        print(f"mixture_sizes={args.mixture_sizes}")
        print(f"mc_samples={args.mc_samples}, mc_trials={args.mc_trials}")
        print(f"ground_truth_samples={args.ground_truth_samples}, sample_seed={args.sample_seed}")
        print("=" * 90)

    gt_start = time.time()
    gt_stats = compute_ground_truth_stats(
        control_model,
        prior,
        z0,
        gt_sample_size=args.ground_truth_samples,
        seed=args.sample_seed,
        beta=args.cvar_beta,
    )
    gt_elapsed = time.time() - gt_start

    if rank == 0:
        print(
            "[Ground truth             ] "
            f"N={args.ground_truth_samples:6d} | "
            f"mean={gt_stats['mean']:.6e} | "
            f"std={gt_stats['std']:.6e} | "
            f"cvar={gt_stats['cvar']:.6e} | "
            f"time={gt_elapsed:.2f}s"
        )

    rows: List[Dict[str, float]] = []

    for sample_size in args.mc_samples:
        mc_start = time.time()
        trial_stats = compute_mc_trial_statistics(
            control_model,
            prior,
            z0,
            sample_size=int(sample_size),
            n_trials=args.mc_trials,
            base_seed=args.sample_seed,
            beta=args.cvar_beta,
        )
        mc_elapsed = time.time() - mc_start
        stats_rmse = summarize_mc_trials(trial_stats, gt_stats)
        row = {
            "model": "mc",
            "resolution_type": "mc_samples",
            "resolution_value": int(sample_size),
            "mean": stats_rmse["mean"],
            "std": stats_rmse["std"],
            "cvar": stats_rmse["cvar"],
            "mean_rel_error": stats_rmse["mean_rel_error"],
            "std_rel_error": stats_rmse["std_rel_error"],
            "cvar_rel_error": stats_rmse["cvar_rel_error"],
            "runtime_sec": float(mc_elapsed),
        }
        rows.append(row)
        if rank == 0:
            print_result_line(row)

        stats_single = summarize_mc_single_trial(trial_stats, gt_stats)
        row_single = {
            "model": "mc_single",
            "resolution_type": "mc_samples",
            "resolution_value": int(sample_size),
            "mean": stats_single["mean"],
            "std": stats_single["std"],
            "cvar": stats_single["cvar"],
            "mean_rel_error": stats_single["mean_rel_error"],
            "std_rel_error": stats_single["std_rel_error"],
            "cvar_rel_error": stats_single["cvar_rel_error"],
            "runtime_sec": float(mc_elapsed),
        }
        rows.append(row_single)
        if rank == 0:
            print_result_line(row_single)

    for n_mix in args.mixture_sizes:
        args.current_n_mix = n_mix
        for model_name in MIXTURE_MODEL_ORDER:
            row = evaluate_mixture_model(model_name, control_model, prior, args, gt_stats, z0)
            rows.append(row)
            if rank == 0:
                print_result_line(row)

    if rank == 0:
        save_rows_csv(os.path.join(args.save_dir, "semilinear_adr_static_model_errors.csv"), rows)
        with open(os.path.join(args.save_dir, "semilinear_adr_static_ground_truth.json"), "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "config": {
                        **vars(args),
                        "current_n_mix": None,
                    },
                    "ground_truth": gt_stats,
                    "ground_truth_runtime_sec": float(gt_elapsed),
                },
                f,
                indent=2,
            )
        plot_decay(rows, args.save_dir)

        print("\n" + "-" * 90)
        print("Minimum relative errors over tested resolutions")
        print("-" * 90)
        for model_name in MODEL_ORDER:
            model_rows = [r for r in rows if r["model"] == model_name]
            print(
                f"{MODEL_LABELS[model_name]:22s} "
                f"min mean rel={min(r['mean_rel_error'] for r in model_rows):.3e}, "
                f"min std rel={min(r['std_rel_error'] for r in model_rows):.3e}, "
                f"min cvar rel={min(r['cvar_rel_error'] for r in model_rows):.3e}"
            )
        print("-" * 90)
        print(f"Outputs written to: {args.save_dir}")


if __name__ == "__main__":
    main()
