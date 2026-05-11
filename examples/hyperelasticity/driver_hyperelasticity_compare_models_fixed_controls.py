"""Compare model approximation errors on fixed controls for hyperelasticity."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from typing import Dict, List, Tuple

# Configure macOS compiler BEFORE importing dolfin or soupy
_SOUPY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, os.path.join(_SOUPY_ROOT, "soupy", "utils"))
try:
    from macos_config import configure_macos_compiler, configure_dolfin_form_compiler

    configure_macos_compiler()
except ImportError:
    pass
sys.path.pop(0)

sys.path.append(os.environ.get("HIPPYLIB_PATH", ""))
sys.path.append(_SOUPY_ROOT)

import dolfin as dl
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

import soupy
from driver_hyperelasticity_compare_taylor_models import MODEL_ORDER, make_saa_cost, make_taylor_cost, setup_problem


logging.getLogger("FFC").setLevel(logging.WARNING)
logging.getLogger("UFL").setLevel(logging.WARNING)
dl.set_log_active(False)
try:
    configure_dolfin_form_compiler(dl)
except Exception:
    pass


MODEL_COLORS = {
    "linear": "tab:blue",
    "quadratic": "tab:orange",
    "mixture_linear_kle": "tab:green",
    "mixture_linear_hep": "tab:olive",
    "mixture_quadratic_kle": "tab:red",
    "mixture_quadratic_hep": "tab:brown",
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


def build_test_controls(Vh_control, seed: int) -> List[Tuple[str, dl.Vector]]:
    controls: List[Tuple[str, dl.Vector]] = []
    template = dl.Function(Vh_control).vector()

    def _const(name: str, value: float):
        z = template.copy()
        z.set_local(np.full(z.local_size(), value))
        z.apply("")
        controls.append((name, z))

    _const("const_0", 0.0)
    _const("const_05", 0.5)
    _const("const_1", 1.0)

    coords = Vh_control.tabulate_dof_coordinates().reshape((-1, Vh_control.mesh().geometry().dim()))
    x = coords[:, 0]
    xmin, xmax = float(np.min(x)), float(np.max(x))
    xhat = (x - xmin) / max(xmax - xmin, 1e-14)

    z_aff_small = template.copy()
    z_aff_small.set_local(np.clip(0.5 + 0.25 * (xhat - 0.5), 0.0, 1.0))
    z_aff_small.apply("")
    controls.append(("affine_small", z_aff_small))

    z_aff_large = template.copy()
    z_aff_large.set_local(np.clip(0.5 + 0.6 * (xhat - 0.5), 0.0, 1.0))
    z_aff_large.apply("")
    controls.append(("affine_large", z_aff_large))

    rng = np.random.default_rng(seed)
    z_rand_s = template.copy()
    z_rand_s.set_local(np.clip(0.5 + 0.1 * rng.standard_normal(z_rand_s.local_size()), 0.0, 1.0))
    z_rand_s.apply("")
    controls.append(("random_small", z_rand_s))

    z_rand_l = template.copy()
    z_rand_l.set_local(np.clip(0.5 + 0.35 * rng.standard_normal(z_rand_l.local_size()), 0.0, 1.0))
    z_rand_l.apply("")
    controls.append(("random_large", z_rand_l))
    return controls


def evaluate_cost(cost_functional, z_vec: dl.Vector) -> float:
    return float(cost_functional.cost(z_vec, order=0))


def evaluate_mean_variance(cost_functional, z_vec: dl.Vector):
    """
    Evaluate mean and variance terms for mean-variance risk objectives.
    Returns (mean, variance) if available, otherwise (nan, nan).
    """
    try:
        cost_functional.cost(z_vec, order=0)
        risk = cost_functional.risk_measure
        mean = float(risk.q_bar)
        variance = float(risk.q2_bar - risk.q_bar ** 2)
        return mean, variance
    except Exception:
        return float("nan"), float("nan")


def vec_l2_norm(z_vec: dl.Vector) -> float:
    return float(np.sqrt(max(z_vec.inner(z_vec), 0.0)))


def vec_linf_norm(z_vec: dl.Vector) -> float:
    arr = z_vec.get_local()
    return float(np.max(np.abs(arr))) if arr.size else 0.0


def save_rows_csv(path: str, rows: List[Dict]):
    fieldnames = [
        "control_name",
        "model",
        "model_cost",
        "true_cost",
        "abs_error",
        "rel_error",
        "z_l2_norm",
        "z_linf_norm",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_model_summary_csv(path: str, summary_rows: List[Dict]):
    fieldnames = ["model", "mean_abs_error", "max_abs_error", "mean_rel_error", "max_rel_error"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def plot_abs_error(rows: List[Dict], save_dir: str):
    controls = sorted({r["control_name"] for r in rows})
    x = np.arange(len(controls))
    width = 0.12

    plt.figure(figsize=(11, 5))
    for i, model in enumerate(MODEL_ORDER):
        vals = []
        for cname in controls:
            matched = [r for r in rows if r["control_name"] == cname and r["model"] == model]
            vals.append(matched[0]["abs_error"] if matched else np.nan)
        plt.bar(x + (i - (len(MODEL_ORDER)-1)/2) * width, vals, width=width, color=MODEL_COLORS[model], label=model)

    plt.yscale("log")
    plt.xticks(x, controls, rotation=20, ha="right")
    plt.ylabel("Absolute Approximation Error")
    plt.title("Hyperelasticity: Model Approximation Error at Fixed Controls")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fixed_control_abs_error_comparison.png"), dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare approximation error at fixed controls (no optimization)")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--n-tr", type=int, default=50)
    parser.add_argument("--n-mix", type=int, default=11)
    parser.add_argument("--saa-samples", type=int, default=1000)
    parser.add_argument("--saa-seed", type=int, default=1)
    parser.add_argument("--qoi-type", type=str, default="virtual_work",
                        choices=["all", "stiffness", "point", "virtual_work"])
    parser.add_argument("--penalty", type=float, default=1e-2)
    parser.add_argument("--z-seed", type=int, default=7)
    parser.add_argument("--save-dir", type=str, default="results_fixed_z_model_error")
    parser.add_argument(
        "--log-file",
        type=str,
        default="terminal_output.txt",
        help="Log file name (or path) for terminal outputs",
    )
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
            print("Hyperelasticity Fixed-Control Approximation Error Comparison")
            print("=" * 78)
            print(f"save_dir={args.save_dir}")
            print(f"log_file={log_path}")
            print(f"saa_samples={args.saa_samples}, beta={args.beta}")
            print("=" * 78)

        _, Vh, control_model, prior, penalty = setup_problem(args, comm_mesh)
        true_cost = make_saa_cost(control_model, prior, penalty, args.beta, args.saa_samples, args.saa_seed)
        approx_costs = {m: make_taylor_cost(m, control_model, prior, penalty, args) for m in MODEL_ORDER}

        controls = build_test_controls(Vh[soupy.CONTROL], args.z_seed)
        rows: List[Dict] = []
        control_truth: Dict[str, float] = {}
        control_norms: Dict[str, Dict[str, float]] = {}

        for cname, z_vec in controls:
            j_true = evaluate_cost(true_cost, z_vec)
            q_mean, q_var = evaluate_mean_variance(true_cost, z_vec)
            control_truth[cname] = j_true
            control_norms[cname] = {"z_l2_norm": vec_l2_norm(z_vec), "z_linf_norm": vec_linf_norm(z_vec)}
            if rank == 0:
                print(
                    f"[{cname:12s}] J_true={j_true:.6e}, mean={q_mean:.6e}, var={q_var:.6e}, "
                    f"||z||_2={control_norms[cname]['z_l2_norm']:.3e}, "
                    f"||z||_inf={control_norms[cname]['z_linf_norm']:.3e}"
                )

            for model_name in MODEL_ORDER:
                j_model = evaluate_cost(approx_costs[model_name], z_vec)
                abs_err = abs(j_model - j_true)
                rel_err = abs_err / max(abs(j_true), 1e-14)
                row = {
                    "control_name": cname,
                    "model": model_name,
                    "model_cost": j_model,
                    "true_cost": j_true,
                    "abs_error": abs_err,
                    "rel_error": rel_err,
                    "z_l2_norm": control_norms[cname]["z_l2_norm"],
                    "z_linf_norm": control_norms[cname]["z_linf_norm"],
                }
                rows.append(row)
                if rank == 0:
                    print(f"  - {model_name:21s} J_model={j_model:.6e}, |err|={abs_err:.3e}, rel={rel_err:.3e}")

        summary_rows = []
        for model_name in MODEL_ORDER:
            rr = [r for r in rows if r["model"] == model_name]
            summary_rows.append(
                {
                    "model": model_name,
                    "mean_abs_error": float(np.mean([r["abs_error"] for r in rr])),
                    "max_abs_error": float(np.max([r["abs_error"] for r in rr])),
                    "mean_rel_error": float(np.mean([r["rel_error"] for r in rr])),
                    "max_rel_error": float(np.max([r["rel_error"] for r in rr])),
                }
            )

        if rank == 0:
            save_rows_csv(os.path.join(args.save_dir, "fixed_control_model_errors.csv"), rows)
            save_model_summary_csv(os.path.join(args.save_dir, "fixed_control_model_summary.csv"), summary_rows)
            with open(os.path.join(args.save_dir, "fixed_control_truth_costs.json"), "w") as f:
                json.dump(
                    {
                        "config": vars(args),
                        "true_cost_by_control": control_truth,
                        "control_norms": control_norms,
                        "model_summary": summary_rows,
                    },
                    f,
                    indent=2,
                )
            plot_abs_error(rows, args.save_dir)

            print("\n" + "-" * 78)
            print("Model summary over fixed controls")
            print("-" * 78)
            for s in summary_rows:
                print(
                    f"{s['model']:21s} mean|err|={s['mean_abs_error']:.3e}, max|err|={s['max_abs_error']:.3e}, "
                    f"mean rel={s['mean_rel_error']:.3e}, max rel={s['max_rel_error']:.3e}"
                )
            print("-" * 78)
            print(f"Outputs written to: {args.save_dir}")
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
