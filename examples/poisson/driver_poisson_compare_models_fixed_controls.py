"""Compare model approximation errors on fixed controls (no optimization).

This script uses the same Poisson setup and model constructors as
`driver_poisson_compare_taylor_models.py`, but evaluates approximation
errors only at predefined controls z.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import math
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

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
from mpi4py import MPI

import soupy
from driver_poisson_compare_taylor_models import MODEL_ORDER, make_saa_cost, make_taylor_cost, setup_problem


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


def _control_from_expression(Vh_control, expr_str: str, amp: float = 1.0):
    expr = dl.Expression(expr_str, a=amp, degree=2, mpi_comm=Vh_control.mesh().mpi_comm())
    return dl.interpolate(expr, Vh_control).vector()


def build_test_controls(Vh_control, seed: int) -> List[Tuple[str, dl.Vector]]:
    controls: List[Tuple[str, dl.Vector]] = []
    controls.append(("zero", _control_from_expression(Vh_control, "0.0", amp=1.0)))
    controls.append(("affine_small", _control_from_expression(Vh_control, "a*(x[0]+x[1]-1.0)", amp=0.5)))
    controls.append(("affine_large", _control_from_expression(Vh_control, "a*(x[0]+x[1]-1.0)", amp=2.0)))
    controls.append(("sin_small", _control_from_expression(Vh_control, "a*sin(2*pi*x[0])*sin(2*pi*x[1])", amp=0.5)))
    controls.append(("sin_large", _control_from_expression(Vh_control, "a*sin(2*pi*x[0])*sin(2*pi*x[1])", amp=2.0)))

    rng = np.random.default_rng(seed)
    z_rand = dl.Function(Vh_control).vector()
    z_rand_arr = rng.standard_normal(z_rand.get_local().shape[0])
    z_rand.set_local(0.3 * z_rand_arr)
    z_rand.apply("")
    controls.append(("random_small", z_rand))

    z_rand2 = dl.Function(Vh_control).vector()
    z_rand_arr2 = rng.standard_normal(z_rand2.get_local().shape[0])
    z_rand2.set_local(1.0 * z_rand_arr2)
    z_rand2.apply("")
    controls.append(("random_large", z_rand2))
    return controls


def evaluate_cost(cost_functional, z_vec: dl.Vector) -> float:
    return float(cost_functional.cost(z_vec, order=0))


def evaluate_true_cost_stats(true_cost_functional, z_vec: dl.Vector) -> Dict[str, float]:
    j_true = float(true_cost_functional.cost(z_vec, order=0))
    mean_q = float("nan")
    var_q = float("nan")
    penalty_cost = 0.0

    risk_measure = getattr(true_cost_functional, "risk_measure", None)
    if risk_measure is not None and hasattr(risk_measure, "q_bar") and hasattr(risk_measure, "q2_bar"):
        mean_q = float(risk_measure.q_bar)
        var_q = float(max(risk_measure.q2_bar - risk_measure.q_bar**2, 0.0))

    penalization = getattr(true_cost_functional, "penalization", None)
    if penalization is not None:
        penalty_cost = float(penalization.cost(z_vec))

    return {
        "true_cost": j_true,
        "mean_q": mean_q,
        "var_q": var_q,
        "penalty_cost": penalty_cost,
    }


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
        "rel_error",
        "z_l2_norm",
        "z_linf_norm",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_model_summary_csv(path: str, summary_rows: List[Dict]):
    fieldnames = ["model", "mean_rel_error", "max_rel_error"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def plot_rel_error(rows: List[Dict], save_dir: str):
    controls = sorted({r["control_name"] for r in rows})
    x = np.arange(len(controls))
    # Keep grouped bars within ~80% of unit spacing so bars do not crowd together.
    width = min(0.12, 0.8 / max(len(MODEL_ORDER), 1))

    plt.figure(figsize=(10, 5))
    for i, model in enumerate(MODEL_ORDER):
        vals = []
        for cname in controls:
            matched = [r for r in rows if r["control_name"] == cname and r["model"] == model]
            vals.append(matched[0]["rel_error"] if matched else np.nan)
        plt.bar(x + (i - 1.5) * width, vals, width=width, color=MODEL_COLORS[model], label=model)

    plt.yscale("log")
    plt.xticks(x, controls, rotation=20, ha="right")
    plt.ylabel("Relative Approximation Error")
    plt.title("Model Relative Approximation Error at Fixed Controls")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fixed_control_rel_error_comparison.png"), dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare approximation error at fixed controls (no optimization)")
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--n-tr", type=int, default=10)
    parser.add_argument("--n-mix", type=int, default=39)
    parser.add_argument("--mix-linear-direction", type=str, default="kle", choices=["hep", "kle"])
    parser.add_argument("--mix-quadratic-direction", type=str, default="kle", choices=["hep", "kle"])
    parser.add_argument("--saa-samples", type=int, default=1000000)
    parser.add_argument("--saa-seed", type=int, default=1)
    parser.add_argument("--prior-gamma", type=float, default=0.2)
    parser.add_argument("--prior-delta", type=float, default=1.0)
    parser.add_argument("--penalty", type=float, default=1e-2)
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--ny", type=int, default=20)
    parser.add_argument("--z-seed", type=int, default=7, help="Seed for random fixed controls")
    parser.add_argument("--save-dir", type=str, default="results_fixed_z_model_error")
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    args = parser.parse_args()

    rank = MPI.COMM_WORLD.Get_rank()
    comm_mesh = MPI.COMM_SELF
    os.makedirs(args.save_dir, exist_ok=True)

    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(args.save_dir, f"terminal_output.txt")
        log_file = open(log_path, "w")
        sys.stdout = TeeStream(sys.stdout, log_file)
        sys.stderr = TeeStream(sys.stderr, log_file)

        def _cleanup_log():
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                log_file.close()

        atexit.register(_cleanup_log)

    if rank == 0:
        print("=" * 78)
        print("Poisson Fixed-Control Approximation Error Comparison")
        print("=" * 78)
        print(f"log_file={log_path}")
        print(f"save_dir={args.save_dir}")
        print(f"saa_samples={args.saa_samples}, beta={args.beta}")
        print("=" * 78)

    _, Vh, control_model, prior, penalty = setup_problem(args, comm_mesh)
    true_cost = make_saa_cost(
        control_model,
        prior,
        penalty,
        beta=args.beta,
        sample_size=args.saa_samples,
        seed=args.saa_seed,
    )
    approx_costs = {
        model_name: make_taylor_cost(model_name, control_model, prior, penalty, args)
        for model_name in MODEL_ORDER
    }

    controls = build_test_controls(Vh[soupy.CONTROL], args.z_seed)
    rows: List[Dict] = []
    control_truth: Dict[str, Dict[str, float]] = {}
    control_norms: Dict[str, Dict[str, float]] = {}

    for cname, z_vec in controls:
        true_stats = evaluate_true_cost_stats(true_cost, z_vec)
        j_true = true_stats["true_cost"]
        control_truth[cname] = true_stats
        control_norms[cname] = {
            "z_l2_norm": vec_l2_norm(z_vec),
            "z_linf_norm": vec_linf_norm(z_vec),
        }
        if rank == 0:
            print(
                f"[{cname:12s}] J_true={j_true:.6e}, "
                f"E[Q]={true_stats['mean_q']:.6e}, Var[Q]={true_stats['var_q']:.6e}, "
                f"P(z)={true_stats['penalty_cost']:.6e}, "
                f"||z||_2={control_norms[cname]['z_l2_norm']:.3e}, "
                f"||z||_inf={control_norms[cname]['z_linf_norm']:.3e}"
            )

        for model_name in MODEL_ORDER:
            j_model = evaluate_cost(approx_costs[model_name], z_vec)
            rel_err = abs(j_model - j_true) / max(abs(j_true), 1e-14)
            row = {
                "control_name": cname,
                "model": model_name,
                "model_cost": j_model,
                "true_cost": j_true,
                "rel_error": rel_err,
                "z_l2_norm": control_norms[cname]["z_l2_norm"],
                "z_linf_norm": control_norms[cname]["z_linf_norm"],
            }
            rows.append(row)
            if rank == 0:
                print(
                    f"  - {model_name:16s} J_model={j_model:.6e}, "
                    f"rel_err={rel_err:.3e}"
                )

    model_summary_rows = []
    for model_name in MODEL_ORDER:
        rr = [r for r in rows if r["model"] == model_name]
        model_summary_rows.append(
            {
                "model": model_name,
                "mean_rel_error": float(np.mean([r["rel_error"] for r in rr])),
                "max_rel_error": float(np.max([r["rel_error"] for r in rr])),
            }
        )

    if rank == 0:
        save_rows_csv(os.path.join(args.save_dir, "fixed_control_model_errors.csv"), rows)
        save_model_summary_csv(os.path.join(args.save_dir, "fixed_control_model_summary.csv"), model_summary_rows)
        with open(os.path.join(args.save_dir, "fixed_control_truth_costs.json"), "w") as f:
            json.dump(
                {
                    "config": vars(args),
                    "true_cost_by_control": control_truth,
                    "control_norms": control_norms,
                    "model_summary": model_summary_rows,
                },
                f,
                indent=2,
            )
        plot_rel_error(rows, args.save_dir)

        print("\n" + "-" * 78)
        print("Model summary over fixed controls")
        print("-" * 78)
        for s in model_summary_rows:
            print(
                f"{s['model']:16s} "
                f"mean rel={s['mean_rel_error']:.3e}, max rel={s['max_rel_error']:.3e}"
            )
        print("-" * 78)
        print(f"Outputs written to: {args.save_dir}")


if __name__ == "__main__":
    main()
