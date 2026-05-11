"""Repeated fixed-z memory diagnostic for hyperelasticity CVaR models.

This script is the first-stage memory test:

1. Fix one control vector z (and one scalar t when the model uses an
   augmented control variable).
2. Repeatedly call either
      - cost(point, order=0), or
      - cost(point, order=1) followed by grad(g)
3. Print the resident memory after each repetition.

The goal is to separate objective-path memory growth from gradient-path
memory growth without involving the optimizer or line search.
"""

import argparse
import gc
import logging
import os
import sys
from dataclasses import dataclass

import dolfin as dl
import numpy as np
from mpi4py import MPI

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SOUPY_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../../"))
if _SOUPY_ROOT not in sys.path:
    sys.path.insert(0, _SOUPY_ROOT)
if os.environ.get("HIPPYLIB_PATH", "") not in sys.path:
    sys.path.append(os.environ.get("HIPPYLIB_PATH", ""))

import soupy
from soupy.approximations.taylor import (
    TaylorLinearCVaRControlCostFunctional,
    TaylorMixtureLinearCVaRControlCostFunctional,
    TaylorMixtureQuadraticCVaRControlCostFunctional,
    TaylorQuadraticCVaRControlCostFunctional,
)

from setupHyperelasticityProblem import hyperelasticity_problem_settings, setup_hyperelasticity_problem


logging.getLogger("FFC").setLevel(logging.WARNING)
logging.getLogger("UFL").setLevel(logging.WARNING)
dl.set_log_active(False)

MODEL_CHOICES = [
    "linear",
    "quadratic",
    "mixture_linear_kle",
    "mixture_linear_hep",
    "mixture_quadratic_kle",
    "mixture_quadratic_hep",
]


def get_current_rss_mb():
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 ** 2)
    except Exception:
        pass

    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return float(parts[1]) / 1024.0
    except Exception:
        pass

    return float("nan")


def is_augmented_vector(vector):
    return hasattr(vector, "get_vector") and hasattr(vector, "get_scalar")


def set_control_part(vector, control_np, scalar=0.0):
    if is_augmented_vector(vector):
        vector.get_vector().set_local(control_np)
        vector.get_vector().apply("")
        vector.set_scalar(float(scalar))
    else:
        vector.set_local(control_np)
        vector.apply("")


def build_control_np(control_model, init_mode, value, seed):
    z = control_model.generate_vector(soupy.CONTROL)
    if init_mode == "zero":
        z.zero()
        z.apply("")
        return np.array(z.get_local(), copy=True)

    if init_mode == "constant":
        z.set_local(np.full(z.local_size(), float(value)))
        z.apply("")
        return np.array(z.get_local(), copy=True)

    rng = np.random.default_rng(seed)
    z.set_local(rng.standard_normal(z.local_size()))
    z.apply("")
    return np.array(z.get_local(), copy=True)


def setup_problem(args):
    settings = hyperelasticity_problem_settings()
    settings["qoi_type"] = args.qoi_type
    settings["geometry"]["lx"] = args.lx
    settings["geometry"]["ly"] = args.ly
    settings["geometry"]["lz"] = args.lz
    settings["geometry"]["dim"] = args.geometry_dim
    settings["mesh"]["nx"] = args.nx
    settings["mesh"]["ny"] = args.ny
    settings["mesh"]["nz"] = args.nz
    _, Vh, _, control_model, prior = setup_hyperelasticity_problem(settings, MPI.COMM_SELF)
    penalty = soupy.L2Penalization(Vh, args.penalty)
    return control_model, prior, penalty


def make_cvar_cost(model_name, control_model, prior, penalty, args):
    if model_name == "linear":
        return TaylorLinearCVaRControlCostFunctional(
            control_model,
            prior,
            penalty,
            {"beta": args.cvar_beta, "verbose": args.verbose},
        )
    if model_name == "quadratic":
        return TaylorQuadraticCVaRControlCostFunctional(
            control_model,
            prior,
            penalty,
            {
                "beta": args.cvar_beta,
                "N_tr": args.n_tr,
                "N_mc": args.quadratic_cvar_n_mc,
                "verbose": args.verbose,
            },
        )
    if model_name == "mixture_linear_kle":
        return TaylorMixtureLinearCVaRControlCostFunctional(
            control_model,
            prior,
            penalty,
            {"beta": args.cvar_beta, "N_mix": args.n_mix, "direction": "kle", "verbose": args.verbose},
        )
    if model_name == "mixture_linear_hep":
        return TaylorMixtureLinearCVaRControlCostFunctional(
            control_model,
            prior,
            penalty,
            {"beta": args.cvar_beta, "N_mix": args.n_mix, "direction": "hep", "verbose": args.verbose},
        )
    if model_name == "mixture_quadratic_kle":
        return TaylorMixtureQuadraticCVaRControlCostFunctional(
            control_model,
            prior,
            penalty,
            {
                "beta": args.cvar_beta,
                "N_mix": args.n_mix,
                "direction": "kle",
                "N_tr": args.n_tr,
                "N_mc": args.quadratic_cvar_n_mc,
                "verbose": args.verbose,
            },
        )
    if model_name == "mixture_quadratic_hep":
        return TaylorMixtureQuadraticCVaRControlCostFunctional(
            control_model,
            prior,
            penalty,
            {
                "beta": args.cvar_beta,
                "N_mix": args.n_mix,
                "direction": "hep",
                "N_tr": args.n_tr,
                "N_mc": args.quadratic_cvar_n_mc,
                "verbose": args.verbose,
            },
        )
    raise ValueError(f"Unsupported model: {model_name}")


@dataclass
class RunSummary:
    label: str
    rss_start_mb: float
    rss_end_mb: float
    rss_max_mb: float


def repeat_cost_only(cost, point, repeats, print_every, collect_gc):
    rss_start = get_current_rss_mb()
    rss_max = rss_start
    for k in range(repeats):
        rss_before = get_current_rss_mb()
        value = float(cost.cost(point, order=0))
        if collect_gc:
            gc.collect()
        rss_after = get_current_rss_mb()
        rss_max = max(rss_max, rss_after)
        if (k + 1) % print_every == 0 or k == 0 or k + 1 == repeats:
            print(
                f"[cost ] rep {k+1:4d}: value={value:.6e}, "
                f"rss={rss_after:.1f} MB, delta_call={rss_after - rss_before:.1f} MB, "
                f"delta_total={rss_after - rss_start:.1f} MB"
            )
            sys.stdout.flush()
    rss_end = get_current_rss_mb()
    return RunSummary("cost", rss_start, rss_end, rss_max)


def repeat_cost1_grad(cost, point, repeats, print_every, collect_gc):
    g = cost.generate_vector(soupy.CONTROL)
    rss_start = get_current_rss_mb()
    rss_max = rss_start
    for k in range(repeats):
        rss_before = get_current_rss_mb()
        value = float(cost.cost(point, order=1))
        grad_norm = float(cost.grad(g))
        if collect_gc:
            gc.collect()
        rss_after = get_current_rss_mb()
        rss_max = max(rss_max, rss_after)
        if (k + 1) % print_every == 0 or k == 0 or k + 1 == repeats:
            print(
                f"[grad ] rep {k+1:4d}: value={value:.6e}, ||g||={grad_norm:.3e}, "
                f"rss={rss_after:.1f} MB, delta_call={rss_after - rss_before:.1f} MB, "
                f"delta_total={rss_after - rss_start:.1f} MB"
            )
            sys.stdout.flush()
    rss_end = get_current_rss_mb()
    return RunSummary("grad", rss_start, rss_end, rss_max)


def run_experiment(label, args, control_np):
    control_model, prior, penalty = setup_problem(args)
    cost = make_cvar_cost(args.model, control_model, prior, penalty, args)
    point = cost.generate_vector(soupy.CONTROL)
    set_control_part(point, control_np, scalar=args.scalar_t)

    print("")
    print(f"=== {label} ===")
    print(f"model={args.model}, repeats={args.repeats}, qoi={args.qoi_type}")
    print(
        f"n_tr={args.n_tr}, n_mix={args.n_mix}, quadratic_cvar_n_mc={args.quadratic_cvar_n_mc}, "
        f"beta={args.cvar_beta}, scalar_t={args.scalar_t}"
    )
    print(f"rss_before_build={get_current_rss_mb():.1f} MB")

    if label == "cost":
        summary = repeat_cost_only(cost, point, args.repeats, args.print_every, args.collect_gc)
    else:
        summary = repeat_cost1_grad(cost, point, args.repeats, args.print_every, args.collect_gc)

    print(
        f"[{summary.label}] summary: start={summary.rss_start_mb:.1f} MB, "
        f"end={summary.rss_end_mb:.1f} MB, max={summary.rss_max_mb:.1f} MB, "
        f"net={summary.rss_end_mb - summary.rss_start_mb:.1f} MB"
    )
    sys.stdout.flush()

    del point
    del cost
    del penalty
    del prior
    del control_model
    if args.collect_gc:
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Repeated fixed-z memory diagnostic for hyperelasticity CVaR models")
    parser.add_argument("--model", type=str, default="mixture_quadratic_kle", choices=MODEL_CHOICES)
    parser.add_argument("--mode", type=str, default="both", choices=["cost", "grad", "both"])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--collect-gc", action="store_true", help="Call gc.collect() after each repetition")

    parser.add_argument("--qoi-type", type=str, default="virtual_work",
                        choices=["all", "stiffness", "point", "virtual_work"])
    parser.add_argument("--penalty", type=float, default=1e-2)
    parser.add_argument("--cvar-beta", type=float, default=0.95)
    parser.add_argument("--n-tr", type=int, default=10)
    parser.add_argument("--n-mix", type=int, default=39)
    parser.add_argument("--quadratic-cvar-n-mc", type=int, default=1000)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--control-init", type=str, default="zero", choices=["zero", "constant", "random"])
    parser.add_argument("--control-value", type=float, default=0.5)
    parser.add_argument("--scalar-t", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--lx", type=float, default=2.0)
    parser.add_argument("--ly", type=float, default=0.5)
    parser.add_argument("--lz", type=float, default=0.25)
    parser.add_argument("--geometry-dim", type=int, default=2, choices=[2, 3])
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--nz", type=int, default=12)
    args = parser.parse_args()

    base_control_model, _, _ = setup_problem(args)
    control_np = build_control_np(base_control_model, args.control_init, args.control_value, args.seed)
    del base_control_model
    if args.collect_gc:
        gc.collect()

    print("Repeated fixed-z memory diagnostic")
    print(
        f"initial_rss={get_current_rss_mb():.1f} MB, model={args.model}, mode={args.mode}, "
        f"control_init={args.control_init}"
    )
    sys.stdout.flush()

    if args.mode in ("cost", "both"):
        run_experiment("cost", args, control_np)
    if args.mode in ("grad", "both"):
        run_experiment("grad", args, control_np)


if __name__ == "__main__":
    main()
