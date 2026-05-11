"""Stage-level memory diagnostic for quadratic CVaR eigendecomposition.

This script isolates the quadratic Taylor CVaR legacy solver and repeatedly
executes selected internal stages:

1. ``_linearize_at_mean()``
2. ``_compute_eigendecomposition()``
3. both in sequence

The goal is to determine whether memory growth comes from linearization,
the randomized eigensolver, or both.
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
from soupy.approximations.taylor import TaylorQuadraticCVaRControlCostFunctional

from diagnose_hyperelasticity_cvar_memory import (
    build_control_np,
    get_current_rss_mb,
    is_augmented_vector,
    set_control_part,
    setup_problem,
)


logging.getLogger("FFC").setLevel(logging.WARNING)
logging.getLogger("UFL").setLevel(logging.WARNING)
dl.set_log_active(False)


@dataclass
class RunSummary:
    label: str
    rss_start_mb: float
    rss_end_mb: float
    rss_max_mb: float


def make_quadratic_cost(control_model, prior, penalty, args):
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


def build_augmented_point(cost, control_np, scalar_t):
    point = cost.generate_vector(soupy.CONTROL)
    set_control_part(point, control_np, scalar=scalar_t)
    return point


def _reset_legacy_for_stage(legacy, point):
    legacy._copy_zt(point)
    legacy.grad_cache = None


def repeat_linearize_only(legacy, point, repeats, print_every, collect_gc):
    rss_start = get_current_rss_mb()
    rss_max = rss_start
    for k in range(repeats):
        _reset_legacy_for_stage(legacy, point)
        rss_before = get_current_rss_mb()
        legacy._linearize_at_mean()
        if collect_gc:
            gc.collect()
        rss_after = get_current_rss_mb()
        rss_max = max(rss_max, rss_after)
        if (k + 1) % print_every == 0 or k == 0 or k + 1 == repeats:
            print(
                f"[lin  ] rep {k+1:4d}: rss={rss_after:.1f} MB, "
                f"delta_call={rss_after - rss_before:.1f} MB, "
                f"delta_total={rss_after - rss_start:.1f} MB"
            )
            sys.stdout.flush()
    rss_end = get_current_rss_mb()
    return RunSummary("linearize", rss_start, rss_end, rss_max)


def repeat_eig_only(legacy, point, repeats, print_every, collect_gc):
    _reset_legacy_for_stage(legacy, point)
    legacy._linearize_at_mean()

    rss_start = get_current_rss_mb()
    rss_max = rss_start
    for k in range(repeats):
        rss_before = get_current_rss_mb()
        legacy._compute_eigendecomposition()
        if collect_gc:
            gc.collect()
        rss_after = get_current_rss_mb()
        rss_max = max(rss_max, rss_after)
        eig_count = len(legacy.d) if hasattr(legacy.d, "__len__") else 0
        top_eval = float(legacy.d[0]) if eig_count > 0 else float("nan")
        if (k + 1) % print_every == 0 or k == 0 or k + 1 == repeats:
            print(
                f"[eig  ] rep {k+1:4d}: top_eval={top_eval:.6e}, rss={rss_after:.1f} MB, "
                f"delta_call={rss_after - rss_before:.1f} MB, "
                f"delta_total={rss_after - rss_start:.1f} MB"
            )
            sys.stdout.flush()
    rss_end = get_current_rss_mb()
    return RunSummary("eigendecomposition", rss_start, rss_end, rss_max)


def repeat_linearize_plus_eig(legacy, point, repeats, print_every, collect_gc):
    rss_start = get_current_rss_mb()
    rss_max = rss_start
    for k in range(repeats):
        _reset_legacy_for_stage(legacy, point)
        rss_before = get_current_rss_mb()
        legacy._linearize_at_mean()
        legacy._compute_eigendecomposition()
        if collect_gc:
            gc.collect()
        rss_after = get_current_rss_mb()
        rss_max = max(rss_max, rss_after)
        eig_count = len(legacy.d) if hasattr(legacy.d, "__len__") else 0
        top_eval = float(legacy.d[0]) if eig_count > 0 else float("nan")
        if (k + 1) % print_every == 0 or k == 0 or k + 1 == repeats:
            print(
                f"[both ] rep {k+1:4d}: top_eval={top_eval:.6e}, rss={rss_after:.1f} MB, "
                f"delta_call={rss_after - rss_before:.1f} MB, "
                f"delta_total={rss_after - rss_start:.1f} MB"
            )
            sys.stdout.flush()
    rss_end = get_current_rss_mb()
    return RunSummary("linearize+eigendecomposition", rss_start, rss_end, rss_max)


def print_summary(summary):
    print(
        f"[{summary.label}] summary: start={summary.rss_start_mb:.1f} MB, "
        f"end={summary.rss_end_mb:.1f} MB, max={summary.rss_max_mb:.1f} MB, "
        f"net={summary.rss_end_mb - summary.rss_start_mb:.1f} MB"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Memory diagnostic for quadratic CVaR eigendecomposition stages."
    )
    parser.add_argument("--mode", choices=["linearize", "eig", "both"], default="eig")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--collect-gc", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--qoi-type", default="virtual_work",
                        choices=["all", "stiffness", "point", "virtual_work"])
    parser.add_argument("--cvar-beta", type=float, default=0.95)
    parser.add_argument("--n-tr", type=int, default=10)
    parser.add_argument("--quadratic-cvar-n-mc", type=int, default=1000)
    parser.add_argument("--scalar-t", type=float, default=0.0)
    parser.add_argument("--penalty", type=float, default=1e-3)

    parser.add_argument("--init-mode", choices=["zero", "constant", "random"], default="zero")
    parser.add_argument("--init-value", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--lx", type=float, default=2.0)
    parser.add_argument("--ly", type=float, default=0.5)
    parser.add_argument("--lz", type=float, default=0.25)
    parser.add_argument("--geometry-dim", type=int, default=2)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--nz", type=int, default=4)

    return parser.parse_args()


def main():
    args = parse_args()

    control_model, prior, penalty = setup_problem(args)
    cost = make_quadratic_cost(control_model, prior, penalty, args)
    legacy = cost._legacy

    control_np = build_control_np(control_model, args.init_mode, args.init_value, args.seed)
    point = build_augmented_point(cost, control_np, args.scalar_t)

    print(f"mode={args.mode}, repeats={args.repeats}, qoi={args.qoi_type}")
    print(
        f"n_tr={args.n_tr}, quadratic_cvar_n_mc={args.quadratic_cvar_n_mc}, "
        f"beta={args.cvar_beta}, scalar_t={args.scalar_t}"
    )
    print(f"rss_before_stage_test={get_current_rss_mb():.1f} MB")

    if args.mode == "linearize":
        summary = repeat_linearize_only(
            legacy, point, args.repeats, args.print_every, args.collect_gc
        )
    elif args.mode == "eig":
        summary = repeat_eig_only(
            legacy, point, args.repeats, args.print_every, args.collect_gc
        )
    else:
        summary = repeat_linearize_plus_eig(
            legacy, point, args.repeats, args.print_every, args.collect_gc
        )

    print_summary(summary)


if __name__ == "__main__":
    main()
