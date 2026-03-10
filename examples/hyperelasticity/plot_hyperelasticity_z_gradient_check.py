"""Directional finite-difference check for z-gradient on hyperelasticity."""

import argparse
import logging
import os
import sys

import dolfin as dl
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SOUPY_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../../"))
if _SOUPY_ROOT not in sys.path:
    sys.path.insert(0, _SOUPY_ROOT)
if os.environ.get("HIPPYLIB_PATH", "") not in sys.path:
    sys.path.append(os.environ.get("HIPPYLIB_PATH", ""))

import soupy
from soupy import CONTROL
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


def sweep_fd_error(cost, z, dz, epsilons):
    g = cost.generate_vector(CONTROL)
    q0 = cost.cost(z, order=1)
    cost.grad(g)
    true_dir = g.inner(dz)

    z_eps = cost.generate_vector(CONTROL)
    fd_values = np.zeros_like(epsilons)
    abs_errors = np.zeros_like(epsilons)
    rel_errors = np.zeros_like(epsilons)
    denom = max(abs(true_dir), 1e-14)

    for i, eps in enumerate(epsilons):
        z_eps.zero()
        z_eps.axpy(1.0, z)
        z_eps.axpy(float(eps), dz)
        q_eps = cost.cost(z_eps, order=0)
        fd = (q_eps - q0) / float(eps)
        fd_values[i] = fd
        abs_errors[i] = abs(fd - true_dir)
        rel_errors[i] = abs_errors[i] / denom

    return true_dir, fd_values, abs_errors, rel_errors


def fit_loglog_slope(epsilons, errors, start_idx, end_idx):
    i0 = max(0, start_idx)
    i1 = min(len(epsilons), end_idx)
    x = np.log10(epsilons[i0:i1])
    y = np.log10(errors[i0:i1])
    slope, _ = np.polyfit(x, y, 1)
    return slope


def main():
    parser = argparse.ArgumentParser(description="z-gradient FD check for hyperelasticity")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--n-tr", type=int, default=10)
    parser.add_argument("--n-mix", type=int, default=11)
    parser.add_argument("--mix-direction", type=str, default="kle", choices=["hep", "kle"])
    parser.add_argument("--qoi-type", type=str, default="stiffness", choices=["all", "stiffness", "point"])
    parser.add_argument("--penalty", type=float, default=1e-1)
    parser.add_argument("--eps-min", type=float, default=1e-4)
    parser.add_argument("--eps-max", type=float, default=1e1)
    parser.add_argument("--n-eps", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fit-start", type=int, default=2)
    parser.add_argument("--fit-end", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default="results_z_gradient_check")
    parser.add_argument("--out", type=str, default="hyperelasticity_z_gradient_fd_error.png")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    out = args.out if os.path.isabs(args.out) else os.path.join(args.save_dir, args.out)

    epsilons = np.logspace(np.log10(args.eps_min), np.log10(args.eps_max), args.n_eps)

    settings = hyperelasticity_problem_settings()
    settings["qoi_type"] = args.qoi_type
    _, Vh, _, control_model, prior = setup_hyperelasticity_problem(settings, MPI.COMM_SELF)
    penalty = soupy.L2Penalization(Vh, args.penalty)

    lin = TaylorLinearControlCostFunctional(control_model, prior, penalty, {"beta": args.beta})
    mix_lin = TaylorMixtureLinearControlCostFunctional(
        control_model,
        prior,
        penalty,
        {"beta": args.beta, "N_mix": args.n_mix, "direction": args.mix_direction},
    )
    quad = TaylorQuadraticControlCostFunctional(control_model, prior, penalty, {"beta": args.beta, "N_tr": args.n_tr})
    mix_quad = TaylorMixtureQuadraticControlCostFunctional(
        control_model,
        prior,
        penalty,
        {"beta": args.beta, "N_mix": args.n_mix, "direction": args.mix_direction, "N_tr": args.n_tr},
    )

    z = lin.generate_vector(CONTROL)
    z.set_local(np.full(z.local_size(), 0.5))
    z.apply("")

    dz = lin.generate_vector(CONTROL)
    np.random.seed(args.seed)
    dz.set_local(np.random.randn(dz.local_size()))
    dz.apply("")
    dz_norm = np.sqrt(dz.inner(dz))
    if dz_norm > 0:
        dlocal = dz.get_local() / dz_norm
        dz.set_local(dlocal)
        dz.apply("")

    lin_true, _, lin_err, lin_rel = sweep_fd_error(lin, z, dz, epsilons)
    mix_true, _, mix_err, mix_rel = sweep_fd_error(mix_lin, z, dz, epsilons)
    quad_true, _, quad_err, quad_rel = sweep_fd_error(quad, z, dz, epsilons)
    mix_quad_true, _, mix_quad_err, mix_quad_rel = sweep_fd_error(mix_quad, z, dz, epsilons)

    lin_s = fit_loglog_slope(epsilons, lin_err, args.fit_start, args.fit_end)
    mix_s = fit_loglog_slope(epsilons, mix_err, args.fit_start, args.fit_end)
    quad_s = fit_loglog_slope(epsilons, quad_err, args.fit_start, args.fit_end)
    mix_quad_s = fit_loglog_slope(epsilons, mix_quad_err, args.fit_start, args.fit_end)

    print("Directional derivative check:")
    print(f"  linear:    g.dz = {lin_true:.12e}")
    print(f"  mix-linear: g.dz = {mix_true:.12e}")
    print(f"  quadratic: g.dz = {quad_true:.12e}")
    print(f"  mix-quad:  g.dz = {mix_quad_true:.12e}")
    print("Estimated slope:")
    print(f"  linear={lin_s:.3f}, mix-linear={mix_s:.3f}, quadratic={quad_s:.3f}, mix-quad={mix_quad_s:.3f}")

    plt.figure(figsize=(7, 5))
    plt.loglog(epsilons, lin_err, "o-", label=f"Linear ({lin_s:.2f})")
    plt.loglog(epsilons, mix_err, "^-", label=f"Mix linear ({mix_s:.2f})")
    plt.loglog(epsilons, quad_err, "s-", label=f"Quadratic ({quad_s:.2f})")
    plt.loglog(epsilons, mix_quad_err, "d-", label=f"Mix quad ({mix_quad_s:.2f})")
    ref = lin_err[max(1, min(len(lin_err) - 1, args.fit_start))]
    eps_ref = epsilons[max(1, min(len(epsilons) - 1, args.fit_start))]
    plt.loglog(epsilons, ref * (epsilons / eps_ref), "k--", linewidth=1.0, label="O(eps)")
    plt.xlabel("epsilon")
    plt.ylabel("|FD - g.dz|")
    plt.title("Hyperelasticity z-gradient FD error")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    print(f"Saved plot to: {out}")

    out_root, out_ext = os.path.splitext(out)
    if out_ext == "":
        out_ext = ".png"
    rel_out = f"{out_root}_relative{out_ext}"

    plt.figure(figsize=(7, 5))
    plt.loglog(epsilons, lin_rel, "o-", label="Linear")
    plt.loglog(epsilons, mix_rel, "^-", label="Mix linear")
    plt.loglog(epsilons, quad_rel, "s-", label="Quadratic")
    plt.loglog(epsilons, mix_quad_rel, "d-", label="Mix quad")
    plt.xlabel("epsilon")
    plt.ylabel("|FD - g.dz| / max(|g.dz|, 1e-14)")
    plt.title("Hyperelasticity z-gradient relative FD error")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(rel_out, dpi=180)
    print(f"Saved plot to: {rel_out}")


if __name__ == "__main__":
    main()
