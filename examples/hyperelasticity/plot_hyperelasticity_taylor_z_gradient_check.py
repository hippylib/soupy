"""Directional finite-difference check for z-gradient of hyperelasticity Taylor approximations.

For each approximation, this script computes

    FD(eps) = [Q(z + eps * dz) - Q(z)] / eps

and compares it with the algorithmic directional derivative

    g(z) . dz

where g(z) is the z-gradient returned by the cost functional.

The script plots |FD(eps) - g.dz| versus eps on log-log axes. For a forward
difference with accurate gradients, the error should scale approximately as
O(eps) over a suitable epsilon range.
"""

import argparse
import logging
import os
import sys

import dolfin as dl
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI


logging.getLogger("FFC").setLevel(logging.WARNING)
logging.getLogger("UFL").setLevel(logging.WARNING)
dl.set_log_active(False)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SOUPY_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../.."))
if SOUPY_ROOT not in sys.path:
    sys.path.insert(0, SOUPY_ROOT)
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


def sweep_fd_error(cost, z, dz, epsilons):
    """Compare algorithmic and FD directional derivatives along dz."""
    g = cost.generate_vector(CONTROL)

    q0 = cost.cost(z, order=1, FD_gradient_check=True)
    cost.grad(g)
    directional_true = g.inner(dz)

    z_eps = cost.generate_vector(CONTROL)
    fd_values = np.zeros_like(epsilons)
    abs_errors = np.zeros_like(epsilons)
    rel_errors = np.zeros_like(epsilons)
    denom = max(abs(directional_true), 1e-14)

    for i, eps in enumerate(epsilons):
        z_eps.zero()
        z_eps.axpy(1.0, z)
        z_eps.axpy(float(eps), dz)

        q_eps = cost.cost(z_eps, order=0, FD_gradient_check=True)
        directional_fd = (q_eps - q0) / float(eps)

        fd_values[i] = directional_fd
        abs_errors[i] = abs(directional_fd - directional_true)
        rel_errors[i] = abs_errors[i] / denom

    return directional_true, fd_values, abs_errors, rel_errors


def fit_loglog_slope(epsilons, errors, start_idx, end_idx):
    i0 = max(0, start_idx)
    i1 = min(len(epsilons), end_idx)
    x = np.log10(epsilons[i0:i1])
    y = np.log10(errors[i0:i1])
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept


def main():
    parser = argparse.ArgumentParser(
        description="Check z-gradient accuracy for hyperelasticity Taylor approximations"
    )
    parser.add_argument("--beta", type=float, default=1.0, help="Variance weight beta")
    parser.add_argument("--n-tr", type=int, default=10, help="Number of dominant Hessian modes")
    parser.add_argument("--n-mix", type=int, default=11, help="Number of mixture components")
    parser.add_argument("--qoi-type", type=str, default="virtual_work",
                        choices=["all", "stiffness", "point", "virtual_work"])
    parser.add_argument("--penalty", type=float, default=1e-2, help="Control penalty")
    parser.add_argument("--z-value", type=float, default=0.5, help="Constant control test value")
    parser.add_argument("--lx", type=float, default=2.0, help="Beam length in x")
    parser.add_argument("--ly", type=float, default=0.5, help="Beam length in y")
    parser.add_argument("--lz", type=float, default=0.25, help="Beam length in z")
    parser.add_argument("--geometry-dim", type=int, default=2, choices=[2, 3], help="Geometry dimension")
    parser.add_argument("--nx", type=int, default=32, help="Mesh cells in x")
    parser.add_argument("--ny", type=int, default=8, help="Mesh cells in y")
    parser.add_argument("--nz", type=int, default=8, help="Mesh cells in z")
    parser.add_argument("--eps-min", type=float, default=1e-4, help="Minimum epsilon")
    parser.add_argument("--eps-max", type=float, default=1e1, help="Maximum epsilon")
    parser.add_argument("--n-eps", type=int, default=14, help="Number of epsilon values")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dz")
    parser.add_argument("--fit-start", type=int, default=2, help="Fit window start index")
    parser.add_argument("--fit-end", type=int, default=10, help="Fit window end index (exclusive)")
    parser.add_argument("--save-dir", type=str, default="results_z_gradient_check", help="Output directory")
    parser.add_argument(
        "--out",
        type=str,
        default="hyperelasticity_taylor_z_gradient_fd_error.png",
        help="Output figure path",
    )
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Verbose Taylor output")
    args = parser.parse_args()

    if args.eps_min <= 0.0 or args.eps_max <= 0.0:
        raise ValueError("eps-min and eps-max must be positive")
    if args.eps_min >= args.eps_max:
        raise ValueError("eps-min must be smaller than eps-max")
    if args.n_eps < 3:
        raise ValueError("n-eps must be >= 3")

    os.makedirs(args.save_dir, exist_ok=True)
    out = args.out if os.path.isabs(args.out) else os.path.join(args.save_dir, args.out)
    epsilons = np.logspace(np.log10(args.eps_min), np.log10(args.eps_max), args.n_eps)

    control_model, prior, penalty = setup_problem(args)

    model_specs = [
        (
            "linear",
            "Linear Taylor",
            "o-",
            TaylorLinearControlCostFunctional(
                control_model,
                prior,
                penalty,
                {"beta": args.beta, "correction": False, "N_mc": 0, "verbose": args.verbose},
            ),
        ),
        (
            "quadratic",
            "Quadratic Taylor",
            "d-",
            TaylorQuadraticControlCostFunctional(
                control_model,
                prior,
                penalty,
                {"beta": args.beta, "N_tr": args.n_tr, "correction": False, "N_mc": 0, "verbose": args.verbose},
            ),
        ),
        (
            "mixture-linear-kle",
            f"Mixture Linear (N_mix={args.n_mix}, KLE)",
            "^-",
            TaylorMixtureLinearControlCostFunctional(
                control_model,
                prior,
                penalty,
                {"beta": args.beta, "N_mix": args.n_mix, "direction": "kle", "verbose": args.verbose},
                comm_sampler=MPI.COMM_WORLD,
            ),
        ),
        (
            "mixture-linear-hep",
            f"Mixture Linear (N_mix={args.n_mix}, HEP)",
            "s-",
            TaylorMixtureLinearControlCostFunctional(
                control_model,
                prior,
                penalty,
                {"beta": args.beta, "N_mix": args.n_mix, "direction": "hep", "verbose": args.verbose},
                comm_sampler=MPI.COMM_WORLD,
            ),
        ),
        (
            "mixture-quadratic-kle",
            f"Mixture Quadratic (N_mix={args.n_mix}, KLE)",
            "x-",
            TaylorMixtureQuadraticControlCostFunctional(
                control_model,
                prior,
                penalty,
                {"beta": args.beta, "N_mix": args.n_mix, "direction": "kle", "N_tr": args.n_tr, "verbose": args.verbose},
                comm_sampler=MPI.COMM_WORLD,
            ),
        ),
        (
            "mixture-quadratic-hep",
            f"Mixture Quadratic (N_mix={args.n_mix}, HEP)",
            "*-",
            TaylorMixtureQuadraticControlCostFunctional(
                control_model,
                prior,
                penalty,
                {"beta": args.beta, "N_mix": args.n_mix, "direction": "hep", "N_tr": args.n_tr, "verbose": args.verbose},
                comm_sampler=MPI.COMM_WORLD,
            ),
        ),
    ]

    np.random.seed(args.seed)
    z_probe = control_model.generate_vector(CONTROL)
    z_local = np.full(z_probe.local_size(), args.z_value)
    dz_local = np.random.randn(z_probe.local_size())
    dz_norm = np.sqrt(np.dot(dz_local, dz_local))
    if dz_norm <= 0.0:
        raise RuntimeError("Random direction has zero norm")
    dz_local /= dz_norm

    results = []
    for key, label, style, cost in model_specs:
        z = cost.generate_vector(CONTROL)
        z.set_local(np.array(z_local, copy=True))
        z.apply("")

        dz = cost.generate_vector(CONTROL)
        dz.set_local(np.array(dz_local, copy=True))
        dz.apply("")

        true_val, fd_vals, abs_err, rel_err = sweep_fd_error(cost, z, dz, epsilons)
        slope, _ = fit_loglog_slope(epsilons, abs_err, args.fit_start, args.fit_end)
        results.append(
            {
                "key": key,
                "label": label,
                "style": style,
                "true": true_val,
                "fd": fd_vals,
                "abs_err": abs_err,
                "rel_err": rel_err,
                "slope": slope,
            }
        )

    print("Directional derivative check:")
    for result in results:
        print(f"  {result['key']:<22} g.dz = {result['true']:.12e}")

    print("Estimated convergence slope (log-log error vs epsilon):")
    for result in results:
        print(f"  {result['key']:<22} slope ~ {result['slope']:.3f}")

    plt.figure(figsize=(8.0, 5.5))
    for result in results:
        plt.loglog(
            epsilons,
            result["abs_err"],
            result["style"],
            label=f"{result['label']} (slope~{result['slope']:.2f})",
        )

    ref_source = results[0]["abs_err"]
    ref = ref_source[max(1, min(len(ref_source) - 1, args.fit_start))]
    eps_ref = epsilons[max(1, min(len(epsilons) - 1, args.fit_start))]
    plt.loglog(
        epsilons,
        ref * (epsilons / eps_ref),
        "k--",
        linewidth=1.2,
        label="O(epsilon) ref",
    )
    plt.xlabel("epsilon")
    plt.ylabel("|FD - g·dz|")
    plt.title("Hyperelasticity Taylor z-gradient FD error")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    print(f"Saved plot to: {out}")

    out_root, out_ext = os.path.splitext(out)
    if out_ext == "":
        out_ext = ".png"
    rel_out = f"{out_root}_relative{out_ext}"

    plt.figure(figsize=(8.0, 5.5))
    for result in results:
        plt.loglog(
            epsilons,
            result["rel_err"],
            result["style"],
            label=result["label"],
        )
    plt.xlabel("epsilon")
    plt.ylabel("|FD - g·dz| / max(|g·dz|, 1e-14)")
    plt.title("Hyperelasticity Taylor z-gradient relative FD error")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(rel_out, dpi=180)
    print(f"Saved plot to: {rel_out}")


if __name__ == "__main__":
    main()
