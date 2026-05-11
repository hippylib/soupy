"""Directional finite-difference checks for hyperelasticity CVaR Taylor-model gradients.

For each approximation, this script computes

    FD(eps) = [Q(z + eps * dz) - Q(z)] / eps

and compares it with the algorithmic directional derivative

    g(z, t) . dz

where g(z, t) is the returned gradient. For augmented CVaR models we also
check the pure ``t`` direction separately.

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
    TaylorLinearCVaRControlCostFunctional,
    TaylorMixtureLinearCVaRControlCostFunctional,
    TaylorMixtureQuadraticCVaRControlCostFunctional,
    TaylorQuadraticCVaRControlCostFunctional,
)
from soupy.modeling.augmentedVector import AugmentedVector

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


def is_augmented_control(vector):
    return isinstance(vector, AugmentedVector)


def set_test_point(vector, z_local, t_value=0.0):
    if is_augmented_control(vector):
        base = vector.get_vector()
        base.set_local(np.array(z_local, copy=True))
        base.apply("")
        vector.set_scalar(float(t_value))
    else:
        vector.set_local(np.array(z_local, copy=True))
        vector.apply("")


def set_z_direction(vector, dz_local):
    if is_augmented_control(vector):
        base = vector.get_vector()
        base.set_local(np.array(dz_local, copy=True))
        base.apply("")
        vector.set_scalar(0.0)
    else:
        vector.set_local(np.array(dz_local, copy=True))
        vector.apply("")


def set_t_direction(vector, dt_value=1.0):
    if not is_augmented_control(vector):
        raise RuntimeError("Pure t-direction requires an augmented control vector.")
    base = vector.get_vector()
    base.zero()
    base.apply("")
    vector.set_scalar(float(dt_value))


def sweep_fd_error(cost, point, direction, epsilons):
    """Compare algorithmic and FD directional derivatives along direction."""
    g = cost.generate_vector(CONTROL)

    q0 = cost.cost(point, order=1, FD_gradient_check=True)
    cost.grad(g)
    directional_true = g.inner(direction)

    point_eps = cost.generate_vector(CONTROL)
    fd_values = np.zeros_like(epsilons)
    abs_errors = np.zeros_like(epsilons)
    rel_errors = np.zeros_like(epsilons)
    denom = max(abs(directional_true), 1e-14)

    for i, eps in enumerate(epsilons):
        point_eps.zero()
        point_eps.axpy(1.0, point)
        point_eps.axpy(float(eps), direction)

        q_eps = cost.cost(point_eps, order=0, FD_gradient_check=True)
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
        description="Check z-gradient accuracy for hyperelasticity CVaR Taylor approximations"
    )
    parser.add_argument("--beta", type=float, default=0.95, help="CVaR confidence level")
    parser.add_argument("--n-mix", type=int, default=11, help="Number of mixture components")
    parser.add_argument("--n-tr", type=int, default=10, help="Number of dominant Hessian modes")
    parser.add_argument("--quad-n-mc", type=int, default=200, help="MC samples for quadratic CVaR surrogates")
    parser.add_argument("--qoi-type", type=str, default="virtual_work",
                        choices=["all", "stiffness", "point", "virtual_work"])
    parser.add_argument("--penalty", type=float, default=1e-2, help="Control penalty")
    parser.add_argument("--z-value", type=float, default=0.5, help="Constant control test value")
    parser.add_argument("--t-value", type=float, default=0.0, help="Initial auxiliary scalar t")
    parser.add_argument("--lx", type=float, default=2.0, help="Beam length in x")
    parser.add_argument("--ly", type=float, default=0.5, help="Beam length in y")
    parser.add_argument("--lz", type=float, default=0.25, help="Beam length in z")
    parser.add_argument("--geometry-dim", type=int, default=2, choices=[2, 3], help="Geometry dimension")
    parser.add_argument("--nx", type=int, default=32, help="Mesh cells in x")
    parser.add_argument("--ny", type=int, default=8, help="Mesh cells in y")
    parser.add_argument("--nz", type=int, default=8, help="Mesh cells in z")
    parser.add_argument("--eps-min", type=float, default=1e-6, help="Minimum epsilon")
    parser.add_argument("--eps-max", type=float, default=1e1, help="Maximum epsilon")
    parser.add_argument("--n-eps", type=int, default=16, help="Number of epsilon values")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dz")
    parser.add_argument("--fit-start", type=int, default=3, help="Fit window start index")
    parser.add_argument("--fit-end", type=int, default=12, help="Fit window end index (exclusive)")
    parser.add_argument("--save-dir", type=str, default="results_z_gradient_check", help="Output directory")
    parser.add_argument(
        "--out",
        type=str,
        default="hyperelasticity_cvar_z_gradient_fd_error.png",
        help="Output figure path",
    )
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Verbose CVaR output")
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
            "Linear CVaR",
            "o-",
            TaylorLinearCVaRControlCostFunctional(
                control_model,
                prior,
                penalty,
                {"beta": args.beta, "verbose": args.verbose},
            ),
        ),
        (
            "quadratic",
            "Quadratic CVaR",
            "d-",
            TaylorQuadraticCVaRControlCostFunctional(
                control_model,
                prior,
                penalty,
                {"beta": args.beta, "N_tr": args.n_tr, "N_mc": args.quad_n_mc, "verbose": args.verbose},
            ),
        ),
        (
            "mixture-linear-kle",
            f"Mixture Linear CVaR (N_mix={args.n_mix}, KLE)",
            "^-",
            TaylorMixtureLinearCVaRControlCostFunctional(
                control_model,
                prior,
                penalty,
                {"beta": args.beta, "N_mix": args.n_mix, "direction": "kle", "verbose": args.verbose},
                comm_sampler=MPI.COMM_WORLD,
            ),
        ),
        (
            "mixture-linear-hep",
            f"Mixture Linear CVaR (N_mix={args.n_mix}, HEP)",
            "s-",
            TaylorMixtureLinearCVaRControlCostFunctional(
                control_model,
                prior,
                penalty,
                {"beta": args.beta, "N_mix": args.n_mix, "direction": "hep", "verbose": args.verbose},
                comm_sampler=MPI.COMM_WORLD,
            ),
        ),
        (
            "mixture-quadratic-kle",
            f"Mixture Quadratic CVaR (N_mix={args.n_mix}, KLE)",
            "x-",
            TaylorMixtureQuadraticCVaRControlCostFunctional(
                control_model,
                prior,
                penalty,
                {"beta": args.beta, "N_mix": args.n_mix, "direction": "kle", "N_tr": args.n_tr, "N_mc": args.quad_n_mc, "verbose": args.verbose},
                comm_sampler=MPI.COMM_WORLD,
            ),
        ),
        (
            "mixture-quadratic-hep",
            f"Mixture Quadratic CVaR (N_mix={args.n_mix}, HEP)",
            "*-",
            TaylorMixtureQuadraticCVaRControlCostFunctional(
                control_model,
                prior,
                penalty,
                {"beta": args.beta, "N_mix": args.n_mix, "direction": "hep", "N_tr": args.n_tr, "N_mc": args.quad_n_mc, "verbose": args.verbose},
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

    z_results = []
    t_results = []
    for key, label, style, cost in model_specs:
        point = cost.generate_vector(CONTROL)
        z_dir = cost.generate_vector(CONTROL)
        set_test_point(point, z_local, t_value=args.t_value)
        set_z_direction(z_dir, dz_local)

        true_val, fd_vals, abs_err, rel_err = sweep_fd_error(cost, point, z_dir, epsilons)
        slope, _ = fit_loglog_slope(epsilons, abs_err, args.fit_start, args.fit_end)
        z_results.append(
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

        if key in {"quadratic", "mixture-quadratic-kle", "mixture-quadratic-hep"}:
            if not is_augmented_control(point):
                raise RuntimeError(f"{key} is expected to use an augmented control vector.")
            t_dir = cost.generate_vector(CONTROL)
            set_t_direction(t_dir, dt_value=1.0)
            t_true, t_fd, t_abs_err, t_rel_err = sweep_fd_error(cost, point, t_dir, epsilons)
            t_slope, _ = fit_loglog_slope(epsilons, t_abs_err, args.fit_start, args.fit_end)
            t_results.append(
                {
                    "key": key,
                    "label": label,
                    "style": style,
                    "true": t_true,
                    "fd": t_fd,
                    "abs_err": t_abs_err,
                    "rel_err": t_rel_err,
                    "slope": t_slope,
                }
            )

    print("z-directional derivative check:")
    for result in z_results:
        print(f"  {result['key']:<22} g.dz = {result['true']:.12e}")

    print("Estimated z-gradient convergence slope (log-log error vs epsilon):")
    for result in z_results:
        print(f"  {result['key']:<22} slope ~ {result['slope']:.3f}")

    print("t-directional derivative check:")
    for result in t_results:
        print(f"  {result['key']:<22} g.dt = {result['true']:.12e}")

    print("Estimated t-gradient convergence slope (log-log error vs epsilon):")
    for result in t_results:
        print(f"  {result['key']:<22} slope ~ {result['slope']:.3f}")

    plt.figure(figsize=(8.0, 5.5))
    for result in z_results:
        plt.loglog(
            epsilons,
            result["abs_err"],
            result["style"],
            label=f"{result['label']} (slope~{result['slope']:.2f})",
        )

    ref_source = z_results[0]["abs_err"]
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
    plt.title("Hyperelasticity CVaR z-gradient FD error")
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
    for result in z_results:
        plt.loglog(
            epsilons,
            result["rel_err"],
            result["style"],
            label=result["label"],
        )
    plt.xlabel("epsilon")
    plt.ylabel("|FD - g·dz| / max(|g·dz|, 1e-14)")
    plt.title("Hyperelasticity CVaR z-gradient relative FD error")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(rel_out, dpi=180)
    print(f"Saved plot to: {rel_out}")

    t_out = f"{out_root}_t_gradient{out_ext}"
    plt.figure(figsize=(8.0, 5.5))
    for result in t_results:
        plt.loglog(
            epsilons,
            result["abs_err"],
            result["style"],
            label=f"{result['label']} (slope~{result['slope']:.2f})",
        )
    if t_results:
        t_ref_source = t_results[0]["abs_err"]
        t_ref = t_ref_source[max(1, min(len(t_ref_source) - 1, args.fit_start))]
        plt.loglog(
            epsilons,
            t_ref * (epsilons / eps_ref),
            "k--",
            linewidth=1.2,
            label="O(epsilon) ref",
        )
    plt.xlabel("epsilon")
    plt.ylabel("|FD - g·dt|")
    plt.title("Hyperelasticity CVaR t-gradient FD error")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(t_out, dpi=180)
    print(f"Saved plot to: {t_out}")

    t_rel_out = f"{out_root}_t_gradient_relative{out_ext}"
    plt.figure(figsize=(8.0, 5.5))
    for result in t_results:
        plt.loglog(
            epsilons,
            result["rel_err"],
            result["style"],
            label=result["label"],
        )
    plt.xlabel("epsilon")
    plt.ylabel("|FD - g·dt| / max(|g·dt|, 1e-14)")
    plt.title("Hyperelasticity CVaR t-gradient relative FD error")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(t_rel_out, dpi=180)
    print(f"Saved plot to: {t_rel_out}")


if __name__ == "__main__":
    main()
