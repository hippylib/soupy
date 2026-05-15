"""Compare CVaR model z-gradients against a shared SAA_10000 FD reference.

For each approximation, this script computes the model directional derivative

    g_model(z, t) . dz

and compares it against the forward-difference reference built from the same
"true" objective for every model:

    FD_true(eps) = [J_true(z + eps * dz) - J_true(z)] / eps

where J_true is the SAA_10000 CVaR objective with the same L2 control penalty.
Only the three epsilon levels 1e-3, 1e-4, and 1e-5 are used.
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
from soupy import (
    CONTROL,
    RiskMeasureControlCostFunctional,
    SuperquantileRiskMeasureSAA,
    superquantileRiskMeasureSAASettings,
)
from soupy.approximations.taylor import (
    TaylorLinearCVaRControlCostFunctional,
    TaylorMixtureLinearCVaRControlCostFunctional,
    TaylorMixtureQuadraticCVaRControlCostFunctional,
    TaylorQuadraticCVaRControlCostFunctional,
)
from soupy.modeling.augmentedVector import AugmentedVector

from setupHyperelasticityProblem import hyperelasticity_problem_settings, setup_hyperelasticity_problem


TRUE_OBJECTIVE_SAMPLE_SIZE = 10000
EPSILON_LEVELS = np.array([1e-3, 1e-4, 1e-5], dtype=float)


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


def make_cvar_saa_cost(control_model, prior, penalty, beta, sample_size, seed, comm_sampler, epsilon=1e-4):
    settings = superquantileRiskMeasureSAASettings()
    settings["beta"] = beta
    settings["sample_size"] = sample_size
    settings["seed"] = seed
    settings["epsilon"] = float(epsilon)
    risk = SuperquantileRiskMeasureSAA(control_model, prior, settings=settings, comm_sampler=comm_sampler)
    return RiskMeasureControlCostFunctional(risk, penalty)


def compute_model_directional_derivative(cost, point, direction):
    gradient = cost.generate_vector(CONTROL)
    cost.cost(point, order=1)
    cost.grad(gradient)
    return gradient.inner(direction)


def sweep_true_fd_reference(true_cost, z_local, dz_local, epsilons):
    z0 = true_cost.generate_vector(CONTROL)
    z0.set_local(np.array(z_local, copy=True))
    z0.apply("")
    q0 = float(true_cost.cost(z0, order=0))

    z_eps = true_cost.generate_vector(CONTROL)
    fd_values = np.zeros_like(epsilons)
    for i, eps in enumerate(epsilons):
        z_eps.set_local(z_local + float(eps) * dz_local)
        z_eps.apply("")
        q_eps = float(true_cost.cost(z_eps, order=0))
        fd_values[i] = (q_eps - q0) / float(eps)
    return fd_values


def compute_errors(reference_fd, model_directional_derivative):
    abs_errors = np.abs(reference_fd - model_directional_derivative)
    rel_errors = abs_errors / np.maximum(np.abs(reference_fd), 1e-14)
    return abs_errors, rel_errors


def main():
    parser = argparse.ArgumentParser(
        description="Check hyperelasticity CVaR model z-gradients against a shared SAA_10000 FD reference"
    )
    parser.add_argument("--beta", type=float, default=0.95, help="CVaR confidence level")
    parser.add_argument("--n-mix", type=int, default=11, help="Number of mixture components")
    parser.add_argument("--n-tr", type=int, default=50, help="Number of dominant Hessian modes")
    parser.add_argument("--quad-n-mc", type=int, default=1000, help="MC samples for quadratic CVaR surrogates")
    parser.add_argument(
        "--qoi-type",
        type=str,
        default="virtual_work",
        choices=["all", "stiffness", "point", "virtual_work"],
    )
    parser.add_argument("--penalty", type=float, default=1e-1, help="Control penalty")
    parser.add_argument("--z-value", type=float, default=0.5, help="Constant control test value")
    parser.add_argument("--t-value", type=float, default=0.0, help="Initial auxiliary scalar t")
    parser.add_argument("--lx", type=float, default=2.0, help="Beam length in x")
    parser.add_argument("--ly", type=float, default=0.5, help="Beam length in y")
    parser.add_argument("--lz", type=float, default=0.25, help="Beam length in z")
    parser.add_argument("--geometry-dim", type=int, default=2, choices=[2, 3], help="Geometry dimension")
    parser.add_argument("--nx", type=int, default=32, help="Mesh cells in x")
    parser.add_argument("--ny", type=int, default=8, help="Mesh cells in y")
    parser.add_argument("--nz", type=int, default=8, help="Mesh cells in z")
    parser.add_argument("--saa-seed", type=int, default=1, help="Seed used by the shared SAA_10000 true objective")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dz")
    parser.add_argument("--save-dir", type=str, default="results_z_gradient_check", help="Output directory")
    parser.add_argument(
        "--out",
        type=str,
        default="hyperelasticity_cvar_z_gradient_fd_error.png",
        help="Output figure path",
    )
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Verbose CVaR output")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    out = args.out if os.path.isabs(args.out) else os.path.join(args.save_dir, args.out)
    epsilons = np.array(EPSILON_LEVELS, copy=True)

    control_model, prior, penalty = setup_problem(args)
    true_cost = make_cvar_saa_cost(
        control_model,
        prior,
        penalty,
        beta=args.beta,
        sample_size=TRUE_OBJECTIVE_SAMPLE_SIZE,
        seed=args.saa_seed,
        comm_sampler=MPI.COMM_WORLD,
        epsilon=1e-4,
    )

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
                {
                    "beta": args.beta,
                    "N_mix": args.n_mix,
                    "direction": "kle",
                    "N_tr": args.n_tr,
                    "N_mc": args.quad_n_mc,
                    "verbose": args.verbose,
                },
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
                {
                    "beta": args.beta,
                    "N_mix": args.n_mix,
                    "direction": "hep",
                    "N_tr": args.n_tr,
                    "N_mc": args.quad_n_mc,
                    "verbose": args.verbose,
                },
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

    true_fd_values = sweep_true_fd_reference(true_cost, z_local, dz_local, epsilons)

    z_results = []
    for key, label, style, cost in model_specs:
        point = cost.generate_vector(CONTROL)
        z_dir = cost.generate_vector(CONTROL)
        set_test_point(point, z_local, t_value=args.t_value)
        set_z_direction(z_dir, dz_local)

        model_val = compute_model_directional_derivative(cost, point, z_dir)
        abs_err, rel_err = compute_errors(true_fd_values, model_val)
        z_results.append(
            {
                "key": key,
                "label": label,
                "style": style,
                "model": model_val,
                "fd": np.array(true_fd_values, copy=True),
                "abs_err": abs_err,
                "rel_err": rel_err,
            }
        )

    print(f"Shared SAA_{TRUE_OBJECTIVE_SAMPLE_SIZE} z-direction FD reference:")
    for eps, fd_value in zip(epsilons, true_fd_values):
        print(f"  eps = {eps:.0e}  FD_true = {fd_value:.12e}")

    print("Model z-directional derivatives:")
    for result in z_results:
        print(f"  {result['key']:<22} g_model.dz = {result['model']:.12e}")

    plt.figure(figsize=(8.0, 5.5))
    for result in z_results:
        plt.loglog(
            epsilons,
            result["abs_err"],
            result["style"],
            label=result["label"],
        )
    plt.xlabel("epsilon")
    plt.ylabel(r"$|FD_{\mathrm{true}} - g_{\mathrm{model}} \cdot dz|$")
    plt.title(f"Hyperelasticity CVaR z-gradient error vs SAA_{TRUE_OBJECTIVE_SAMPLE_SIZE}")
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
    plt.ylabel(r"$|FD_{\mathrm{true}} - g_{\mathrm{model}} \cdot dz| / \max(|FD_{\mathrm{true}}|, 10^{-14})$")
    plt.title(f"Hyperelasticity CVaR relative z-gradient error vs SAA_{TRUE_OBJECTIVE_SAMPLE_SIZE}")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(rel_out, dpi=180)
    print(f"Saved plot to: {rel_out}")


if __name__ == "__main__":
    main()
