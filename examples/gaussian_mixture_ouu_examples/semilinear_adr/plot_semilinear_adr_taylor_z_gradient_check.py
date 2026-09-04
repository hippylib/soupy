"""Directional finite-difference check for z-gradient of mean-variance Taylor
approximations on semilinear ADR.

This mirrors the style of ``soupy/test/plot_taylor_z_gradient_check.py``, adapted to
the semilinear ADR control problem used by
``driver_semilinear_adr_compare_taylor_models.py``. For each approximation (linear,
quadratic, mixture-linear, and mixture-quadratic), this script computes

    FD(eps) = [Q(z + eps * dz) - Q(z)] / eps

and compares it with the algorithmic directional derivative

    g(z) . dz

where g(z) is the z-gradient returned by the cost functional.

The evaluation point z is the same fixed control used in
``driver_semilinear_adr_compare_models_static.py``: the Gaussian-well coefficients
zᵢ = sin(2π·xᵢ)·sin(2π·yᵢ) at each well center. All model hyperparameters (beta,
N_tr, N_mix) match the defaults used in
``driver_semilinear_adr_compare_taylor_models.py`` (no control penalty is used,
matching that driver).

The script plots |FD(eps) - g.dz| versus eps on log-log axes. For a forward
difference with accurate gradients, the error should scale approximately as O(eps)
over a suitable epsilon range.
"""

import argparse
import logging
import os
import site
import sys

import dolfin as dl
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

logging.getLogger("FFC").setLevel(logging.WARNING)
logging.getLogger("UFL").setLevel(logging.WARNING)
dl.set_log_active(False)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SOUPY_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../../"))

USER_SITE = site.getusersitepackages()
if USER_SITE in sys.path:
    sys.path.remove(USER_SITE)

if _SOUPY_ROOT not in sys.path:
    sys.path.insert(0, _SOUPY_ROOT)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
sys.path.append(os.environ.get("HIPPYLIB_PATH", ""))

import soupy
from soupy import CONTROL, ControlModel, PDEVariationalControlProblem, VariationalControlQoI
from soupy.approximations.taylor import (
    TaylorLinearControlCostFunctional,
    TaylorMixtureLinearControlCostFunctional,
    TaylorMixtureQuadraticControlCostFunctional,
    TaylorQuadraticControlCostFunctional,
)

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

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 14,
})


class SemilinearADRQoIFormHandler:
    """Wrap the ADR QoI form so it is compatible with a control variable."""

    def __init__(self, qoi_varf):
        self.qoi_varf = qoi_varf

    def __call__(self, u, m, z):
        del z
        return self.qoi_varf(u, m)


def build_problem(args):
    """Set up the semilinear ADR control problem exactly as in the Taylor comparison driver."""
    mesh_parameters = MeshParameters()
    pde_parameters = PDEParameters()
    prior_parameters = PriorParameters()
    control_parameters = ControlParameters()
    qoi_type = "mismatch"

    mesh_parameters.nx = args.nx
    mesh_parameters.ny = args.ny

    mesh = setup_mesh(mesh_parameters, MPI.COMM_SELF)
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
    qoi = VariationalControlQoI(Vh, SemilinearADRQoIFormHandler(base_qoi.qoi_varf))
    control_model = ControlModel(pde, qoi)

    return control_model, prior, control_parameters


def target_control_np(control_parameters):
    """The fixed control point used in driver_semilinear_adr_compare_models_static.py."""
    centers = control_well_centers(control_parameters)
    return np.sin(2.0 * np.pi * centers[:, 0]) * np.sin(2.0 * np.pi * centers[:, 1])


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
    """Fit slope of log10(error) vs log10(epsilon) on a selected index window."""
    i0 = max(0, start_idx)
    i1 = min(len(epsilons), end_idx)
    x = np.log10(epsilons[i0:i1])
    y = np.log10(np.maximum(errors[i0:i1], np.finfo(float).tiny))
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept


def save_combined_error_figure(results, epsilons, save_dir, base_name, abs_ylabel, rel_ylabel,
                                title_abs, title_rel, fit_start=3, add_reference=True):
    """Plot |FD - g.dz| (left) and its relative version (right) side by side.

    The legend (including the O(epsilon) reference line) is drawn only once,
    to the right of both subplots.
    """
    fig, (ax_abs, ax_rel) = plt.subplots(1, 2, figsize=(14, 5.5))
    for r in results:
        ax_abs.loglog(epsilons, r["abs_err"], r["style"], label=f"{r['label']} (slope~{r['slope']:.2f})")
        ax_rel.loglog(epsilons, r["rel_err"], r["style"], label=r["label"])

    if add_reference and results:
        ref_source = results[0]["abs_err"]
        idx = max(1, min(len(ref_source) - 1, fit_start))
        ref = ref_source[idx]
        eps_ref = epsilons[idx]
        ax_abs.loglog(epsilons, ref * (epsilons / eps_ref), "k--", linewidth=1.2, label="O(epsilon) ref")

    ax_abs.set_xlabel("epsilon")
    ax_abs.set_ylabel(abs_ylabel)
    ax_abs.set_title(title_abs)
    ax_abs.grid(True, which="both", alpha=0.25)

    ax_rel.set_xlabel("epsilon")
    ax_rel.set_ylabel(rel_ylabel)
    ax_rel.set_title(title_rel)
    ax_rel.grid(True, which="both", alpha=0.25)

    handles, labels = ax_abs.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), borderaxespad=0)
    fig.tight_layout()

    out_path = os.path.join(save_dir, f"{base_name}.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    data_path = os.path.join(save_dir, f"{base_name}_data.npz")
    payload = {
        "epsilons": epsilons,
        "keys": np.array([r["key"] for r in results]),
        "labels": np.array([r["label"] for r in results]),
    }
    for r in results:
        payload[f"abs_err__{r['key']}"] = r["abs_err"]
        payload[f"rel_err__{r['key']}"] = r["rel_err"]
        payload[f"fd__{r['key']}"] = r["fd"]
        payload[f"true__{r['key']}"] = np.array(r["true"])
        payload[f"slope__{r['key']}"] = np.array(r["slope"])
    np.savez(data_path, **payload)
    return out_path, data_path


def main():
    parser = argparse.ArgumentParser(
        description="Check z-gradient accuracy for mean-variance Taylor approximations on semilinear ADR"
    )
    # Mean-variance hyperparameters: match driver_semilinear_adr_compare_taylor_models.py defaults
    parser.add_argument("--beta", type=float, default=1.0, help="Variance weight beta")
    parser.add_argument("--n-tr", type=int, default=50, help="Number of Hessian modes for quadratic models")
    parser.add_argument("--n-mix", type=int, default=11, help="Number of mixture components")

    # Semilinear ADR problem settings: match driver_semilinear_adr_compare_taylor_models.py defaults
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--newton-max-it", type=int, default=50)
    parser.add_argument("--newton-rtol", type=float, default=1e-8)
    parser.add_argument("--newton-atol", type=float, default=1e-10)

    # FD sweep settings
    parser.add_argument("--eps-min", type=float, default=1e-4, help="Minimum epsilon")
    parser.add_argument("--eps-max", type=float, default=1e02, help="Maximum epsilon")
    parser.add_argument("--n-eps", type=int, default=16, help="Number of epsilon values")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for direction dz")
    parser.add_argument("--fit-start", type=int, default=3, help="Fit window start index in epsilon grid")
    parser.add_argument("--fit-end", type=int, default=12, help="Fit window end index (exclusive) in epsilon grid")
    parser.add_argument("--save-dir", type=str, default="results_z_gradient_check", help="Output directory")
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    args = parser.parse_args()

    if args.eps_min <= 0.0 or args.eps_max <= 0.0:
        raise ValueError("eps-min and eps-max must be positive")
    if args.eps_min >= args.eps_max:
        raise ValueError("eps-min must be smaller than eps-max")
    if args.n_eps < 3:
        raise ValueError("n-eps must be >= 3")

    os.makedirs(args.save_dir, exist_ok=True)
    epsilons = np.logspace(np.log10(args.eps_min), np.log10(args.eps_max), args.n_eps)

    control_model, prior, control_parameters = build_problem(args)

    model_specs = [
        (
            "linear",
            "Linear Taylor",
            "o-",
            TaylorLinearControlCostFunctional(
                control_model, prior, None,
                {"beta": args.beta, "correction": False, "N_mc": 0, "verbose": args.verbose},
            ),
        ),
        (
            "quadratic",
            "Quadratic Taylor",
            "d-",
            TaylorQuadraticControlCostFunctional(
                control_model, prior, None,
                {"beta": args.beta, "N_tr": args.n_tr, "correction": False, "N_mc": 0, "verbose": args.verbose},
            ),
        ),
        (
            "mixture-linear-kle",
            f"Mixture Linear (N_mix={args.n_mix}, KLE)",
            "^-",
            TaylorMixtureLinearControlCostFunctional(
                control_model, prior, None,
                {"beta": args.beta, "N_mix": args.n_mix, "direction": "kle", "verbose": args.verbose},
            ),
        ),
        (
            "mixture-linear-hep",
            f"Mixture Linear (N_mix={args.n_mix}, HEP)",
            "s-",
            TaylorMixtureLinearControlCostFunctional(
                control_model, prior, None,
                {"beta": args.beta, "N_mix": args.n_mix, "direction": "hep", "verbose": args.verbose},
            ),
        ),
        (
            "mixture-quadratic-kle",
            f"Mixture Quadratic (N_mix={args.n_mix}, N_tr={args.n_tr}, KLE)",
            "x-",
            TaylorMixtureQuadraticControlCostFunctional(
                control_model, prior, None,
                {"beta": args.beta, "N_mix": args.n_mix, "direction": "kle", "N_tr": args.n_tr,
                 "N_mc": 0, "verbose": args.verbose},
            ),
        ),
        (
            "mixture-quadratic-hep",
            f"Mixture Quadratic (N_mix={args.n_mix}, N_tr={args.n_tr}, HEP)",
            "*-",
            TaylorMixtureQuadraticControlCostFunctional(
                control_model, prior, None,
                {"beta": args.beta, "N_mix": args.n_mix, "direction": "hep", "N_tr": args.n_tr,
                 "N_mc": 0, "verbose": args.verbose},
            ),
        ),
    ]

    # Fixed evaluation point: the target Gaussian-well control used in
    # driver_semilinear_adr_compare_models_static.py.
    z_local = target_control_np(control_parameters)

    np.random.seed(args.seed)
    dz_local = np.random.randn(z_local.size)
    dz_norm = np.sqrt(np.dot(dz_local, dz_local))
    if dz_norm > 0.0:
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
            {"key": key, "label": label, "style": style, "true": true_val,
             "fd": fd_vals, "abs_err": abs_err, "rel_err": rel_err, "slope": slope}
        )

    print("Directional derivative check:")
    for result in results:
        print(f"  {result['key']:<22} g.dz = {result['true']:.12e}")
    print("Estimated convergence slope (log-log error vs epsilon):")
    for result in results:
        print(f"  {result['key']:<22} slope ~ {result['slope']:.3f}")

    out_path, data_path = save_combined_error_figure(
        results,
        epsilons,
        args.save_dir,
        "semilinear_adr_taylor_z_gradient_fd_error",
        abs_ylabel="|FD - g·dz|",
        rel_ylabel="|FD - g·dz| / max(|g·dz|, 1e-14)",
        title_abs="Absolute z-gradient FD error",
        title_rel="Relative z-gradient FD error",
        fit_start=args.fit_start,
    )
    print(f"Saved plot to: {out_path}")
    print(f"Saved data to: {data_path}")


if __name__ == "__main__":
    main()
