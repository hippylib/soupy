"""Directional finite-difference checks for CVaR Taylor-model gradients.

For each approximation, this script computes

    FD(eps) = [Q(z + eps * dz) - Q(z)] / eps

and compares it with the algorithmic directional derivative

    g(z) . dz

where g(z, t) is the returned gradient. For augmented CVaR models we also check
the pure ``t`` direction separately.

The script plots |FD(eps) - g.dz| versus eps on log-log axes. For a forward
difference with accurate gradients, the error should scale approximately as
O(eps) over a suitable epsilon range.

Before running this script, make sure to comment the lines denoted "# Comment the following lines for finite difference gradient test " 
in the Mixture CVaR Taylor expansion scirpts (including mixture_linear_cvar and mixture_quadratic_cvar) to avoid recomputing HEP directions. 
"""

import argparse
import logging
import os
import site
import sys

import dolfin as dl
import matplotlib.pyplot as plt
import numpy as np

logging.getLogger("FFC").setLevel(logging.WARNING)
logging.getLogger("UFL").setLevel(logging.WARNING)
dl.set_log_active(False)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SOUPY_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../.."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../../.."))
HIPPYLIB_ROOT = os.path.join(WORKSPACE_ROOT, "hippylib")

# Avoid accidentally importing user-local wheels that may be ABI-incompatible
# with the active environment.
USER_SITE = site.getusersitepackages()
if USER_SITE in sys.path:
    sys.path.remove(USER_SITE)

if SOUPY_ROOT not in sys.path:
    sys.path.insert(0, SOUPY_ROOT)
if HIPPYLIB_ROOT not in sys.path:
    sys.path.insert(0, HIPPYLIB_ROOT)

if os.environ.get("HIPPYLIB_PATH", "") not in sys.path:
    sys.path.append(os.environ.get("HIPPYLIB_PATH", ""))

import hippylib as hp

from soupy import CONTROL, ControlModel, PDEVariationalControlProblem
from soupy.approximations.taylor import (
    TaylorLinearCVaRControlCostFunctional,
    TaylorMixtureLinearCVaRControlCostFunctional,
    TaylorMixtureQuadraticCVaRControlCostFunctional,
    TaylorQuadraticCVaRControlCostFunctional,
)
from soupy.modeling.controlQoI import L2MisfitControlQoI
from soupy.modeling.augmentedVector import AugmentedVector


def setup_poisson_problem(nx=16, ny=16, gamma=1, delta=5):
    """Set up the compact Poisson control problem used in Taylor tests."""
    mesh = dl.UnitSquareMesh(nx, ny)
    Vh_state = dl.FunctionSpace(mesh, "CG", 1)
    Vh_parameter = dl.FunctionSpace(mesh, "CG", 1)
    Vh_control = dl.FunctionSpace(mesh, "CG", 1)
    Vh = [Vh_state, Vh_parameter, Vh_state, Vh_control]

    def residual(u, m, p, z):
        return dl.exp(m) * dl.inner(dl.grad(u), dl.grad(p)) * dl.dx - z * p * dl.dx

    def boundary(x, on_boundary):
        return on_boundary and (dl.near(x[0], 0.0) or dl.near(x[1], 0.0))

    bc = dl.DirichletBC(Vh_state, dl.Expression("x[1]", degree=1), boundary)
    bc0 = dl.DirichletBC(Vh_state, dl.Constant(0.0), boundary)
    pde = PDEVariationalControlProblem(Vh, residual, [bc], [bc0], is_fwd_linear=True)

    mean_vector = dl.interpolate(dl.Constant(-2.0), Vh_parameter).vector()
    prior = hp.BiLaplacianPrior(
        Vh_parameter,
        gamma,
        delta,
        mean=mean_vector,
        robin_bc=True,
    )

    u_target = dl.Expression(
        "x[1] + sin(k*x[0]) * sin(k*x[1])",
        k=1.5 * np.pi,
        degree=2,
    )
    u_target_fn = dl.interpolate(u_target, Vh_state)
    qoi = L2MisfitControlQoI(Vh, u_target_fn.vector())

    model = ControlModel(pde, qoi)
    return model, prior


def is_augmented_control(v):
    return isinstance(v, AugmentedVector)


def set_test_point(v, z_local, t_value=0.0):
    """Initialize a control test point, including t when augmented."""
    if is_augmented_control(v):
        base = v.get_vector()
        base.set_local(np.array(z_local, copy=True))
        base.apply("")
        v.set_scalar(float(t_value))
    else:
        v.set_local(np.array(z_local, copy=True))
        v.apply("")


def set_z_direction(v, dz_local):
    """Initialize a pure z-direction, with zero t-component if augmented."""
    if is_augmented_control(v):
        base = v.get_vector()
        base.set_local(np.array(dz_local, copy=True))
        base.apply("")
        v.set_scalar(0.0)
    else:
        v.set_local(np.array(dz_local, copy=True))
        v.apply("")


def set_t_direction(v, dt_value=1.0):
    """Initialize a pure t-direction for augmented controls."""
    if not is_augmented_control(v):
        raise RuntimeError("Pure t-direction requires an augmented control vector.")
    base = v.get_vector()
    base.zero()
    base.apply("")
    v.set_scalar(float(dt_value))


def sweep_fd_error(cost, point, direction, epsilons):
    """Compute directional FD errors against the algorithmic gradient."""
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
    """Fit slope of log10(error) vs log10(epsilon) on a selected index window."""
    i0 = max(0, start_idx)
    i1 = min(len(epsilons), end_idx)
    x = np.log10(epsilons[i0:i1])
    y = np.log10(errors[i0:i1])
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept


def main():
    parser = argparse.ArgumentParser(
        description="Check z-gradient accuracy for CVaR Taylor approximations"
    )
    parser.add_argument("--beta", type=float, default=0.95, help="CVaR risk level alpha")
    parser.add_argument(
        "--n-mix",
        type=int,
        default=5,
        help="N_mix for mixture Taylor CVaR models",
    )
    parser.add_argument(
        "--n-tr",
        type=int,
        default=10,
        help="Number of dominant Hessian modes for quadratic CVaR models",
    )
    parser.add_argument(
        "--quad-n-mc",
        type=int,
        default=200,
        help="Number of surrogate samples for quadratic CVaR models",
    )
    parser.add_argument("--nx", type=int, default=16, help="Mesh cells in x")
    parser.add_argument("--ny", type=int, default=16, help="Mesh cells in y")
    parser.add_argument("--gamma", type=float, default=0.2, help="Prior gamma")
    parser.add_argument("--delta", type=float, default=1.0, help="Prior delta")
    parser.add_argument("--eps-min", type=float, default=1e-3, help="Minimum epsilon")
    parser.add_argument("--eps-max", type=float, default=1e1, help="Maximum epsilon")
    parser.add_argument("--n-eps", type=int, default=16, help="Number of epsilon values")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for direction dz")
    parser.add_argument(
        "--fit-start",
        type=int,
        default=3,
        help="Fit window start index in epsilon grid",
    )
    parser.add_argument(
        "--fit-end",
        type=int,
        default=12,
        help="Fit window end index (exclusive) in epsilon grid",
    )
    parser.add_argument(
        "--z-value",
        type=float,
        default=10.0,
        help="Constant value used to initialize the control vector",
    )
    parser.add_argument(
        "--t-value",
        type=float,
        default=10.0,
        help="Constant value used to initialize the auxiliary scalar t",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="cvar_z_gradient_fd_error.png",
        help="Output figure path",
    )
    args = parser.parse_args()

    if args.eps_min <= 0.0 or args.eps_max <= 0.0:
        raise ValueError("eps-min and eps-max must be positive")
    if args.eps_min >= args.eps_max:
        raise ValueError("eps-min must be smaller than eps-max")
    if args.n_eps < 3:
        raise ValueError("n-eps must be >= 3")

    epsilons = np.logspace(np.log10(args.eps_min), np.log10(args.eps_max), args.n_eps)

    model, prior = setup_poisson_problem(
        nx=args.nx,
        ny=args.ny,
        gamma=args.gamma,
        delta=args.delta,
    )

    model_specs = [
        (
            "linear",
            "Linear CVaR",
            "o-",
            TaylorLinearCVaRControlCostFunctional(
                model,
                prior,
                None,
                {"beta": args.beta},
            ),
        ),
        (
            "quadratic",
            "Quadratic CVaR",
            "d-",
            TaylorQuadraticCVaRControlCostFunctional(
                model,
                prior,
                None,
                {
                    "beta": args.beta,
                    "N_tr": args.n_tr,
                    "N_mc": args.quad_n_mc,
                },
            ),
        ),
        (
            "mixture-linear-kle",
            f"Mixture Linear CVaR (N_mix={args.n_mix}, kle)",
            "^-",
            TaylorMixtureLinearCVaRControlCostFunctional(
                model,
                prior,
                None,
                {
                    "beta": args.beta,
                    "N_mix": args.n_mix,
                    "direction": "kle",
                },
            ),
        ),
        (
            "mixture-linear-hep",
            f"Mixture Linear CVaR (N_mix={args.n_mix}, hep)",
            "s-",
            TaylorMixtureLinearCVaRControlCostFunctional(
                model,
                prior,
                None,
                {
                    "beta": args.beta,
                    "N_mix": args.n_mix,
                    "direction": "hep",
                },
            ),
        ),
        (
            "mixture-quadratic-kle",
            f"Mixture Quadratic CVaR (N_mix={args.n_mix}, kle)",
            "x-",
            TaylorMixtureQuadraticCVaRControlCostFunctional(
                model,
                prior,
                None,
                {
                    "beta": args.beta,
                    "N_mix": args.n_mix,
                    "direction": "kle",
                    "N_tr": args.n_tr,
                    "N_mc": args.quad_n_mc,
                },
            ),
        ),
        (
            "mixture-quadratic-hep",
            f"Mixture Quadratic CVaR (N_mix={args.n_mix}, hep)",
            "*-",
            TaylorMixtureQuadraticCVaRControlCostFunctional(
                model,
                prior,
                None,
                {
                    "beta": args.beta,
                    "N_mix": args.n_mix,
                    "direction": "hep",
                    "N_tr": args.n_tr,
                    "N_mc": args.quad_n_mc,
                },
            ),
        ),
    ]

    np.random.seed(args.seed)
    z_probe = model.generate_vector(CONTROL)
    z_local = np.full(z_probe.local_size(), args.z_value)
    t_value = args.t_value
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
        set_test_point(point, z_local, t_value=t_value)
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
    plt.title("z-gradient FD error for CVaR Taylor approximations")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    print(f"Saved plot to: {args.out}")

    out_root, out_ext = os.path.splitext(args.out)
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
    plt.title("z-gradient relative FD error for CVaR Taylor approximations")
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
    plt.title("t-gradient FD error for quadratic CVaR Taylor approximations")
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
    plt.title("t-gradient relative FD error for quadratic CVaR Taylor approximations")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(t_rel_out, dpi=180)
    print(f"Saved plot to: {t_rel_out}")


if __name__ == "__main__":
    main()
