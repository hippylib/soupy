"""Directional finite-difference check for the linear CVaR z-gradient.

This script evaluates the directional derivative of the linear Taylor CVaR
objective in a random control direction `dz` using

    FD(eps) = [J(z + eps * dz) - J(z)] / eps

and compares it against the algorithmic directional derivative

    g(z) . dz

returned by `TaylorLinearCVaRControlCostFunctional`.

The absolute and relative errors are plotted on log-log axes. With a correct
gradient, the forward-difference error should scale approximately like O(eps)
over a suitable epsilon range.
"""

import argparse
import logging
import os
import sys

import dolfin as dl
import matplotlib.pyplot as plt
import numpy as np

logging.getLogger("FFC").setLevel(logging.WARNING)
logging.getLogger("UFL").setLevel(logging.WARNING)
dl.set_log_active(False)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SOUPY_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../.."))
if SOUPY_ROOT not in sys.path:
    sys.path.insert(0, SOUPY_ROOT)

if os.environ.get("HIPPYLIB_PATH", "") not in sys.path:
    sys.path.append(os.environ.get("HIPPYLIB_PATH", ""))

import hippylib as hp

from soupy import CONTROL, ControlModel, PDEVariationalControlProblem
from soupy.approximations.taylor import TaylorLinearCVaRControlCostFunctional
from soupy.modeling.controlQoI import L2MisfitControlQoI


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


def sweep_fd_error(cost, z, dz, epsilons):
    """Compute finite-difference directional derivative errors."""
    g = cost.generate_vector(CONTROL)

    q0 = cost.cost(z, order=1)
    cost.grad(g)
    djdzdz_true = g.inner(dz)

    z_eps = cost.generate_vector(CONTROL)
    fd_values = np.zeros_like(epsilons)
    abs_errors = np.zeros_like(epsilons)
    rel_errors = np.zeros_like(epsilons)
    denom = max(abs(djdzdz_true), 1e-14)

    for i, eps in enumerate(epsilons):
        z_eps.zero()
        z_eps.axpy(1.0, z)
        z_eps.axpy(float(eps), dz)

        q_eps = cost.cost(z_eps, order=0)
        djdzdz_fd = (q_eps - q0) / float(eps)

        fd_values[i] = djdzdz_fd
        abs_errors[i] = abs(djdzdz_fd - djdzdz_true)
        rel_errors[i] = abs_errors[i] / denom

    return djdzdz_true, fd_values, abs_errors, rel_errors


def fit_loglog_slope(epsilons, errors, start_idx, end_idx):
    """Fit slope of log10(error) vs log10(epsilon) over a chosen window."""
    i0 = max(0, start_idx)
    i1 = min(len(epsilons), end_idx)
    x = np.log10(epsilons[i0:i1])
    y = np.log10(errors[i0:i1])
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept


def main():
    parser = argparse.ArgumentParser(
        description="Check z-gradient accuracy for the linear Taylor CVaR approximation"
    )
    parser.add_argument("--beta", type=float, default=0.95, help="CVaR risk level alpha")
    parser.add_argument("--nx", type=int, default=16, help="Mesh cells in x")
    parser.add_argument("--ny", type=int, default=16, help="Mesh cells in y")
    parser.add_argument("--gamma", type=float, default=1.0, help="Prior gamma")
    parser.add_argument("--delta", type=float, default=5.0, help="Prior delta")
    parser.add_argument("--eps-min", type=float, default=1e-4, help="Minimum epsilon")
    parser.add_argument("--eps-max", type=float, default=1e0, help="Maximum epsilon")
    parser.add_argument("--n-eps", type=int, default=16, help="Number of epsilon values")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for direction dz")
    parser.add_argument(
        "--fit-start",
        type=int,
        default=3,
        help="Fit window start index in the epsilon grid",
    )
    parser.add_argument(
        "--fit-end",
        type=int,
        default=12,
        help="Fit window end index (exclusive) in the epsilon grid",
    )
    parser.add_argument(
        "--z-value",
        type=float,
        default=10.0,
        help="Constant value used to initialize the control vector",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="linear_cvar_z_gradient_fd_error.png",
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

    cost = TaylorLinearCVaRControlCostFunctional(
        model,
        prior,
        None,
        {"beta": args.beta},
    )

    z = cost.generate_vector(CONTROL)
    z.set_local(np.full(z.local_size(), args.z_value))
    z.apply("")

    dz = cost.generate_vector(CONTROL)
    np.random.seed(args.seed)
    dz.set_local(np.random.randn(dz.local_size()))
    dz.apply("")

    dz_norm = np.sqrt(dz.inner(dz))
    if dz_norm <= 0.0:
        raise RuntimeError("Random direction has zero norm")
    dz_local = dz.get_local()
    dz_local /= dz_norm
    dz.set_local(dz_local)
    dz.apply("")

    true_derivative, fd_values, abs_errors, rel_errors = sweep_fd_error(
        cost, z, dz, epsilons
    )
    slope, _ = fit_loglog_slope(epsilons, abs_errors, args.fit_start, args.fit_end)

    print("Directional derivative check for linear Taylor CVaR:")
    print(f"  alpha = {args.beta:.6f}")
    print(f"  J(z) directional derivative g.dz = {true_derivative:.12e}")
    print(f"  Estimated log-log slope = {slope:.3f}")
    print(f"  lin_mean = {cost.lin_mean:.12e}")
    print(f"  lin_std  = {cost.lin_std:.12e}")
    print(f"  cvar     = {cost.cvar:.12e}")
    print("")
    print("  epsilon                 FD(eps)                 |FD - g.dz|")
    for eps, fd, err in zip(epsilons, fd_values, abs_errors):
        print(f"  {eps: .6e}   {fd: .12e}   {err: .12e}")

    plt.figure(figsize=(7.0, 5.0))
    plt.loglog(
        epsilons,
        abs_errors,
        "o-",
        label=f"Linear CVaR (slope~{slope:.2f})",
    )

    ref_idx = max(1, min(len(abs_errors) - 1, args.fit_start))
    ref = abs_errors[ref_idx]
    eps_ref = epsilons[ref_idx]
    plt.loglog(
        epsilons,
        ref * (epsilons / eps_ref),
        "k--",
        linewidth=1.2,
        label="O(epsilon) ref",
    )

    plt.xlabel("epsilon")
    plt.ylabel("|FD - g·dz|")
    plt.title("z-gradient FD error for linear Taylor CVaR")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    print(f"Saved plot to: {args.out}")

    out_root, out_ext = os.path.splitext(args.out)
    if out_ext == "":
        out_ext = ".png"
    rel_out = f"{out_root}_relative{out_ext}"

    plt.figure(figsize=(7.0, 5.0))
    plt.loglog(epsilons, rel_errors, "o-", label="Linear CVaR")
    plt.xlabel("epsilon")
    plt.ylabel("|FD - g·dz| / max(|g·dz|, 1e-14)")
    plt.title("z-gradient relative FD error for linear Taylor CVaR")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(rel_out, dpi=180)
    print(f"Saved plot to: {rel_out}")


if __name__ == "__main__":
    main()
