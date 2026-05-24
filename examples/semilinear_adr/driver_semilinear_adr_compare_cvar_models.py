"""Compare Taylor and SAA CVaR models on semilinear ADR control.

This mirrors the semilinear ADR mean-variance comparison driver, but uses the
superquantile/CVaR objective. The objective includes an L2 control penalty.

Models compared:
- linear
- quadratic
- mixture_linear_kle
- mixture_linear_hep
- mixture_quadratic_kle
- mixture_quadratic_hep
- saa_1
- saa_10
- saa_39
- saa_100
- saa_200
- saa_500
- saa_1000
- saa_10000

Reference truth:
- Exact empirical CVaR from a large fixed QoI sample set
- Plus the same L2 control penalty used by each model
- Evaluated at the initial control and each model optimum

Quadratic CVaR Taylor models use Monte Carlo samples to estimate the surrogate
CVaR; the default is 10000.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List

# Configure macOS compiler BEFORE importing dolfin or soupy
_soupy_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, os.path.join(_soupy_root, "soupy", "utils"))
try:
    from macos_config import configure_macos_compiler, configure_dolfin_form_compiler

    configure_macos_compiler()
except ImportError:
    configure_dolfin_form_compiler = None
sys.path.pop(0)

sys.path.append(os.environ.get("HIPPYLIB_PATH", ""))
sys.path.append(_soupy_root)

import dolfin as dl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import scipy.optimize
from mpi4py import MPI

import hippylib as hp
import soupy
from semilinear_adr_problem import (
    ControlParameters,
    MeshParameters,
    PDEParameters,
    PriorParameters,
    ControlledSemilinearADRWellVarfHandler,
    control_coefficients_to_function,
    setup_control_function_space,
    setup_mesh,
    setup_prior,
    setup_qoi,
)
from soupy import (
    ControlModel,
    PDEVariationalControlProblem,
    RiskMeasureControlCostFunctional,
    SuperquantileRiskMeasureSAA,
    VariationalControlQoI,
    superquantileRiskMeasureSAASettings,
)
from soupy.approximations.taylor import (
    TaylorLinearCVaRControlCostFunctional,
    TaylorMixtureLinearCVaRControlCostFunctional,
    TaylorMixtureQuadraticCVaRControlCostFunctional,
    TaylorQuadraticCVaRControlCostFunctional,
)


dl.set_log_active(False)
if configure_dolfin_form_compiler is not None:
    try:
        configure_dolfin_form_compiler(dl)
    except Exception:
        pass


MODEL_ORDER = [
    "linear",
    "quadratic",
    "mixture_linear_kle",
    "mixture_linear_hep",
    "mixture_quadratic_kle",
    "mixture_quadratic_hep",
    "saa_1",
    "saa_10",
    "saa_39",
    "saa_100",
    "saa_200",
    "saa_500",
    "saa_1000",
    "saa_10000",
]

MODEL_COLORS = {
    "linear": "tab:blue",
    "quadratic": "tab:orange",
    "mixture_linear_kle": "tab:green",
    "mixture_linear_hep": "tab:olive",
    "mixture_quadratic_kle": "tab:red",
    "mixture_quadratic_hep": "tab:brown",
    "saa_1": "black",
    "saa_10": "#17becf",
    "saa_39": "#7f7f7f",
    "saa_100": "#9467bd",
    "saa_200": "#e377c2",
    "saa_500": "#8c564b",
    "saa_1000": "#bcbd22",
    "saa_10000": "#1f9e89",
}

EXPLICIT_CVAR_CONTINUATION_LEVELS = [1e-4]


class TeeStream:
    def __init__(self, terminal_stream, file_stream):
        self._terminal = terminal_stream
        self._file = file_stream

    def write(self, data):
        self._terminal.write(data)
        self._file.write(data)

    def flush(self):
        self._terminal.flush()
        self._file.flush()


@dataclass
class IterRecord:
    iteration: int
    model_cost: float
    residual: float
    epsilon: float
    iter_time_sec: float
    rss_mb: float
    peak_rss_mb: float
    cost_rss_before_mb: float
    cost_rss_after_mb: float
    grad_rss_before_mb: float
    grad_rss_after_mb: float


class ControlledSemilinearADRVarfHandler:
    """Legacy field-control wrapper retained for backward compatibility."""

    def __init__(self, base_varf_handler):
        self.base_varf_handler = base_varf_handler

    def __call__(self, u, m, p, z):
        return self.base_varf_handler(u, m, p) - z * p * dl.dx


class SemilinearADRQoIFormHandler:
    """Wrap the ADR QoI form so it is compatible with a control variable."""

    def __init__(self, qoi_varf):
        self.qoi_varf = qoi_varf

    def __call__(self, u, m, z):
        del z
        return self.qoi_varf(u, m)


def _rss_from_proc_status_mb() -> float:
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    match = re.search(r"(\d+)", line)
                    if match:
                        return float(match.group(1)) / 1024.0
    except OSError:
        pass
    return float("nan")


def get_current_rss_mb() -> float:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 ** 2)
    except Exception:
        return _rss_from_proc_status_mb()


def get_peak_rss_mb() -> float:
    try:
        import resource

        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return float(peak_kb) / (1024.0 ** 2)
        return float(peak_kb) / 1024.0
    except Exception:
        return float("nan")


def projected_gradient_inf_norm(x, grad, bounds):
    if grad is None:
        return float("nan")
    x = np.asarray(x, dtype=float)
    projected_grad = np.asarray(grad, dtype=float).copy()
    if bounds is not None:
        lb = np.asarray(bounds.lb, dtype=float)
        ub = np.asarray(bounds.ub, dtype=float)
        at_lb = np.isfinite(lb) & np.isclose(x, lb) & (projected_grad > 0.0)
        at_ub = np.isfinite(ub) & np.isclose(x, ub) & (projected_grad < 0.0)
        projected_grad[at_lb | at_ub] = 0.0
    return float(np.linalg.norm(projected_grad, ord=np.inf))


class ScipyObjectiveWithHistory:
    def __init__(self, cost_functional):
        self.cost_functional = cost_functional
        self._z = cost_functional.generate_vector(soupy.CONTROL)
        self._g = cost_functional.generate_vector(soupy.CONTROL)
        self.latest_cost = np.nan
        self.latest_grad_norm = np.nan
        self.latest_gradient = None
        self.latest_cost_rss_before_mb = np.nan
        self.latest_cost_rss_after_mb = np.nan
        self.latest_grad_rss_before_mb = np.nan
        self.latest_grad_rss_after_mb = np.nan
        self.n_func = 0
        self.n_grad = 0

    def current_smoothing_epsilon(self):
        if (
            hasattr(self.cost_functional, "risk_measure")
            and hasattr(self.cost_functional.risk_measure, "smoothplus")
            and hasattr(self.cost_functional.risk_measure.smoothplus, "epsilon")
        ):
            return float(self.cost_functional.risk_measure.smoothplus.epsilon)
        if hasattr(self.cost_functional, "_legacy") and hasattr(self.cost_functional._legacy, "epsilon"):
            return float(self.cost_functional._legacy.epsilon)
        if hasattr(self.cost_functional, "epsilon"):
            return float(self.cost_functional.epsilon)
        return float("nan")

    def function(self):
        def f(z_np):
            self._z.set_local(z_np)
            self._z.apply("")
            self.latest_cost_rss_before_mb = get_current_rss_mb()
            self.latest_cost = float(self.cost_functional.cost(self._z, order=0))
            self.latest_cost_rss_after_mb = get_current_rss_mb()
            self.n_func += 1
            return self.latest_cost

        return f

    def jac(self):
        def g(z_np):
            self._z.set_local(z_np)
            self._z.apply("")
            self.latest_grad_rss_before_mb = get_current_rss_mb()
            self.cost_functional.cost(self._z, order=1)
            self.latest_grad_norm = float(self.cost_functional.grad(self._g))
            self.latest_gradient = np.array(self._g.get_local(), copy=True)
            self.latest_grad_rss_after_mb = get_current_rss_mb()
            self.n_grad += 1
            return self.latest_gradient

        return g


def setup_problem(args, comm_mesh):
    mesh_parameters = MeshParameters()
    pde_parameters = PDEParameters()
    prior_parameters = PriorParameters()
    control_parameters = ControlParameters()
    qoi_type = "mismatch"

    if args.nx is not None:
        mesh_parameters.nx = args.nx
    if args.ny is not None:
        mesh_parameters.ny = args.ny

    mesh = setup_mesh(mesh_parameters, comm_mesh)
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
    penalty = soupy.L2Penalization(Vh, args.penalty)

    return {
        "mesh_parameters": mesh_parameters,
        "pde_parameters": pde_parameters,
        "prior_parameters": prior_parameters,
        "control_parameters": control_parameters,
        "qoi_type": qoi_type,
        "mesh": mesh,
        "Vh": Vh,
        "control_model": control_model,
        "prior": prior,
        "penalty": penalty,
    }


def is_augmented_vector(vector) -> bool:
    return hasattr(vector, "get_vector") and hasattr(vector, "get_scalar")


def set_control_part(vector, control_np, scalar=0.0):
    if is_augmented_vector(vector):
        vector.get_vector().set_local(control_np)
        vector.get_vector().apply("")
        vector.set_scalar(float(scalar))
    else:
        vector.set_local(control_np)
        vector.apply("")


def split_control(vector):
    if is_augmented_vector(vector):
        return np.array(vector.get_vector().get_local(), copy=True), float(vector.get_scalar())
    return np.array(vector.get_local(), copy=True), None


def np_to_control(cost_functional, z_np):
    z = cost_functional.generate_vector(soupy.CONTROL)
    z.set_local(z_np)
    z.apply("")
    return z


def exact_cvar_from_samples(samples, beta):
    quantile = float(np.percentile(samples, beta * 100.0))
    cvar = quantile + float(np.mean(np.maximum(samples - quantile, 0.0))) / (1.0 - beta)
    return quantile, cvar


def initial_model_point_at_control(cost_functional, control_np, scalar=0.0):
    point = cost_functional.generate_vector(soupy.CONTROL)
    set_control_part(point, control_np, scalar=scalar)
    value = float(cost_functional.cost(point, order=0))
    t_init = float(point.get_scalar()) if is_augmented_vector(point) else None
    return point.get_local(), value, t_init


def get_model_var_warm_start(cost_functional):
    var_value = getattr(cost_functional, "var", None)
    if var_value is None:
        return None
    return float(var_value)


def zero_control_np(control_model):
    z = control_model.generate_vector(soupy.CONTROL)
    z.zero()
    z.apply("")
    return np.array(z.get_local(), copy=True)


def make_cvar_saa_cost(
    control_model,
    prior,
    penalty,
    beta,
    sample_size,
    seed,
    comm_sampler,
    epsilon=1e-4,
):
    settings = superquantileRiskMeasureSAASettings()
    settings["beta"] = beta
    settings["sample_size"] = sample_size
    settings["seed"] = seed
    settings["epsilon"] = float(epsilon)
    risk = SuperquantileRiskMeasureSAA(control_model, prior, settings=settings, comm_sampler=comm_sampler)
    return RiskMeasureControlCostFunctional(risk, penalty)


def model_uses_world_parallel(model_name: str) -> bool:
    if model_name.startswith("mixture_"):
        return True
    if model_name.startswith("saa_"):
        return int(model_name.split("_", 1)[1]) >= 100
    return False


def is_saa_model(model_name: str) -> bool:
    return model_name.startswith("saa_")


def should_use_linear_warm_start(model_name: str) -> bool:
    return model_name != "linear" and not is_saa_model(model_name)


def is_explicit_continuation_model(model_name: str) -> bool:
    return (
        model_name in {"quadratic", "mixture_quadratic_kle", "mixture_quadratic_hep"}
        or is_saa_model(model_name)
    )


def make_cvar_cost(model_name, control_model, prior, penalty, args, epsilon_override=None):
    if is_saa_model(model_name):
        sample_size = int(model_name.split("_", 1)[1])
        return make_cvar_saa_cost(
            control_model,
            prior,
            penalty,
            beta=args.cvar_beta,
            sample_size=sample_size,
            seed=args.saa_seed,
            comm_sampler=MPI.COMM_WORLD if sample_size >= 100 else MPI.COMM_SELF,
            epsilon=1e-4 if epsilon_override is None else float(epsilon_override),
        )

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
                "epsilon": 1e-4 if epsilon_override is None else float(epsilon_override),
                "verbose": args.verbose,
            },
        )
    if model_name == "mixture_linear_kle":
        return TaylorMixtureLinearCVaRControlCostFunctional(
            control_model,
            prior,
            penalty,
            {"beta": args.cvar_beta, "N_mix": args.n_mix, "direction": "kle", "verbose": args.verbose},
            comm_sampler=MPI.COMM_WORLD,
        )
    if model_name == "mixture_linear_hep":
        return TaylorMixtureLinearCVaRControlCostFunctional(
            control_model,
            prior,
            penalty,
            {"beta": args.cvar_beta, "N_mix": args.n_mix, "direction": "hep", "verbose": args.verbose},
            comm_sampler=MPI.COMM_WORLD,
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
                "epsilon": 1e-4 if epsilon_override is None else float(epsilon_override),
                "verbose": args.verbose,
            },
            comm_sampler=MPI.COMM_WORLD,
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
                "epsilon": 1e-4 if epsilon_override is None else float(epsilon_override),
                "verbose": args.verbose,
            },
            comm_sampler=MPI.COMM_WORLD,
        )
    raise ValueError(f"Unknown model name: {model_name}")


def evaluate_true_cvar_stats(cost_functional, control_np, beta):
    """Return exact sample CVaR statistics from the risk measure's QoI samples."""
    point = cost_functional.generate_vector(soupy.CONTROL)
    set_control_part(point, control_np, scalar=0.0)
    cost_functional.cost(point, order=0)
    risk = cost_functional.risk_measure
    samples = risk.gather_samples()
    t_opt, cvar_qoi = exact_cvar_from_samples(samples, beta)
    if cost_functional.penalization is None:
        penalty = 0.0
    else:
        penalty = float(cost_functional.penalization.cost(point))
    total_cost = float(cvar_qoi + penalty)
    return total_cost, float(np.mean(samples)), float(np.var(samples)), float(cvar_qoi), float(t_opt)


def optimize_with_tracking(model_name, approx_cost, x0, bounds, args, rank, maxiter, root_only=True):
    payload = None
    if (not root_only) or rank == 0:
        wrapper = ScipyObjectiveWithHistory(approx_cost)
        iter_records: List[IterRecord] = []
        callback_last_time = None

        def callback(xk):
            nonlocal callback_last_time
            now = time.perf_counter()
            iter_time = np.nan if callback_last_time is None else now - callback_last_time
            callback_last_time = now
            iter_records.append(
                IterRecord(
                    iteration=len(iter_records) + 1,
                    model_cost=float(wrapper.latest_cost),
                    residual=projected_gradient_inf_norm(xk, wrapper.latest_gradient, bounds),
                    epsilon=float(wrapper.current_smoothing_epsilon()),
                    iter_time_sec=iter_time,
                    rss_mb=get_current_rss_mb(),
                    peak_rss_mb=get_peak_rss_mb(),
                    cost_rss_before_mb=float(wrapper.latest_cost_rss_before_mb),
                    cost_rss_after_mb=float(wrapper.latest_cost_rss_after_mb),
                    grad_rss_before_mb=float(wrapper.latest_grad_rss_before_mb),
                    grad_rss_after_mb=float(wrapper.latest_grad_rss_after_mb),
                )
            )

        t0 = time.perf_counter()
        result = scipy.optimize.minimize(
            wrapper.function(),
            x0,
            method="L-BFGS-B",
            jac=wrapper.jac(),
            callback=callback,
            bounds=bounds,
            options={
                "maxiter": maxiter,
                "disp": False,
                "ftol": 1e-20,
                "gtol": 1e-4,
                "maxls": 100,
            },
        )
        total_time = time.perf_counter() - t0
        iter_count = len(iter_records) if iter_records else int(result.nit)
        avg_iter_time_sec = total_time / max(iter_count, 1)

        for r in iter_records:
            if rank == 0 and r.iteration % args.print_every == 0:
                print(
                    f"  [{model_name:20s}] iter {r.iteration:4d}: "
                    f"J_model={r.model_cost:.6e}, ||proj g||_inf={r.residual:.3e}, "
                    f"epsilon={r.epsilon:.3e}, RSS={r.rss_mb:.1f} MB, PeakRSS={r.peak_rss_mb:.1f} MB"
                )
                sys.stdout.flush()

        z_opt_vec = np_to_control(approx_cost, result.x)
        control_opt_np, t_opt = split_control(z_opt_vec)
        approx_opt = float(approx_cost.cost(z_opt_vec, order=0))

        if rank == 0:
            serial_note = " [root-only serial]" if root_only else ""
            print(
                f"  [{model_name:20s}] done{serial_note}: success={result.success}, nit={iter_count}, "
                f"avg_iter_time={avg_iter_time_sec:.2f}s"
            )
            print(
                f"  [{model_name:20s}] termination: status={int(result.status)}, "
                f"reason={str(result.message)}"
            )
            sys.stdout.flush()

        payload = {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "njev": int(result.njev),
            "iter_records": iter_records,
            "iter_count": iter_count,
            "avg_iter_time_sec": avg_iter_time_sec,
            "total_time_sec": total_time,
            "x0_np": np.array(x0, copy=True),
            "z_opt_np": np.array(result.x, copy=True),
            "control_opt_np": control_opt_np,
            "t_opt": t_opt,
            "approx_opt": approx_opt,
            "n_func": wrapper.n_func,
            "n_grad": wrapper.n_grad,
        }

    if root_only:
        payload = MPI.COMM_WORLD.bcast(payload if rank == 0 else None, root=0)
    return payload


def optimize_with_explicit_continuation(
    model_name,
    control_model,
    prior,
    penalty,
    x0_np,
    bounds,
    args,
    rank,
    maxiter,
):
    initial_x0_np = np.array(x0_np, copy=True)
    current_x0_np = np.array(x0_np, copy=True)
    total_iter_count = 0
    total_time_sec = 0.0
    total_nfev = 0
    total_njev = 0
    combined_iter_records: List[IterRecord] = []
    stage_payloads = []
    final_payload = None
    root_only = not model_uses_world_parallel(model_name)

    for stage_idx, epsilon in enumerate(EXPLICIT_CVAR_CONTINUATION_LEVELS, start=1):
        if rank == 0:
            print(
                f"  [{model_name:20s}] continuation stage "
                f"{stage_idx}/{len(EXPLICIT_CVAR_CONTINUATION_LEVELS)} with epsilon={epsilon:.3e}"
            )
            sys.stdout.flush()

        approx_cost = make_cvar_cost(
            model_name,
            control_model,
            prior,
            penalty,
            args,
            epsilon_override=epsilon,
        )
        inner_bounds = make_bounds(current_x0_np, args)
        inner_label = f"{model_name}[eps={epsilon:.0e}]"
        inner_payload = optimize_with_tracking(
            inner_label,
            approx_cost,
            current_x0_np,
            inner_bounds,
            args,
            rank,
            maxiter=maxiter,
            root_only=root_only,
        )
        del approx_cost

        for record in inner_payload["iter_records"]:
            combined_iter_records.append(
                IterRecord(
                    iteration=record.iteration + total_iter_count,
                    model_cost=record.model_cost,
                    residual=record.residual,
                    epsilon=record.epsilon,
                    iter_time_sec=record.iter_time_sec,
                    rss_mb=record.rss_mb,
                    peak_rss_mb=record.peak_rss_mb,
                    cost_rss_before_mb=record.cost_rss_before_mb,
                    cost_rss_after_mb=record.cost_rss_after_mb,
                    grad_rss_before_mb=record.grad_rss_before_mb,
                    grad_rss_after_mb=record.grad_rss_after_mb,
                )
            )

        total_iter_count += int(inner_payload["iter_count"])
        total_time_sec += float(inner_payload["total_time_sec"])
        total_nfev += int(inner_payload["nfev"])
        total_njev += int(inner_payload["njev"])
        current_x0_np = np.array(inner_payload["z_opt_np"], copy=True)
        stage_payloads.append(
            {
                "stage": stage_idx,
                "epsilon": float(epsilon),
                "success": bool(inner_payload["success"]),
                "status": int(inner_payload["status"]),
                "message": str(inner_payload["message"]),
                "iter_count": int(inner_payload["iter_count"]),
                "approx_opt": float(inner_payload["approx_opt"]),
                "outer_payloads": inner_payload.get("outer_payloads"),
            }
        )
        final_payload = inner_payload

    if final_payload is None:
        raise RuntimeError(f"Explicit continuation for {model_name} produced no payload.")

    final_payload = dict(final_payload)
    final_payload["success"] = all(payload["success"] for payload in stage_payloads)
    final_payload["x0_np"] = np.array(initial_x0_np, copy=True)
    final_payload["z_opt_np"] = np.array(current_x0_np, copy=True)
    final_payload["iter_records"] = combined_iter_records
    final_payload["iter_count"] = total_iter_count
    final_payload["avg_iter_time_sec"] = total_time_sec / max(total_iter_count, 1)
    final_payload["total_time_sec"] = total_time_sec
    final_payload["nfev"] = total_nfev
    final_payload["njev"] = total_njev
    final_payload["continuation_stages"] = stage_payloads
    if not final_payload["success"]:
        final_payload["message"] = (
            "One or more continuation stages failed; final stage: " + str(final_payload["message"])
        )
    return final_payload


def make_bounds(x0, args):
    lb = np.full_like(x0, args.bound_lb, dtype=float)
    ub = np.full_like(x0, args.bound_ub, dtype=float)
    if len(x0) == args.control_dim + 1:
        lb[-1] = -np.inf
        ub[-1] = np.inf
    return scipy.optimize.Bounds(lb, ub)


def plot_curves(results, save_dir):
    plt.figure(figsize=(9, 5))
    for model in MODEL_ORDER:
        rec = results[model]["iter_records"]
        if not rec:
            continue
        x = [r.iteration for r in rec]
        y = [max(abs(r.model_cost), 1e-16) for r in rec]
        plt.semilogy(x, y, marker="o", linewidth=1.5, markersize=3, color=MODEL_COLORS[model], label=model)
    plt.xlabel("Iteration")
    plt.ylabel("Model CVaR Objective")
    plt.title("CVaR Objective Value per Iteration")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "objective_per_iteration.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    for model in MODEL_ORDER:
        rec = results[model]["iter_records"]
        if not rec:
            continue
        x = [r.iteration for r in rec]
        y = [max(r.residual, 1e-16) for r in rec]
        plt.semilogy(x, y, marker="o", linewidth=1.5, markersize=3, color=MODEL_COLORS[model], label=model)
    plt.xlabel("Iteration")
    plt.ylabel("Projected Gradient Inf Norm")
    plt.title("Projected Gradient Inf Norm per Iteration")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "residual_per_iteration.png"), dpi=180)
    plt.close()

    x = np.arange(len(MODEL_ORDER))
    plt.figure(figsize=(11, 5))
    plt.bar(x, [results[m]["avg_iter_time_sec"] for m in MODEL_ORDER], color=[MODEL_COLORS[m] for m in MODEL_ORDER])
    plt.xticks(x, MODEL_ORDER, rotation=25, ha="right")
    plt.ylabel("Average Time per Iteration (s)")
    plt.title("Average Iteration Time Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "avg_iteration_time_comparison.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.bar(x, [results[m]["iter_count"] for m in MODEL_ORDER], color=[MODEL_COLORS[m] for m in MODEL_ORDER])
    plt.xticks(x, MODEL_ORDER, rotation=25, ha="right")
    plt.ylabel("Total Iteration Count")
    plt.title("Iteration Count Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "iteration_count_comparison.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(x, [results[m]["opt_rel_err"] for m in MODEL_ORDER], color=[MODEL_COLORS[m] for m in MODEL_ORDER])
    plt.yscale("log")
    plt.xticks(x, MODEL_ORDER, rotation=25, ha="right")
    plt.ylabel("|J_model(z*) - J_true(z*)| / |J_true(z*)|")
    plt.title("Final CVaR Objective Relative Error")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "final_objective_rel_error.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(x, [results[m]["init_rel_err"] for m in MODEL_ORDER], color=[MODEL_COLORS[m] for m in MODEL_ORDER])
    plt.yscale("log")
    plt.xticks(x, MODEL_ORDER, rotation=25, ha="right")
    plt.ylabel("|J_model(z0) - J_true(z0)| / |J_true(z0)|")
    plt.title("Initial CVaR Objective Relative Error")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "initial_objective_rel_error.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(x, [results[m]["true_opt"] for m in MODEL_ORDER], color=[MODEL_COLORS[m] for m in MODEL_ORDER])
    plt.xticks(x, MODEL_ORDER, rotation=25, ha="right")
    plt.ylabel("J_true(z*)")
    plt.title("Final True CVaR Objective Value")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "final_true_objective_value.png"), dpi=180)
    plt.close()


def vector_to_function(function_space, vector):
    function = dl.Function(function_space)
    function.vector().zero()
    function.vector().axpy(1.0, vector)
    return function


def scalarize_for_plot(function, scalar_space):
    if function.function_space().num_sub_spaces() > 0:
        return dl.project(dl.sqrt(dl.inner(function, function)), scalar_space)
    return function


def plot_on_axes(function, ax):
    plt.sca(ax)
    return dl.plot(function)


def make_target_state_function(function_space):
    mesh = function_space.mesh()
    target_expr = dl.Expression(
        "sin(2*pi*x[0])*sin(2*pi*x[1])",
        pi=np.pi,
        degree=4,
        mpi_comm=mesh.mpi_comm(),
    )
    return dl.interpolate(target_expr, function_space)


def control_coefficients_to_grid(control_np, control_parameters):
    coeffs = np.asarray(control_np, dtype=float)
    n_wells = int(control_parameters.n_wells_per_side)
    if coeffs.size != n_wells * n_wells:
        raise ValueError(
            f"Expected {n_wells * n_wells} control coefficients, received {coeffs.size}."
        )

    well_grid = np.linspace(
        control_parameters.loc_lower,
        control_parameters.loc_upper,
        n_wells,
    )
    coeff_grid = np.zeros((n_wells, n_wells))
    count = 0
    for i in range(n_wells):
        for j in range(n_wells):
            coeff_grid[j, i] = coeffs[count]
            count += 1
    return well_grid, coeff_grid


def plot_control_coefficients_on_axes(control_np, control_parameters, ax):
    well_grid, coeff_grid = control_coefficients_to_grid(control_np, control_parameters)
    z_max = float(np.max(np.abs(coeff_grid)))
    if z_max == 0.0:
        z_max = 1.0
    width = (control_parameters.loc_upper - control_parameters.loc_lower) / control_parameters.n_wells_per_side
    extent = [
        well_grid[0] - 0.5 * width,
        well_grid[-1] + 0.5 * width,
        well_grid[0] - 0.5 * width,
        well_grid[-1] + 0.5 * width,
    ]
    artist = ax.imshow(
        coeff_grid,
        origin="lower",
        extent=extent,
        cmap=plt.get_cmap("coolwarm"),
        norm=Normalize(vmin=-z_max, vmax=z_max),
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return artist



def save_parameter_sample_plots(prior, Vh, save_dir, sample_count=3, seed=11):
    V_parameter = Vh[soupy.PARAMETER]
    V_parameter_scalar = dl.FunctionSpace(V_parameter.mesh(), "CG", 1)
    noise = dl.Vector(V_parameter.mesh().mpi_comm())
    prior.init_vector(noise, "noise")
    rng = hp.Random(seed=seed)

    fig, axes = plt.subplots(1, sample_count, figsize=(4.2 * sample_count, 3.6))
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        m = prior.mean.copy()
        rng.normal(1.0, noise)
        prior.sample(noise, m)
        m_fun = scalarize_for_plot(vector_to_function(V_parameter, m), V_parameter_scalar)
        artist = plot_on_axes(m_fun, ax)
        ax.set_title(f"parameter sample {i + 1}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "parameter_samples.png"), dpi=180)
    plt.close(fig)

def solve_state_at_control(control_model, prior, control_np):
    x = control_model.generate_vector("ALL")
    x[soupy.PARAMETER].zero()
    x[soupy.PARAMETER].axpy(1.0, prior.mean)
    x[soupy.CONTROL].set_local(control_np)
    x[soupy.CONTROL].apply("")
    control_model.solveFwd(x[soupy.STATE], x)
    return x[soupy.CONTROL].copy(), x[soupy.STATE].copy()


def save_optimal_field_plots(results, control_model, prior, Vh, control_parameters, save_dir):
    V_state = Vh[soupy.STATE]
    V_state_scalar = dl.FunctionSpace(V_state.mesh(), "CG", 1)
    target_fun = make_target_state_function(V_state_scalar)

    overview_payload = []
    for model_name in MODEL_ORDER:
        control_vec, state_vec = solve_state_at_control(control_model, prior, results[model_name]["control_opt_np"])
        control_np = np.array(control_vec.get_local(), copy=True)
        control_fun = control_coefficients_to_function(V_state_scalar, control_vec, control_parameters)
        state_fun = scalarize_for_plot(vector_to_function(V_state, state_vec), V_state_scalar)

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        artist = plot_on_axes(control_fun, axes[0, 0])
        axes[0, 0].set_title(f"{model_name} source(z*)")
        axes[0, 0].set_xlabel("x")
        axes[0, 0].set_ylabel("y")
        plt.colorbar(artist, ax=axes[0, 0], fraction=0.046, pad=0.04)
        artist = plot_control_coefficients_on_axes(control_np, control_parameters, axes[0, 1])
        axes[0, 1].set_title(f"{model_name} well coefficients z*")
        plt.colorbar(artist, ax=axes[0, 1], fraction=0.046, pad=0.04)
        artist = plot_on_axes(state_fun, axes[1, 0])
        axes[1, 0].set_title(f"{model_name} u(z*)")
        axes[1, 0].set_xlabel("x")
        axes[1, 0].set_ylabel("y")
        plt.colorbar(artist, ax=axes[1, 0], fraction=0.046, pad=0.04)
        artist = plot_on_axes(target_fun, axes[1, 1])
        axes[1, 1].set_title("target state")
        axes[1, 1].set_xlabel("x")
        axes[1, 1].set_ylabel("y")
        plt.colorbar(artist, ax=axes[1, 1], fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{model_name}_optimal_fields.png"), dpi=180)
        plt.close(fig)
        overview_payload.append((model_name, control_np, control_fun, state_fun))

    n_models = len(overview_payload)
    fig, axes = plt.subplots(n_models, 4, figsize=(18, 3.6 * n_models))
    if n_models == 1:
        axes = np.array([axes])
    for j, title in enumerate(["optimal control source", "well coefficients z*", "state u(z*)", "target state"]):
        axes[0, j].set_title(title)
    for i, (model_name, control_np, control_fun, state_fun) in enumerate(overview_payload):
        plot_on_axes(control_fun, axes[i, 0])
        plot_control_coefficients_on_axes(control_np, control_parameters, axes[i, 1])
        plot_on_axes(state_fun, axes[i, 2])
        plot_on_axes(target_fun, axes[i, 3])
        for ax in axes[i, :]:
            ax.set_xticks([])
            ax.set_yticks([])
        axes[i, 0].set_ylabel(model_name, rotation=90, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "optimal_fields_overview.png"), dpi=180)
    plt.close(fig)


def save_optimal_pde_solution_plot(results, control_model, prior, Vh, save_dir):
    V_state = Vh[soupy.STATE]
    V_state_scalar = dl.FunctionSpace(V_state.mesh(), "CG", 1)
    target_fun = make_target_state_function(V_state_scalar)

    payload = []
    for model_name in MODEL_ORDER:
        _, state_vec = solve_state_at_control(control_model, prior, results[model_name]["control_opt_np"])
        state_fun = scalarize_for_plot(vector_to_function(V_state, state_vec), V_state_scalar)
        payload.append((model_name, state_fun))

    fig, axes = plt.subplots(len(payload), 2, figsize=(9, 3.6 * len(payload)))
    axes = np.atleast_2d(axes)
    for i, (model_name, state_fun) in enumerate(payload):
        for ax, fun, title in zip(axes[i], [state_fun, target_fun], [f"{model_name} u(z*)", "target state"]):
            artist = plot_on_axes(fun, ax)
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            plt.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "optimal_pde_solutions.png"), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare Taylor models against exact-CVaR SAA truth on semilinear ADR control")
    parser.add_argument("--cvar-beta", "--beta", dest="cvar_beta", type=float, default=0.95, help="CVaR confidence level")
    parser.add_argument("--n-tr", type=int, default=50, help="Number of Hessian modes for quadratic models")
    parser.add_argument("--n-mix", type=int, default=11, help="Number of mixture components")
    parser.add_argument("--quadratic-cvar-n-mc", type=int, default=10000, help="MC samples for quadratic CVaR surrogates")
    parser.add_argument("--truth-saa-samples", type=int, default=100000, help="Sample size for ground-truth CVaR SAA evaluation")
    parser.add_argument("--saa-seed", type=int, default=1)
    parser.add_argument("--penalty", type=float, default=0)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--maxiter-saa", type=int, default=500)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--bound-lb", type=float, default=-4.0, help="Lower bound for each Gaussian-well control coefficient")
    parser.add_argument("--bound-ub", type=float, default=4.0, help="Upper bound for each Gaussian-well control coefficient")
    parser.add_argument("--newton-max-it", type=int, default=50)
    parser.add_argument("--newton-rtol", type=float, default=1e-8)
    parser.add_argument("--newton-atol", type=float, default=1e-10)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--save-dir", type=str, default="results_semilinear_adr_compare_cvar_model")
    parser.add_argument("--log-file", type=str, default="terminal_output.txt")
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    args = parser.parse_args()

    rank = MPI.COMM_WORLD.Get_rank()
    comm_mesh = MPI.COMM_SELF
    os.makedirs(args.save_dir, exist_ok=True)

    log_path = args.log_file
    if not os.path.isabs(log_path):
        log_path = os.path.join(args.save_dir, log_path)

    log_file = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if rank == 0:
        log_file = open(log_path, "w", buffering=1)
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)

    try:
        if rank == 0:
            print("=" * 90)
            print("Semilinear ADR CVaR Model Comparison:")
            print("  linear / quadratic / mixture_linear_kle / mixture_linear_hep")
            print("  mixture_quadratic_kle / mixture_quadratic_hep")
            print("  saa_1 / saa_10 / saa_39 / saa_100 / saa_200 / saa_500 / saa_1000 / saa_10000")
            print("  serial: linear / quadratic / saa_1 / saa_10 / saa_39")
            print("  parallel on all ranks: mixture_* / saa_100 / saa_200 / saa_500 / saa_1000 / saa_10000")
            print("=" * 90)
            print(f"Ground-truth exact-CVaR sample count: {args.truth_saa_samples}")
            print(
                f"cvar_beta={args.cvar_beta}, n_tr={args.n_tr}, n_mix={args.n_mix}, "
                f"quadratic_cvar_n_mc={args.quadratic_cvar_n_mc}"
            )
            print(f"penalty={args.penalty}")
            print(f"box constraint on control: {args.bound_lb} <= z <= {args.bound_ub}")
            print("control source term: rhs = sum_i z_i psi_i (no known source f)")
            print("initial control: zero well coefficients")
            print(
                "Explicit continuation for quadratic and SAA models: "
                + " -> ".join(f"{eps:.0e}" for eps in EXPLICIT_CVAR_CONTINUATION_LEVELS)
            )
            print(f"terminal log file: {log_path}")
            print("=" * 90)
            sys.stdout.flush()

        problem = setup_problem(args, comm_mesh)
        Vh = problem["Vh"]
        control_model = problem["control_model"]
        prior = problem["prior"]
        control_parameters = problem["control_parameters"]
        if rank == 0:
            save_parameter_sample_plots(prior, Vh, args.save_dir)
        penalty = problem["penalty"]
        control0_np = zero_control_np(control_model)
        args.control_dim = int(control0_np.size)
        linear_init_control_np = np.array(control0_np, copy=True)
        linear_init_t = 0.0

        results: Dict[str, Dict] = {}
        for model_name in MODEL_ORDER:
            approx_cost = None
            x0_np = None
            approx_init = None
            init_t = None
            use_linear_warm_start = should_use_linear_warm_start(model_name)
            run_parallel_model = model_uses_world_parallel(model_name)
            epsilon_override = (
                EXPLICIT_CVAR_CONTINUATION_LEVELS[0]
                if is_explicit_continuation_model(model_name)
                else None
            )

            if run_parallel_model:
                approx_cost = make_cvar_cost(
                    model_name, control_model, prior, penalty, args, epsilon_override=epsilon_override
                )
                init_control_np = linear_init_control_np if use_linear_warm_start else control0_np
                init_control_np = MPI.COMM_WORLD.bcast(init_control_np if rank == 0 else None, root=0)
                init_scalar = linear_init_t if use_linear_warm_start else 0.0
                x0_np, approx_init, init_t = initial_model_point_at_control(
                    approx_cost,
                    init_control_np,
                    scalar=init_scalar,
                )
            elif rank == 0:
                approx_cost = make_cvar_cost(
                    model_name, control_model, prior, penalty, args, epsilon_override=epsilon_override
                )
                init_control_np = linear_init_control_np if use_linear_warm_start else control0_np
                init_scalar = linear_init_t if use_linear_warm_start else 0.0
                x0_np, approx_init, init_t = initial_model_point_at_control(
                    approx_cost,
                    init_control_np,
                    scalar=init_scalar,
                )
            else:
                init_control_np = None

            x0_np = MPI.COMM_WORLD.bcast(x0_np if rank == 0 else None, root=0)
            approx_init = MPI.COMM_WORLD.bcast(approx_init if rank == 0 else None, root=0)
            init_t = MPI.COMM_WORLD.bcast(init_t if rank == 0 else None, root=0)
            init_control_np = MPI.COMM_WORLD.bcast(init_control_np if rank == 0 else None, root=0)

            truth_cost = make_cvar_saa_cost(
                control_model,
                prior,
                penalty,
                beta=args.cvar_beta,
                sample_size=args.truth_saa_samples,
                seed=args.saa_seed,
                comm_sampler=MPI.COMM_WORLD,
            )
            true_init, true_init_mean, true_init_var, true_init_cvar, _ = evaluate_true_cvar_stats(
                truth_cost, init_control_np, args.cvar_beta
            )
            del truth_cost
            init_rel_err = relative_error(approx_init, true_init)

            if rank == 0:
                print(f"\nOptimizing {model_name} with L-BFGS-B ...")
                if use_linear_warm_start:
                    print(f"  [{model_name:20s}] initial guess: linear optimum z* with t initialized from linear surrogate VaR")
                else:
                    print(f"  [{model_name:20s}] initial guess: zero control with t=0")
                print(
                    f"  [{model_name:20s}] initial: "
                    f"J_model(init)={approx_init:.6e}, J_true(init)={true_init:.6e}, "
                    f"cvar_true(init)={true_init_cvar:.6e}, mean_qoi(init)={true_init_mean:.6e}, "
                    f"var_qoi(init)={true_init_var:.6e}, rel_err(init)={init_rel_err:.3e}"
                )
                sys.stdout.flush()

            bounds = make_bounds(x0_np, args) if (run_parallel_model or rank == 0) else None
            if is_explicit_continuation_model(model_name):
                res = optimize_with_explicit_continuation(
                    model_name,
                    control_model,
                    prior,
                    penalty,
                    x0_np,
                    bounds,
                    args,
                    rank,
                    maxiter=args.maxiter_saa if is_saa_model(model_name) else args.maxiter,
                )
            else:
                res = optimize_with_tracking(
                    model_name,
                    approx_cost,
                    x0_np,
                    bounds,
                    args,
                    rank,
                    maxiter=args.maxiter_saa if is_saa_model(model_name) else args.maxiter,
                    root_only=not run_parallel_model,
                )
            res["control_init_np"] = np.array(init_control_np, copy=True)
            res["t_init"] = None if init_t is None else float(init_t)
            res["approx_init"] = float(approx_init)
            res["true_init"] = float(true_init)
            res["true_init_mean"] = float(true_init_mean)
            res["true_init_var"] = float(true_init_var)
            res["true_init_cvar"] = float(true_init_cvar)
            res["init_rel_err"] = float(init_rel_err)

            truth_cost = make_cvar_saa_cost(
                control_model,
                prior,
                penalty,
                beta=args.cvar_beta,
                sample_size=args.truth_saa_samples,
                seed=args.saa_seed,
                comm_sampler=MPI.COMM_WORLD,
            )
            true_opt, true_opt_mean, true_opt_var, true_opt_cvar, t_true_opt = evaluate_true_cvar_stats(
                truth_cost, res["control_opt_np"], args.cvar_beta
            )
            del truth_cost
            res["true_opt"] = float(true_opt)
            res["true_opt_mean"] = float(true_opt_mean)
            res["true_opt_var"] = float(true_opt_var)
            res["true_opt_cvar"] = float(true_opt_cvar)
            res["t_true_opt"] = float(t_true_opt)
            res["opt_rel_err"] = relative_error(res["approx_opt"], res["true_opt"])
            res["opt_t_var_rel_err"] = optional_relative_error(res["t_opt"], res["t_true_opt"])
            results[model_name] = res
            if model_name == "linear":
                linear_init_control_np = np.array(res["control_opt_np"], copy=True)
                linear_var = get_model_var_warm_start(approx_cost)
                linear_init_t = 0.0 if linear_var is None else float(linear_var)

            if approx_cost is not None:
                del approx_cost
            if rank == 0:
                print(
                    f"  [{model_name:20s}] optimal: "
                    f"J_model(z*)={res['approx_opt']:.6e}, J_true(z*)={res['true_opt']:.6e}, "
                    f"cvar_true(z*)={res['true_opt_cvar']:.6e}, mean_qoi(z*)={res['true_opt_mean']:.6e}, "
                    f"var_qoi(z*)={res['true_opt_var']:.6e}, rel_err(z*)={res['opt_rel_err']:.3e}"
                )
                sys.stdout.flush()

        if rank == 0:
            for model_name in MODEL_ORDER:
                save_iteration_csv(
                    os.path.join(args.save_dir, f"{model_name}_iteration_metrics.csv"),
                    results[model_name]["iter_records"],
                )

            summary_rows = []
            for model_name in MODEL_ORDER:
                rr = results[model_name]
                summary_rows.append(
                    [
                        model_name,
                        rr["success"],
                        rr["message"],
                        rr["iter_count"],
                        rr["nfev"],
                        rr["njev"],
                        rr["avg_iter_time_sec"],
                        rr["approx_init"],
                        rr["true_init"],
                        rr["init_rel_err"],
                        rr["approx_opt"],
                        rr["true_opt"],
                        rr["opt_rel_err"],
                        rr["t_opt"],
                        rr["t_true_opt"],
                        rr["opt_t_var_rel_err"],
                    ]
                )
            save_summary_csv(os.path.join(args.save_dir, "summary.csv"), summary_rows)

            json_payload = {
                "config": vars(args),
                "problem": {
                    "mesh": {
                        "nx": int(problem["mesh_parameters"].nx),
                        "ny": int(problem["mesh_parameters"].ny),
                    },
                    "qoi_type": problem["qoi_type"],
                    "control_source": "sum_i z_i psi_i",
                    "control_bounds": {"lb": float(args.bound_lb), "ub": float(args.bound_ub)},
                    "penalization": {"type": "L2", "alpha": float(args.penalty)},
                },
                "truth_evaluation": {
                    "sample_size": int(args.truth_saa_samples),
                    "seed": int(args.saa_seed),
                    "cvar_beta": float(args.cvar_beta),
                    "method": "exact empirical CVaR from sampled QoI list plus penalization",
                },
                "models": {
                    name: {
                        "success": bool(results[name]["success"]),
                        "message": str(results[name]["message"]),
                        "nit": int(results[name]["iter_count"]),
                        "nfev": int(results[name]["nfev"]),
                        "njev": int(results[name]["njev"]),
                        "avg_iter_time_sec": float(results[name]["avg_iter_time_sec"]),
                        "initial_model_objective": float(results[name]["approx_init"]),
                        "initial_true_objective": float(results[name]["true_init"]),
                        "initial_true_cvar": float(results[name]["true_init_cvar"]),
                        "initial_rel_error": float(results[name]["init_rel_err"]),
                        "opt_model_objective": float(results[name]["approx_opt"]),
                        "opt_true_objective": float(results[name]["true_opt"]),
                        "opt_true_cvar": float(results[name]["true_opt_cvar"]),
                        "opt_rel_error": float(results[name]["opt_rel_err"]),
                        "opt_model_t": None if results[name]["t_opt"] is None else float(results[name]["t_opt"]),
                        "opt_true_var": float(results[name]["t_true_opt"]),
                        "opt_t_var_rel_error": None
                        if results[name]["opt_t_var_rel_err"] is None
                        else float(results[name]["opt_t_var_rel_err"]),
                    }
                    for name in MODEL_ORDER
                },
            }
            with open(os.path.join(args.save_dir, "summary.json"), "w") as f:
                json.dump(json_payload, f, indent=2)

            plot_curves(results, args.save_dir)
            save_optimal_field_plots(results, control_model, prior, Vh, control_parameters, args.save_dir)
            save_optimal_pde_solution_plot(results, control_model, prior, Vh, args.save_dir)

            with open(os.path.join(args.save_dir, "timing_comparison.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["model", "total_iterations", "avg_iter_time_sec"])
                writer.writerows([[m, results[m]["iter_count"], results[m]["avg_iter_time_sec"]] for m in MODEL_ORDER])

            print("\n" + "-" * 90)
            print("Summary (ground-truth exact CVaR from sampled QoI list, evaluated at z0 and z*)")
            print("-" * 90)
            for model_name in MODEL_ORDER:
                rr = results[model_name]
                t_rel_err = "NA" if rr["opt_t_var_rel_err"] is None else f"{rr['opt_t_var_rel_err']:.3e}"
                print(
                    f"{model_name:20s} | nit={rr['iter_count']:3d} | "
                    f"avg_iter_time={rr['avg_iter_time_sec']:8.2f}s | "
                    f"J_model(z*)={rr['approx_opt']:.6e} | "
                    f"init rel_err={rr['init_rel_err']:.3e} | "
                    f"opt rel_err={rr['opt_rel_err']:.3e} | "
                    f"J_true(z*)={rr['true_opt']:.6e} | "
                    f"t/VaR rel_err={t_rel_err}"
                )
            print("-" * 90)
            print(f"All outputs written to: {args.save_dir}")
            print(f"Terminal outputs saved to: {log_path}")
    finally:
        if rank == 0 and log_file is not None:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()


if __name__ == "__main__":
    main()
