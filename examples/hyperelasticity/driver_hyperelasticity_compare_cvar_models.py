"""Compare Taylor and SAA CVaR models on hyperelasticity control.

Models compared:
- linear
- quadratic
- mixture_linear_kle
- mixture_linear_hep
- mixture_quadratic_kle
- mixture_quadratic_hep
- saa_1
- saa_10
- saa_100
- saa_200
- saa_500
- saa_1000
- saa_10000

Reference truth:
- Exact empirical CVaR from a large fixed QoI sample set
- Plus the same L2 control penalty used by each model
- Evaluated at the initial control and each model optimum

The CVaR objective includes the same L2 control penalty as the mean-variance
comparison driver. Quadratic CVaR Taylor models use Monte Carlo samples to
estimate the surrogate CVaR; the default is 10000.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List

# Configure macOS compiler BEFORE importing dolfin or soupy
_SOUPY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, os.path.join(_SOUPY_ROOT, "soupy", "utils"))
try:
    from macos_config import configure_macos_compiler, configure_dolfin_form_compiler

    configure_macos_compiler()
except ImportError:
    configure_dolfin_form_compiler = None
sys.path.pop(0)

sys.path.append(os.environ.get("HIPPYLIB_PATH", ""))
sys.path.append(_SOUPY_ROOT)

import dolfin as dl
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from mpi4py import MPI

import hippylib as hp
import soupy
from soupy import (
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

from setupHyperelasticityProblem import hyperelasticity_problem_settings, setup_hyperelasticity_problem


logging.getLogger("FFC").setLevel(logging.WARNING)
logging.getLogger("UFL").setLevel(logging.WARNING)
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
    "saa_100": "#7f7f7f",
    "saa_200": "#9467bd",
    "saa_500": "#e377c2",
    "saa_1000": "#8c564b",
    "saa_10000": "#bcbd22",
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


class VolumeFractionPenalization(soupy.Penalization):
    def __init__(self, Vh, alpha=1.0, target_fraction=0.5):
        self.Vh = Vh
        self.alpha = float(alpha)
        self.domain_measure = float(dl.assemble(dl.Constant(1.0) * dl.dx(domain=self.Vh[soupy.CONTROL].mesh())))
        self.target_volume = float(target_fraction) * self.domain_measure
        z_test = dl.TestFunction(self.Vh[soupy.CONTROL])
        self.volume_vec = dl.assemble(z_test * dl.dx)

    def init_vector(self, z):
        if hasattr(z, "get_vector"):
            return
        self.volume_vec.init_vector(z, 0)

    def _control_vector(self, z):
        if hasattr(z, "get_vector"):
            return z.get_vector()
        return z

    def cost(self, z):
        z_vec = self._control_vector(z)
        volume_misfit = self.volume_vec.inner(z_vec) - self.target_volume
        return self.alpha * volume_misfit * volume_misfit

    def grad(self, z, out):
        if hasattr(z, "get_vector"):
            out.set_scalar(0.0)
            out_vec = out.get_vector()
            z_vec = z.get_vector()
        else:
            out_vec = out
            z_vec = z
        out_vec.zero()
        volume_misfit = self.volume_vec.inner(z_vec) - self.target_volume
        out_vec.axpy(2.0 * self.alpha * volume_misfit, self.volume_vec)

    def hessian(self, z, zhat, out):
        del z
        if hasattr(zhat, "get_vector"):
            out.set_scalar(0.0)
            out_vec = out.get_vector()
            zhat_vec = zhat.get_vector()
        else:
            out_vec = out
            zhat_vec = zhat
        out_vec.zero()
        out_vec.axpy(2.0 * self.alpha * self.volume_vec.inner(zhat_vec), self.volume_vec)


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
    settings = hyperelasticity_problem_settings()
    settings["qoi_type"] = args.qoi_type
    settings["geometry"]["lx"] = args.lx
    settings["geometry"]["ly"] = args.ly
    settings["geometry"]["lz"] = args.lz
    settings["geometry"]["dim"] = args.geometry_dim
    settings["mesh"]["nx"] = args.nx
    settings["mesh"]["ny"] = args.ny
    settings["mesh"]["nz"] = args.nz
    mesh, Vh, _, control_model, prior = setup_hyperelasticity_problem(settings, comm_mesh)
    penalty = VolumeFractionPenalization(Vh, args.penalty, target_fraction=0.5)
    return mesh, Vh, control_model, prior, penalty


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


def warm_start_source(model_name: str):
    if model_name == "quadratic" or model_name.startswith("mixture_linear_"):
        return "linear"
    if model_name.startswith("mixture_quadratic_"):
        return "quadratic"
    return None


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
    # If rank_only = true, that means we only run the optimization on rank 0. 
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
                "maxls": 50,
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
    if len(x0) > 0 and not np.isfinite(x0[-1]):
        return scipy.optimize.Bounds(lb, ub)
    if len(x0) == args.control_dim + 1:
        lb[-1] = -np.inf
        ub[-1] = np.inf
    return scipy.optimize.Bounds(lb, ub)


def save_iteration_csv(path, records: List[IterRecord]):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "iteration",
                "model_cost",
                "projected_grad_inf_norm",
                "epsilon",
                "iter_time_sec",
                "rss_mb",
                "peak_rss_mb",
                "cost_rss_before_mb",
                "cost_rss_after_mb",
                "grad_rss_before_mb",
                "grad_rss_after_mb",
            ]
        )
        for r in records:
            writer.writerow(
                [
                    r.iteration,
                    r.model_cost,
                    r.residual,
                    r.epsilon,
                    r.iter_time_sec,
                    r.rss_mb,
                    r.peak_rss_mb,
                    r.cost_rss_before_mb,
                    r.cost_rss_after_mb,
                    r.grad_rss_before_mb,
                    r.grad_rss_after_mb,
                ]
            )


def save_summary_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "success",
                "message",
                "nit",
                "nfev",
                "njev",
                "avg_iter_time_sec",
                "initial_model_objective",
                "initial_true_objective",
                "initial_rel_error",
                "opt_model_objective",
                "opt_true_objective",
                "opt_rel_error",
                "opt_model_t",
                "opt_true_var",
                "opt_t_var_rel_error",
            ]
        )
        writer.writerows(rows)


def relative_error(model_value, true_value):
    return float(abs(model_value - true_value) / max(abs(true_value), 1e-14))


def optional_relative_error(model_value, true_value):
    if model_value is None:
        return None
    return relative_error(model_value, true_value)


def plot_curves(results, save_dir):
    plt.figure(figsize=(8, 5))
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

    plt.figure(figsize=(8, 5))
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
    plt.figure(figsize=(10, 5))
    plt.bar(x, [results[m]["avg_iter_time_sec"] for m in MODEL_ORDER], color=[MODEL_COLORS[m] for m in MODEL_ORDER])
    plt.xticks(x, MODEL_ORDER, rotation=20, ha="right")
    plt.ylabel("Average Time per Iteration (s)")
    plt.title("Average Iteration Time Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "avg_iteration_time_comparison.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(x, [results[m]["iter_count"] for m in MODEL_ORDER], color=[MODEL_COLORS[m] for m in MODEL_ORDER])
    plt.xticks(x, MODEL_ORDER, rotation=20, ha="right")
    plt.ylabel("Total Iteration Count")
    plt.title("Iteration Count Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "iteration_count_comparison.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(x, [results[m]["opt_rel_err"] for m in MODEL_ORDER], color=[MODEL_COLORS[m] for m in MODEL_ORDER])
    plt.yscale("log")
    plt.xticks(x, MODEL_ORDER, rotation=20, ha="right")
    plt.ylabel("|J_model(z*) - J_true(z*)| / |J_true(z*)|")
    plt.title("Final CVaR Objective Relative Error")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "final_objective_rel_error.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(x, [results[m]["init_rel_err"] for m in MODEL_ORDER], color=[MODEL_COLORS[m] for m in MODEL_ORDER])
    plt.yscale("log")
    plt.xticks(x, MODEL_ORDER, rotation=20, ha="right")
    plt.ylabel("|J_model(z0) - J_true(z0)| / |J_true(z0)|")
    plt.title("Initial CVaR Objective Relative Error")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "initial_objective_rel_error.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(x, [results[m]["true_opt"] for m in MODEL_ORDER], color=[MODEL_COLORS[m] for m in MODEL_ORDER])
    plt.xticks(x, MODEL_ORDER, rotation=20, ha="right")
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


def save_young_modulus_moment_plots(prior, Vh, save_dir, sample_count=10000, seed=23, E0=10.0, E1=100.0):
    V_parameter = Vh[soupy.PARAMETER]
    noise = dl.Vector(V_parameter.mesh().mpi_comm())
    prior.init_vector(noise, "noise")
    rng = hp.Random(seed=seed)

    sum_E = None
    sum_E2 = None
    for _ in range(sample_count):
        m = prior.mean.copy()
        rng.normal(1.0, noise)
        prior.sample(noise, m)
        E_local = E0 + np.exp(m.get_local()) * (E1 - E0)
        if sum_E is None:
            sum_E = np.zeros_like(E_local)
            sum_E2 = np.zeros_like(E_local)
        sum_E += E_local
        sum_E2 += E_local * E_local

    mean_E = sum_E / float(sample_count)
    var_E = np.maximum(sum_E2 / float(sample_count) - mean_E * mean_E, 0.0)
    std_E = np.sqrt(var_E)

    mean_fun = dl.Function(V_parameter)
    mean_fun.vector().set_local(mean_E)
    mean_fun.vector().apply("")
    std_fun = dl.Function(V_parameter)
    std_fun.vector().set_local(std_E)
    std_fun.vector().apply("")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, fun, title in zip(
        axes,
        [mean_fun, std_fun],
        [f"Young modulus mean ({sample_count} samples)", f"Young modulus std ({sample_count} samples)"],
    ):
        artist = plot_on_axes(fun, ax)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "young_modulus_mean_std.png"), dpi=180)
    plt.close(fig)

def solve_state_at_control(control_model, prior, control_np):
    x = control_model.generate_vector("ALL")
    x[soupy.PARAMETER].zero()
    x[soupy.PARAMETER].axpy(1.0, prior.mean)
    x[soupy.CONTROL].set_local(control_np)
    x[soupy.CONTROL].apply("")
    control_model.solveFwd(x[soupy.STATE], x)
    return x[soupy.CONTROL].copy(), x[soupy.STATE].copy()




def solve_state_at_parameter_control(control_model, parameter_vec, control_np):
    x = control_model.generate_vector("ALL")
    x[soupy.PARAMETER].zero()
    x[soupy.PARAMETER].axpy(1.0, parameter_vec)
    x[soupy.CONTROL].set_local(np.asarray(control_np))
    x[soupy.CONTROL].apply("")
    control_model.solveFwd(x[soupy.STATE], x)
    qoi = float(control_model.cost(x))
    return x[soupy.STATE].copy(), qoi


def optimal_control_np(result):
    if "control_opt_np" in result:
        return result["control_opt_np"]
    return result["z_opt_np"]


def save_saa_parameter_solution_sample_plots(results, saa_model_name, control_model, prior, Vh, save_dir, sample_count=3, seed=11):
    if saa_model_name not in results:
        return
    V_parameter = Vh[soupy.PARAMETER]
    V_state = Vh[soupy.STATE]
    V_parameter_scalar = dl.FunctionSpace(V_parameter.mesh(), "CG", 1)
    V_state_scalar = dl.FunctionSpace(V_state.mesh(), "CG", 1)
    control_np = optimal_control_np(results[saa_model_name])
    noise = dl.Vector(V_parameter.mesh().mpi_comm())
    prior.init_vector(noise, "noise")
    rng = hp.Random(seed=seed)

    fig, axes = plt.subplots(sample_count, 2, figsize=(9, 3.6 * sample_count))
    axes = np.atleast_2d(axes)
    for i in range(sample_count):
        m = prior.mean.copy()
        rng.normal(1.0, noise)
        prior.sample(noise, m)
        state_vec, qoi = solve_state_at_parameter_control(control_model, m, control_np)
        m_fun = scalarize_for_plot(vector_to_function(V_parameter, m), V_parameter_scalar)
        state_fun = scalarize_for_plot(vector_to_function(V_state, state_vec), V_state_scalar)
        for ax, fun, title in zip(
            axes[i],
            [m_fun, state_fun],
            [f"{saa_model_name} sample {i + 1} parameter", f"|u| at sample, QoI={qoi:.3e}"],
        ):
            artist = plot_on_axes(fun, ax)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(title)
            plt.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{saa_model_name}_parameter_solution_samples.png"), dpi=180)
    plt.close(fig)


def save_solution_vs_linear_initial_plots(results, linear_initial_control_np, control_model, prior, Vh, save_dir):
    _, init_state_vec = solve_state_at_control(control_model, prior, linear_initial_control_np)
    init_u_x, init_u_y = state_components_for_plot(Vh[soupy.STATE], init_state_vec)

    fig, axes = plt.subplots(len(MODEL_ORDER), 4, figsize=(17, 3.6 * len(MODEL_ORDER)))
    axes = np.atleast_2d(axes)
    for i, model_name in enumerate(MODEL_ORDER):
        _, opt_state_vec = solve_state_at_control(control_model, prior, optimal_control_np(results[model_name]))
        opt_u_x, opt_u_y = state_components_for_plot(Vh[soupy.STATE], opt_state_vec)
        for ax, fun, title in zip(
            axes[i],
            [opt_u_x, opt_u_y, init_u_x, init_u_y],
            [f"{model_name} u_x(z*)", f"{model_name} u_y(z*)", "linear initial u_x", "linear initial u_y"],
        ):
            artist = plot_on_axes(fun, ax)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(title)
            plt.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "optimal_vs_linear_initial_solutions.png"), dpi=180)
    plt.close(fig)



def save_cvar_tail_parameter_solution_plots(results, saa_model_name, control_model, prior, Vh, save_dir, sample_size, cvar_value, sample_count=3, seed=1):
    if saa_model_name not in results:
        return
    V_parameter = Vh[soupy.PARAMETER]
    V_state = Vh[soupy.STATE]
    V_parameter_scalar = dl.FunctionSpace(V_parameter.mesh(), "CG", 1)
    V_state_scalar = dl.FunctionSpace(V_state.mesh(), "CG", 1)
    control_np = optimal_control_np(results[saa_model_name])
    noise = dl.Vector(V_parameter.mesh().mpi_comm())
    prior.init_vector(noise, "noise")
    rng = hp.Random(seed=seed)
    tail_samples = []

    for _ in range(int(sample_size)):
        m = prior.mean.copy()
        rng.normal(1.0, noise)
        prior.sample(noise, m)
        state_vec, qoi = solve_state_at_parameter_control(control_model, m, control_np)
        if qoi > cvar_value:
            tail_samples.append((qoi, m.copy(), state_vec.copy()))
            if len(tail_samples) >= sample_count:
                break

    if not tail_samples:
        return

    fig, axes = plt.subplots(len(tail_samples), 2, figsize=(9, 3.6 * len(tail_samples)))
    axes = np.atleast_2d(axes)
    for i, (qoi, m, state_vec) in enumerate(tail_samples):
        m_fun = scalarize_for_plot(vector_to_function(V_parameter, m), V_parameter_scalar)
        state_fun = scalarize_for_plot(vector_to_function(V_state, state_vec), V_state_scalar)
        for ax, fun, title in zip(
            axes[i],
            [m_fun, state_fun],
            [f"tail sample {i + 1}: QoI={qoi:.3e} > CVaR={cvar_value:.3e}", "|u| at tail sample"],
        ):
            artist = plot_on_axes(fun, ax)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(title)
            plt.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{saa_model_name}_tail_parameter_solution_samples.png"), dpi=180)
    plt.close(fig)


def save_optimal_field_plots(results, control_model, prior, Vh, save_dir):
    V_control = Vh[soupy.CONTROL]
    V_state = Vh[soupy.STATE]
    V_state_scalar = dl.FunctionSpace(V_state.mesh(), "CG", 1)

    overview_payload = []
    for model_name in MODEL_ORDER:
        control_vec, state_vec = solve_state_at_control(control_model, prior, results[model_name]["control_opt_np"])
        control_fun = vector_to_function(V_control, control_vec)
        state_fun = scalarize_for_plot(vector_to_function(V_state, state_vec), V_state_scalar)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, fun, title in zip(axes, [control_fun, state_fun], [f"{model_name} z*", f"{model_name} |u(z*)|"]):
            artist = plot_on_axes(fun, ax)
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            plt.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{model_name}_optimal_fields.png"), dpi=180)
        plt.close(fig)
        overview_payload.append((model_name, control_fun, state_fun))

    n_models = len(overview_payload)
    fig, axes = plt.subplots(n_models, 2, figsize=(10, 3.6 * n_models))
    if n_models == 1:
        axes = np.array([axes])
    for j, title in enumerate(["optimal control z*", "state |u(z*)|"]):
        axes[0, j].set_title(title)
    for i, (model_name, control_fun, state_fun) in enumerate(overview_payload):
        for j, fun in enumerate([control_fun, state_fun]):
            ax = axes[i, j]
            plot_on_axes(fun, ax)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[i, 0].set_ylabel(model_name, rotation=90, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "optimal_fields_overview.png"), dpi=180)
    plt.close(fig)



def state_components_for_plot(V_state, state_vector):
    state_fun = vector_to_function(V_state, state_vector)
    components = state_fun.split(deepcopy=True)
    if len(components) < 2:
        raise ValueError("Expected a vector-valued hyperelasticity state with at least two components.")
    return components[0], components[1]


def save_optimal_state_component_plots(results, control_model, prior, Vh, save_dir):
    payload = []
    for model_name in MODEL_ORDER:
        z_np = optimal_control_np(results[model_name])
        _, state_vec = solve_state_at_control(control_model, prior, z_np)
        u_x, u_y = state_components_for_plot(Vh[soupy.STATE], state_vec)
        payload.append((model_name, u_x, u_y))

    fig, axes = plt.subplots(len(payload), 2, figsize=(9, 3.6 * len(payload)))
    axes = np.atleast_2d(axes)
    for i, (model_name, u_x, u_y) in enumerate(payload):
        for ax, fun, title in zip(axes[i], [u_x, u_y], [f"{model_name} u_x(z*)", f"{model_name} u_y(z*)"]):
            artist = plot_on_axes(fun, ax)
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            plt.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "optimal_state_components.png"), dpi=180)
    plt.close(fig)


def save_optimal_deformed_material_plots(results, control_model, prior, Vh, save_dir):
    V_parameter = Vh[soupy.PARAMETER]
    parameter_fun = vector_to_function(V_parameter, prior.mean)

    payload = []
    for model_name in MODEL_ORDER:
        z_np = optimal_control_np(results[model_name])
        _, state_vec = solve_state_at_control(control_model, prior, z_np)
        state_fun = vector_to_function(Vh[soupy.STATE], state_vec)
        payload.append((model_name, state_fun))

    fig, axes = plt.subplots(len(payload), 2, figsize=(12, 4.2 * len(payload)))
    axes = np.atleast_2d(axes)
    for i, (model_name, state_fun) in enumerate(payload):
        plt.sca(axes[i, 0])
        artist = dl.plot(parameter_fun, mode="color", cmap="turbo", shading="gouraud")
        axes[i, 0].set_title(f"{model_name} parameter mean")
        axes[i, 0].set_xlabel("X_1")
        axes[i, 0].set_ylabel("X_2")
        plt.colorbar(artist, ax=axes[i, 0], pad=0.03, fraction=0.05, aspect=20)

        plt.sca(axes[i, 1])
        artist = dl.plot(state_fun, mode="displacement", cmap="turbo", shading="gouraud")
        axes[i, 1].set_title(f"{model_name} deformed shape x = X + u(X)")
        axes[i, 1].set_xlabel("X_1")
        axes[i, 1].set_ylabel("X_2")
        cbar = plt.colorbar(artist, ax=axes[i, 1], pad=0.03, fraction=0.05, aspect=20)
        cbar.set_label("||u||_2", rotation=0, labelpad=15)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "optimal_deformed_materials.png"), dpi=180)
    plt.close(fig)

def save_optimal_pde_solution_plot(results, control_model, prior, Vh, save_dir):
    V_state = Vh[soupy.STATE]
    V_state_scalar = dl.FunctionSpace(V_state.mesh(), "CG", 1)

    payload = []
    for model_name in MODEL_ORDER:
        _, state_vec = solve_state_at_control(control_model, prior, results[model_name]["control_opt_np"])
        state_fun = scalarize_for_plot(vector_to_function(V_state, state_vec), V_state_scalar)
        payload.append((model_name, state_fun))

    ncols = 3
    nrows = int(np.ceil(len(payload) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (model_name, state_fun) in zip(axes.ravel(), payload):
        ax.axis("on")
        artist = plot_on_axes(state_fun, ax)
        ax.set_title(f"{model_name} |u(z*)|")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "optimal_pde_solutions.png"), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare Taylor models against CVaR SAA truth on hyperelasticity")
    parser.add_argument("--cvar-beta", "--beta", dest="cvar_beta", type=float, default=0.95, help="CVaR confidence level")
    parser.add_argument("--n-tr", type=int, default=50, help="Number of Hessian modes for quadratic models")
    parser.add_argument("--n-mix", type=int, default=11, help="Number of mixture components")
    parser.add_argument("--quadratic-cvar-n-mc", type=int, default=5000, help="MC samples for quadratic CVaR surrogates")
    parser.add_argument("--truth-saa-samples", "--saa-samples", dest="truth_saa_samples", type=int, default=100000)
    parser.add_argument("--saa-seed", type=int, default=1)
    parser.add_argument("--qoi-type", type=str, default="virtual_work",
                        choices=["all", "stiffness", "point", "virtual_work"])
    parser.add_argument("--penalty", type=float, default=1.0)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--maxiter-saa", type=int, default=500)
    parser.add_argument("--nx", type=int, default= 64)
    parser.add_argument("--ny", type=int, default= 16)
    parser.add_argument("--nz", type=int, default=8)
    parser.add_argument("--lx", type=float, default=2.0)
    parser.add_argument("--ly", type=float, default=0.5)
    parser.add_argument("--lz", type=float, default=0.25)
    parser.add_argument("--geometry-dim", type=int, default=2, choices=[2, 3])
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--save-dir", type=str, default="results_compare_cvar_models")
    parser.add_argument("--log-file", type=str, default="terminal_output.txt")
    parser.add_argument("--bound-lb", type=float, default=0.0)
    parser.add_argument("--bound-ub", type=float, default=1.0)
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
            print("=" * 78)
            print("Hyperelasticity CVaR Model Comparison:")
            print("  linear / quadratic / mixture_linear_kle / mixture_linear_hep")
            print(
                "  mixture_quadratic_kle / mixture_quadratic_hep / "
                "saa_1 / saa_10 / saa_100 / saa_200 / saa_500 / saa_1000 / saa_10000"
            )
            print("  serial: linear / quadratic / saa_1 / saa_10")
            print("  parallel on all ranks: mixture_* / saa_100 / saa_200 / saa_500 / saa_1000 / saa_10000")
            print("=" * 78)
            print(f"Ground-truth exact-CVaR sample count: {args.truth_saa_samples}")
            print(
                f"cvar_beta={args.cvar_beta}, n_tr={args.n_tr}, n_mix={args.n_mix}, "
                f"quadratic_cvar_n_mc={args.quadratic_cvar_n_mc}"
            )
            print(f"penalty={args.penalty}")
            print(f"mesh={args.nx}x{args.ny}" + (f"x{args.nz}" if args.geometry_dim == 3 else ""))
            print(
                "Explicit continuation for quadratic and SAA models: "
                + " -> ".join(f"{eps:.0e}" for eps in EXPLICIT_CVAR_CONTINUATION_LEVELS)
            )
            print(f"terminal log file: {log_path}")
            print("=" * 78)
            sys.stdout.flush()

        _, Vh, control_model, prior, penalty = setup_problem(args, comm_mesh)
        if rank == 0:
            save_young_modulus_moment_plots(prior, Vh, args.save_dir)
        control0_np = zero_control_np(control_model)
        base_control_np = np.full_like(control0_np, 0.5)
        base_t = 0.5
        linear_base_t = 0.0
        args.control_dim = len(control0_np)
        linear_init_control_np = np.array(control0_np, copy=True)
        linear_init_t = 0.0
        quadratic_init_control_np = np.array(control0_np, copy=True)
        quadratic_init_t = 0.0

        results: Dict[str, Dict] = {}
        for model_name in MODEL_ORDER:
            approx_cost = None
            x0_np = None
            approx_init = None
            init_t = None
            source = warm_start_source(model_name)
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
                if source == "linear":
                    init_control_np = linear_init_control_np
                    init_scalar = linear_init_t
                elif source == "quadratic":
                    init_control_np = quadratic_init_control_np
                    init_scalar = quadratic_init_t
                else:
                    init_control_np = base_control_np
                    init_scalar = linear_base_t if model_name == "linear" else base_t
                init_control_np = MPI.COMM_WORLD.bcast(init_control_np if rank == 0 else None, root=0)
                x0_np, approx_init, init_t = initial_model_point_at_control(
                    approx_cost,
                    init_control_np,
                    scalar=init_scalar,
                )
            elif rank == 0:
                approx_cost = make_cvar_cost(
                    model_name, control_model, prior, penalty, args, epsilon_override=epsilon_override
                )
                if source == "linear":
                    init_control_np = linear_init_control_np
                    init_scalar = linear_init_t
                elif source == "quadratic":
                    init_control_np = quadratic_init_control_np
                    init_scalar = quadratic_init_t
                else:
                    init_control_np = base_control_np
                    init_scalar = linear_base_t if model_name == "linear" else base_t
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
                if source == "linear":
                    print(f"  [{model_name:20s}] initial guess: linear optimum z* with t initialized from linear surrogate VaR")
                elif source == "quadratic":
                    print(f"  [{model_name:20s}] initial guess: quadratic optimum z* with t initialized from quadratic model t*")
                elif model_name == "linear":
                    print(f"  [{model_name:20s}] initial guess: 0.5 control with t=0.0")
                else:
                    print(f"  [{model_name:20s}] initial guess: 0.5 control with t=0.5")
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
            if model_name == "quadratic":
                quadratic_init_control_np = np.array(res["control_opt_np"], copy=True)
                quadratic_init_t = 0.0 if res["t_opt"] is None else float(res["t_opt"])

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
                save_iteration_csv(os.path.join(args.save_dir, f"{model_name}_iteration_metrics.csv"), results[model_name]["iter_records"])

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

            payload = {
                "config": vars(args),
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
                json.dump(payload, f, indent=2)

            plot_curves(results, args.save_dir)
            save_optimal_field_plots(results, control_model, prior, Vh, args.save_dir)
            save_optimal_pde_solution_plot(results, control_model, prior, Vh, args.save_dir)
            save_optimal_state_component_plots(results, control_model, prior, Vh, args.save_dir)
            save_optimal_deformed_material_plots(results, control_model, prior, Vh, args.save_dir)
            save_saa_parameter_solution_sample_plots(results, "saa_10000", control_model, prior, Vh, args.save_dir)
            save_solution_vs_linear_initial_plots(results, results["linear"].get("control_init_np", control0_np), control_model, prior, Vh, args.save_dir)
            save_cvar_tail_parameter_solution_plots(
                results, "saa_10000", control_model, prior, Vh, args.save_dir,
                sample_size=10000, cvar_value=results["saa_10000"]["true_opt_cvar"], seed=args.saa_seed,
            )

            with open(os.path.join(args.save_dir, "timing_comparison.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["model", "total_iterations", "avg_iter_time_sec"])
                writer.writerows([[m, results[m]["iter_count"], results[m]["avg_iter_time_sec"]] for m in MODEL_ORDER])

            print("\n" + "-" * 78)
            print("Summary (ground-truth exact CVaR from sampled QoI list, evaluated at z0 and z*)")
            print("-" * 78)
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
            print("-" * 78)
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
