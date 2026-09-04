"""Compare Taylor and SAA models on Navier-Stokes control."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict

_SOUPY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.environ.get("HIPPYLIB_PATH", ""))
sys.path.append(_SOUPY_ROOT)

import scipy.optimize
from mpi4py import MPI

import soupy
from navier_stokes_compare_utils import save_optimal_field_plots, save_optimal_pde_solution_plot, save_optimal_state_component_plots, save_saa_parameter_solution_sample_plots, save_solution_vs_linear_initial_plots, setup_problem
from navier_stokes_driver_common import (
    TeeStream,
    evaluate_cost,
    evaluate_true_cost_stats,
    make_saa_cost,
    model_uses_world_parallel,
    optimize_with_tracking,
    plot_curves,
    relative_error,
    save_iteration_csv,
    save_summary_csv,
)
from soupy.approximations.taylor import (
    TaylorLinearControlCostFunctional,
    TaylorMixtureLinearControlCostFunctional,
    TaylorMixtureQuadraticControlCostFunctional,
    TaylorQuadraticControlCostFunctional,
)


MODEL_ORDER = [
    "linear",
    "quadratic",
    "mixture_linear_kle",
    "mixture_linear_hep",
    "mixture_quadratic_kle",
    "mixture_quadratic_hep",
    "saa_10",
    "saa_20",
    "saa_50",
    "saa_100",
    "saa_200",
    "saa_500",
]

MODEL_COLORS = {
    "linear": "tab:blue",
    "quadratic": "tab:orange",
    "mixture_linear_kle": "tab:green",
    "mixture_linear_hep": "tab:olive",
    "mixture_quadratic_kle": "tab:red",
    "mixture_quadratic_hep": "tab:brown",
    "saa_10": "#17becf",
    "saa_20": "#bcbd22",
    "saa_50": "#9467bd",
    "saa_100": "#7f7f7f",
    "saa_200": "#e377c2",
    "saa_500": "#8c564b",
}


def print_driver_banner(args, log_path):
    print("=" * 78)
    print("Navier-Stokes Taylor Model Comparison:")
    print("  linear / quadratic / mixture_linear_kle / mixture_linear_hep / mixture_quadratic_kle / mixture_quadratic_hep")
    print("  saa_10 / saa_20 / saa_50 / saa_100 / saa_200 / saa_500")
    print("  serial: linear / quadratic / saa_10")
    print("  parallel on all ranks: mixture_* / saa_20 / saa_50 / saa_100 / saa_200 / saa_500")
    print("=" * 78)
    print(f"Ground-truth SAA sample count: {args.truth_saa_samples}")
    print(f"beta={args.beta}, n_tr={args.n_tr}, n_mix={args.n_mix}, penalty={args.penalty}")
    print(
        f"mesh_resolution={args.mesh_resolution}, mesh_format={args.mesh_format}, "
        f"nu={args.nu}, gamma={args.gamma}, delta={args.delta}, mean_velocity={args.mean_velocity}"
    )
    print(
        f"continuation={args.continuation}, stabilization={args.stabilization}, "
        f"nitche={args.nitche}"
    )
    print(f"terminal log file: {log_path}")
    print("=" * 78)
    sys.stdout.flush()


def print_initial_summary(model_name, approx_init, true_init, true_init_mean, true_init_var, init_rel_err):
    print(f"\nOptimizing {model_name} with L-BFGS-B ...")
    print(f"  [{model_name:20s}] initial guess: zero control")
    print(
        f"  [{model_name:20s}] initial: "
        f"J_model(init)={approx_init:.6e}, J_true(init)={true_init:.6e}, "
        f"mean_qoi(init)={true_init_mean:.6e}, var_qoi(init)={true_init_var:.6e}, "
        f"rel_err(init)={init_rel_err:.3e}"
    )
    sys.stdout.flush()


def print_optimal_summary(model_name, res):
    print(
        f"  [{model_name:20s}] optimal: "
        f"J_model(z*)={res['approx_opt']:.6e}, J_true(z*)={res['true_opt']:.6e}, "
        f"mean_qoi(z*)={res['true_opt_mean']:.6e}, var_qoi(z*)={res['true_opt_var']:.6e}, "
        f"rel_err(z*)={res['opt_rel_err']:.3e}"
    )
    sys.stdout.flush()


def print_final_summary(results, log_path, save_dir):
    print("\n" + "-" * 78)
    print("Summary (ground-truth SAA objective evaluated at z0 and z*)")
    print("-" * 78)
    for model_name in MODEL_ORDER:
        rr = results[model_name]
        print(
            f"{model_name:20s} | nit={rr['iter_count']:3d} | "
            f"avg_iter_time={rr['avg_iter_time_sec']:8.2f}s | "
            f"J_model(z*)={rr['approx_opt']:.6e} | "
            f"init rel_err={rr['init_rel_err']:.3e} | "
            f"opt rel_err={rr['opt_rel_err']:.3e} | "
            f"J_true(z*)={rr['true_opt']:.6e}"
        )
    print("-" * 78)
    print(f"All outputs written to: {save_dir}")
    print(f"Terminal outputs saved to: {log_path}")
    sys.stdout.flush()


def make_taylor_cost(model_name, control_model, prior, penalty, args):
    if model_name.startswith("saa_"):
        sample_size = int(model_name.split("_", 1)[1])
        return make_saa_cost(
            control_model,
            prior,
            penalty,
            beta=args.beta,
            sample_size=sample_size,
            seed=args.saa_seed,
            comm_sampler=MPI.COMM_WORLD if sample_size >= 11 else MPI.COMM_SELF,
        )

    if model_name == "linear":
        return TaylorLinearControlCostFunctional(
            control_model,
            prior,
            penalty,
            {"beta": args.beta, "correction": False, "N_mc": 0, "verbose": args.verbose},
        )

    if model_name == "quadratic":
        return TaylorQuadraticControlCostFunctional(
            control_model,
            prior,
            penalty,
            {"beta": args.beta, "N_tr": args.n_tr, "correction": False, "N_mc": 0, "verbose": args.verbose},
        )

    if model_name == "mixture_linear_kle":
        return TaylorMixtureLinearControlCostFunctional(
            control_model,
            prior,
            penalty,
            {"beta": args.beta, "N_mix": args.n_mix, "direction": "kle", "verbose": args.verbose},
            comm_sampler=MPI.COMM_WORLD,
        )

    if model_name == "mixture_linear_hep":
        return TaylorMixtureLinearControlCostFunctional(
            control_model,
            prior,
            penalty,
            {"beta": args.beta, "N_mix": args.n_mix, "direction": "hep", "verbose": args.verbose},
            comm_sampler=MPI.COMM_WORLD,
        )

    if model_name == "mixture_quadratic_kle":
        return TaylorMixtureQuadraticControlCostFunctional(
            control_model,
            prior,
            penalty,
            {"beta": args.beta, "N_mix": args.n_mix, "direction": "kle", "N_tr": args.n_tr, "N_mc": 0, "verbose": args.verbose},
            comm_sampler=MPI.COMM_WORLD,
        )

    if model_name == "mixture_quadratic_hep":
        return TaylorMixtureQuadraticControlCostFunctional(
            control_model,
            prior,
            penalty,
            {"beta": args.beta, "N_mix": args.n_mix, "direction": "hep", "N_tr": args.n_tr, "N_mc": 0, "verbose": args.verbose},
            comm_sampler=MPI.COMM_WORLD,
        )

    raise ValueError(f"Unknown model name: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Compare Taylor models against SAA truth on Navier-Stokes")
    parser.add_argument("--beta", type=float, default=1.0, help="Variance weight beta")
    parser.add_argument("--n-tr", type=int, default=50, help="Number of Hessian modes for quadratic models")
    parser.add_argument("--n-mix", type=int, default=11, help="Number of mixture components")
    parser.add_argument("--truth-saa-samples", "--saa-samples", dest="truth_saa_samples", type=int, default=5000)
    parser.add_argument("--saa-seed", type=int, default=1)
    parser.add_argument("--qoi-type", type=str, default="velocity_tracking", choices=["velocity_tracking"])
    parser.add_argument("--penalty", type=float, default=1.0, help="Penalty coefficient on ||phi(z)||_L2^2")
    parser.add_argument("--maxiter", type=int, default=600)
    parser.add_argument("--maxiter-saa", type=int, default=600)
    parser.add_argument("--mesh-base-directory", type=str, default="./")
    parser.add_argument("--mesh-resolution", type=str, default="medium")
    parser.add_argument("--mesh-format", type=str, default="xdmf")
    parser.add_argument("--nu", type=float, default=5e-3)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=5.0)
    parser.add_argument("--mean-velocity", type=float, default=1.0)
    parser.add_argument("--continuation", action="store_true", help="Use PDE viscosity continuation")
    parser.add_argument("--no-continuation", dest="continuation", action="store_false")
    parser.add_argument("--stabilization", action="store_true")
    parser.add_argument("--no-stabilization", dest="stabilization", action="store_false")
    parser.add_argument("--nitche", action="store_true")
    parser.add_argument("--no-nitche", dest="nitche", action="store_false")
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--save-dir", type=str, default="results_compare_taylor_models")
    parser.add_argument("--log-file", type=str, default="terminal_output.txt")
    parser.add_argument("--bound-lb", type=float, default=-2.0)
    parser.add_argument("--bound-ub", type=float, default=2.0)
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.set_defaults(continuation=True, stabilization=False, nitche=True)
    args = parser.parse_args()

    rank = MPI.COMM_WORLD.Get_rank()
    comm_mesh = MPI.COMM_SELF
    os.makedirs(args.save_dir, exist_ok=True)

    log_path = args.log_file if os.path.isabs(args.log_file) else os.path.join(args.save_dir, args.log_file)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = None
    if rank == 0:
        log_file = open(log_path, "w", buffering=1)
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)

    try:
        if rank == 0:
            print_driver_banner(args, log_path)

        problem = setup_problem(args, comm_mesh)
        Vh = problem["Vh"]
        control_model = problem["control_model"]
        prior = problem["prior"]
        penalty = problem["penalty"]
        control0_np = control_model.generate_vector(soupy.CONTROL).get_local()
        bounds = scipy.optimize.Bounds(lb=args.bound_lb, ub=args.bound_ub)

        results: Dict[str, Dict] = {}
        for model_name in MODEL_ORDER:
            approx_cost = None
            z0_np = None
            approx_init = None
            run_parallel_model = model_uses_world_parallel(model_name)

            if run_parallel_model:
                approx_cost = make_taylor_cost(model_name, control_model, prior, penalty, args)
                z0_np = approx_cost.generate_vector(soupy.CONTROL).get_local()
                approx_init = evaluate_cost(approx_cost, z0_np)
            elif rank == 0:
                approx_cost = make_taylor_cost(model_name, control_model, prior, penalty, args)
                z0_np = approx_cost.generate_vector(soupy.CONTROL).get_local()
                approx_init = evaluate_cost(approx_cost, z0_np)

            z0_np = MPI.COMM_WORLD.bcast(z0_np if rank == 0 else None, root=0)
            approx_init = MPI.COMM_WORLD.bcast(approx_init if rank == 0 else None, root=0)

            truth_cost = make_saa_cost(control_model, prior, penalty, args.beta, args.truth_saa_samples, args.saa_seed, MPI.COMM_WORLD)
            true_init, true_init_mean, true_init_var = evaluate_true_cost_stats(truth_cost, z0_np)
            del truth_cost
            init_rel_err = relative_error(approx_init, true_init)

            if rank == 0:
                print_initial_summary(model_name, approx_init, true_init, true_init_mean, true_init_var, init_rel_err)

            res = optimize_with_tracking(
                model_name,
                approx_cost,
                args,
                rank,
                maxiter=args.maxiter_saa if model_name.startswith("saa_") else args.maxiter,
                bounds=bounds,
                root_only=not run_parallel_model,
            )
            res["approx_init"] = float(approx_init)
            res["true_init"] = float(true_init)
            res["true_init_mean"] = float(true_init_mean)
            res["true_init_var"] = float(true_init_var)
            res["init_rel_err"] = float(init_rel_err)

            truth_cost = make_saa_cost(control_model, prior, penalty, args.beta, args.truth_saa_samples, args.saa_seed, MPI.COMM_WORLD)
            res["true_opt"], res["true_opt_mean"], res["true_opt_var"] = evaluate_true_cost_stats(truth_cost, res["z_opt_np"])
            res["opt_rel_err"] = relative_error(res["approx_opt"], res["true_opt"])
            del truth_cost
            results[model_name] = res
            if approx_cost is not None:
                del approx_cost
            if rank == 0:
                print_optimal_summary(model_name, res)

        if rank == 0:
            for model_name in MODEL_ORDER:
                save_iteration_csv(os.path.join(args.save_dir, f"{model_name}_iteration_metrics.csv"), results[model_name]["iter_records"])

            summary_rows = []
            for model_name in MODEL_ORDER:
                rr = results[model_name]
                summary_rows.append([model_name, rr["success"], rr["iter_count"], rr["nfev"], rr["njev"], rr["avg_iter_time_sec"], rr["approx_init"], rr["true_init"], rr["init_rel_err"], rr["approx_opt"], rr["true_opt"], rr["opt_rel_err"]])
            save_summary_csv(os.path.join(args.save_dir, "summary.csv"), summary_rows)

            with open(os.path.join(args.save_dir, "summary.json"), "w") as f:
                json.dump({
                    "config": vars(args),
                    "problem_settings": problem["settings"],
                    "models": {
                        name: {
                            "success": bool(results[name]["success"]),
                            "nit": int(results[name]["iter_count"]),
                            "nfev": int(results[name]["nfev"]),
                            "njev": int(results[name]["njev"]),
                            "avg_iter_time_sec": float(results[name]["avg_iter_time_sec"]),
                            "initial_model_objective": float(results[name]["approx_init"]),
                            "initial_true_objective": float(results[name]["true_init"]),
                            "initial_rel_error": float(results[name]["init_rel_err"]),
                            "opt_model_objective": float(results[name]["approx_opt"]),
                            "opt_true_objective": float(results[name]["true_opt"]),
                            "opt_rel_error": float(results[name]["opt_rel_err"]),
                        }
                        for name in MODEL_ORDER
                    },
                }, f, indent=2)

            plot_curves(results, MODEL_ORDER, MODEL_COLORS, args.save_dir)
            save_optimal_field_plots(results, MODEL_ORDER, control_model, prior, Vh, args.save_dir)
            save_optimal_pde_solution_plot(results, MODEL_ORDER, control_model, prior, Vh, args.save_dir)
            save_optimal_state_component_plots(results, MODEL_ORDER, control_model, prior, Vh, args.save_dir)
            save_saa_parameter_solution_sample_plots(results, "saa_100", control_model, prior, Vh, args.save_dir)
            save_solution_vs_linear_initial_plots(results, MODEL_ORDER, control0_np, control_model, prior, Vh, args.save_dir)

            with open(os.path.join(args.save_dir, "timing_comparison.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["model", "total_iterations", "avg_iter_time_sec"])
                writer.writerows([[m, results[m]["iter_count"], results[m]["avg_iter_time_sec"]] for m in MODEL_ORDER])

            print_final_summary(results, log_path, args.save_dir)
    finally:
        if rank == 0 and log_file is not None:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()


if __name__ == "__main__":
    main()
