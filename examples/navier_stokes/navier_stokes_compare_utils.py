import os
import sys

import dolfin as dl
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.environ.get("HIPPYLIB_PATH", ""))
sys.path.append(os.environ.get("../../", ""))

import hippylib as hp
import soupy

from setupNavierStokesProblem import navier_stokes_problem_settings, setup_navier_stokes_problem


def build_driver_settings(args):
    settings = navier_stokes_problem_settings()
    settings["nu"] = args.nu
    settings["continuation"] = bool(args.continuation)
    settings["stabilization"] = bool(args.stabilization)
    settings["nitche"] = bool(args.nitche)
    settings["gamma"] = args.gamma
    settings["delta"] = args.delta
    settings["mean_velocity"] = args.mean_velocity
    settings["mesh_base_directory"] = args.mesh_base_directory
    settings["mesh_resolution"] = args.mesh_resolution
    settings["mesh_format"] = args.mesh_format
    settings["qoi_type"] = args.qoi_type
    settings["penalty_alpha"] = args.penalty
    return settings


def setup_problem(args, comm_mesh):
    del comm_mesh
    settings = build_driver_settings(args)
    return setup_navier_stokes_problem(settings)


def vector_to_function(function_space, vector):
    function = dl.Function(function_space)
    function.vector().zero()
    function.vector().axpy(1.0, vector)
    return function


def control_field_magnitude(control_model, control_vector):
    residual_handler = control_model.problem.ns_residual
    Vh = control_model.problem.Vh
    control_fun = hp.vector2Function(control_vector, Vh[soupy.CONTROL])
    phi_z = residual_handler.control_to_obstacle_velocity(control_fun)
    scalar_space = dl.FunctionSpace(Vh[soupy.STATE].mesh(), "CG", 1)
    return dl.project(dl.sqrt(dl.inner(phi_z, phi_z)), scalar_space)


def state_velocity_magnitude(V_state, state_vector):
    state_fun = vector_to_function(V_state, state_vector)
    velocity_fun, _ = state_fun.split(deepcopy=True)
    scalar_space = dl.FunctionSpace(V_state.mesh(), "CG", 1)
    return dl.project(dl.sqrt(dl.inner(velocity_fun, velocity_fun)), scalar_space)


def plot_on_axes(function, ax):
    plt.sca(ax)
    return dl.plot(function)


def solve_state_at_control(control_model, prior, control_np):
    x = control_model.generate_vector("ALL")
    x[soupy.PARAMETER].zero()
    x[soupy.PARAMETER].axpy(1.0, prior.mean)
    x[soupy.CONTROL].set_local(control_np)
    x[soupy.CONTROL].apply("")
    control_model.solveFwd(x[soupy.STATE], x)
    return x[soupy.CONTROL].copy(), x[soupy.STATE].copy()


def save_optimal_field_plots(results, model_order, control_model, prior, Vh, save_dir):
    overview_payload = []

    for model_name in model_order:
        result = results[model_name]
        control_key = "z_opt_np" if "z_opt_np" in result else "control_opt_np"
        control_vec, state_vec = solve_state_at_control(control_model, prior, result[control_key])

        control_field = control_field_magnitude(control_model, control_vec)
        state_field = state_velocity_magnitude(Vh[soupy.STATE], state_vec)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, fun, title in zip(
            axes,
            [control_field, state_field],
            [f"{model_name} |phi(z*)|", f"{model_name} |v(z*)|"],
        ):
            artist = plot_on_axes(fun, ax)
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            plt.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{model_name}_optimal_fields.png"), dpi=180)
        plt.close(fig)
        overview_payload.append((model_name, control_field, state_field))

    n_models = len(overview_payload)
    fig, axes = plt.subplots(n_models, 2, figsize=(10, 3.6 * n_models))
    if n_models == 1:
        axes = np.array([axes])
    for j, title in enumerate(["optimal control |phi(z*)|", "velocity magnitude |v(z*)|"]):
        axes[0, j].set_title(title)
    for i, (model_name, control_field, state_field) in enumerate(overview_payload):
        for j, fun in enumerate([control_field, state_field]):
            ax = axes[i, j]
            plot_on_axes(fun, ax)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[i, 0].set_ylabel(model_name, rotation=90, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "optimal_fields_overview.png"), dpi=180)
    plt.close(fig)


def save_optimal_pde_solution_plot(results, model_order, control_model, prior, Vh, save_dir):
    payload = []
    for model_name in model_order:
        result = results[model_name]
        control_key = "z_opt_np" if "z_opt_np" in result else "control_opt_np"
        _, state_vec = solve_state_at_control(control_model, prior, result[control_key])
        state_field = state_velocity_magnitude(Vh[soupy.STATE], state_vec)
        payload.append((model_name, state_field))

    ncols = 3
    nrows = int(np.ceil(len(payload) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax in axes.ravel():
        ax.axis("off")

    for ax, (model_name, state_field) in zip(axes.ravel(), payload):
        ax.axis("on")
        artist = plot_on_axes(state_field, ax)
        ax.set_title(f"{model_name} |v(z*)|")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "optimal_pde_solutions.png"), dpi=180)
    plt.close(fig)
