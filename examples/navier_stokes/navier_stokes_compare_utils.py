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


def scalarize_parameter_for_plot(function, scalar_space):
    if function.function_space().num_sub_spaces() > 0:
        return dl.project(dl.sqrt(dl.inner(function, function)), scalar_space)
    return function


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
        m_fun = scalarize_parameter_for_plot(vector_to_function(V_parameter, m), V_parameter_scalar)
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
    control_np = np.asarray(control_np)
    expected_size = x[soupy.CONTROL].local_size()
    if control_np.size != expected_size:
        raise ValueError(
            f"Expected a control vector with {expected_size} local entries, "
            f"got {control_np.size}. Pass result['control_opt_np'], not an "
            "augmented CVaR vector such as result['z_opt_np']."
        )
    x[soupy.CONTROL].set_local(control_np)
    x[soupy.CONTROL].apply("")
    control_model.solveFwd(x[soupy.STATE], x)
    return x[soupy.CONTROL].copy(), x[soupy.STATE].copy()


def optimal_control_np(result):
    if "control_opt_np" in result:
        return result["control_opt_np"]
    return result["z_opt_np"]


def boundary_control_profile(control_np, control_bases, side, num_points=300):
    coeffs = np.asarray(control_np, dtype=float)
    n_each_side = len(control_bases) // 2
    if coeffs.size != len(control_bases):
        raise ValueError(f"Expected {len(control_bases)} control coefficients, received {coeffs.size}.")
    if side == "top":
        side_coeffs = coeffs[:n_each_side]
        side_bases = control_bases[:n_each_side]
    elif side == "bottom":
        side_coeffs = coeffs[n_each_side:]
        side_bases = control_bases[n_each_side:]
    else:
        raise ValueError(f"Unknown control side: {side}")

    t = np.linspace(0.0, 1.0, num_points)
    radial_velocity = np.zeros_like(t)
    for coeff, basis in zip(side_coeffs, side_bases):
        radial_velocity += float(coeff) * basis.bSpline(t)
    edge_length = float(np.linalg.norm(side_bases[0].x_end - side_bases[0].x_start))
    arclength = edge_length * t
    return arclength, radial_velocity


def plot_boundary_control_profiles(control_np, control_bases, axes, model_name=None):
    max_abs = 0.0
    profiles = []
    for side in ["top", "bottom"]:
        arclength, radial_velocity = boundary_control_profile(control_np, control_bases, side)
        max_abs = max(max_abs, float(np.max(np.abs(radial_velocity))))
        profiles.append((side, arclength, radial_velocity))
    if max_abs == 0.0:
        max_abs = 1.0

    for ax, (side, arclength, radial_velocity) in zip(axes, profiles):
        ax.axhline(0.0, color="0.35", linewidth=0.8)
        ax.plot(arclength, radial_velocity, color="tab:blue", linewidth=2.0)
        title_prefix = "" if model_name is None else f"{model_name} "
        ax.set_title(f"{title_prefix}{side} radial velocity")
        ax.set_xlabel("arclength along boundary")
        ax.set_ylabel("radial velocity")
        ax.set_ylim(-1.05 * max_abs, 1.05 * max_abs)
        ax.grid(True, alpha=0.3)


def save_optimal_field_plots(results, model_order, control_model, prior, Vh, save_dir):
    overview_payload = []
    control_bases = control_model.problem.ns_residual.control_bases

    for model_name in model_order:
        result = results[model_name]
        control_np = optimal_control_np(result)
        _, state_vec = solve_state_at_control(control_model, prior, control_np)
        state_field = state_velocity_magnitude(Vh[soupy.STATE], state_vec)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        plot_boundary_control_profiles(control_np, control_bases, axes[:2], model_name=model_name)
        artist = plot_on_axes(state_field, axes[2])
        axes[2].set_title(f"{model_name} |v(z*)|")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        plt.colorbar(artist, ax=axes[2], fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{model_name}_optimal_fields.png"), dpi=180)
        plt.close(fig)
        overview_payload.append((model_name, np.array(control_np, copy=True), state_field))

    n_models = len(overview_payload)
    fig, axes = plt.subplots(n_models, 3, figsize=(14, 3.6 * n_models))
    if n_models == 1:
        axes = np.array([axes])
    for j, title in enumerate(["top radial velocity", "bottom radial velocity", "velocity magnitude |v(z*)|"]):
        axes[0, j].set_title(title)
    for i, (model_name, control_np, state_field) in enumerate(overview_payload):
        plot_boundary_control_profiles(control_np, control_bases, axes[i, :2])
        plot_on_axes(state_field, axes[i, 2])
        for ax in axes[i, :]:
            ax.set_xticks([])
            ax.set_yticks([])
        axes[i, 0].set_ylabel(model_name, rotation=90, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "optimal_fields_overview.png"), dpi=180)
    plt.close(fig)



def velocity_components_for_plot(V_state, state_vector):
    state_fun = vector_to_function(V_state, state_vector)
    velocity_fun, _ = state_fun.split(deepcopy=True)
    components = velocity_fun.split(deepcopy=True)
    if len(components) < 2:
        raise ValueError("Expected a two-component Navier-Stokes velocity field.")
    return components[0], components[1]


def save_optimal_state_component_plots(results, model_order, control_model, prior, Vh, save_dir):
    payload = []
    for model_name in model_order:
        result = results[model_name]
        _, state_vec = solve_state_at_control(control_model, prior, optimal_control_np(result))
        v_x, v_y = velocity_components_for_plot(Vh[soupy.STATE], state_vec)
        payload.append((model_name, v_x, v_y))

    fig, axes = plt.subplots(len(payload), 2, figsize=(9, 3.6 * len(payload)))
    axes = np.atleast_2d(axes)
    for i, (model_name, v_x, v_y) in enumerate(payload):
        for ax, fun, title in zip(axes[i], [v_x, v_y], [f"{model_name} v_x(z*)", f"{model_name} v_y(z*)"]):
            artist = plot_on_axes(fun, ax)
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            plt.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "optimal_state_components.png"), dpi=180)
    plt.close(fig)

def save_optimal_pde_solution_plot(results, model_order, control_model, prior, Vh, save_dir):
    payload = []
    for model_name in model_order:
        result = results[model_name]
        _, state_vec = solve_state_at_control(control_model, prior, optimal_control_np(result))
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
