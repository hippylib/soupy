"""Regenerate individual Navier-Stokes domain figures for the mixture-taylor-ouu
paper with horizontal, plot-width colorbars and clean (non-cropped) borders.

This script re-solves the forward PDE once per required field (cheap single
solves) using the optimal control vectors already saved from the original
optimization runs (``*_data.npz``); it does not repeat any optimization.

Run with the ``soupy`` conda environment, from this directory:
    conda run -p /nethome/whao36/.conda/envs/soupy python regenerate_paper_figures.py
"""
from __future__ import annotations

import os
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

import hippylib as hp
import soupy

from navier_stokes_compare_utils import (
    plot_boundary_control_profiles,
    plot_on_axes,
    scalarize_parameter_for_plot,
    setup_problem,
    solve_state_at_control,
    solve_state_at_parameter_control,
    state_velocity_magnitude,
    vector_to_function,
    velocity_components_for_plot,
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

PRETTY_NAMES = {
    "linear": "Linear",
    "quadratic": "Quadratic",
    "mixture_linear_kle": "Mixture linear KLE",
    "mixture_linear_hep": "Mixture linear HEP",
    "mixture_quadratic_kle": "Mixture quadratic KLE",
    "mixture_quadratic_hep": "Mixture quadratic HEP",
}


def pretty_label(model_key):
    if model_key in PRETTY_NAMES:
        return PRETTY_NAMES[model_key]
    if model_key.startswith("saa_"):
        n = model_key.split("_", 1)[1]
        return f"SAA ($N_S = {n}$)"
    return model_key


HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = "/work2/wenbo/mixture-taylor-ouu"
OUT_MAIN = os.path.join(PAPER_DIR, "fig", "main_text")
OUT_SUPP = os.path.join(PAPER_DIR, "fig", "supplementary_material")

TAYLOR_DIR = os.path.join(HERE, "results_compare_taylor_models")  # mean-variance
CVAR_DIR = os.path.join(HERE, "results_compare_cvar_models")  # CVaR

DOMAIN_FIGSIZE = (5.2, 4.4)
CONTROL_PAIR_FIGSIZE = (11.0, 4.4)


def build_args():
    return types.SimpleNamespace(
        nu=0.005, continuation=True, stabilization=False, nitche=True,
        gamma=1.0, delta=5.0, mean_velocity=1.0,
        mesh_base_directory="./", mesh_resolution="medium", mesh_format="xdmf",
        qoi_type="velocity_tracking", penalty=1.0,
    )


def load_control(results_dir, model_name):
    path = os.path.join(results_dir, f"{model_name}_optimal_fields_data.npz")
    return np.load(path)["control_np"]


def save_domain_field(fun, path, title_lines, cbar_format=None):
    fig, ax = plt.subplots(figsize=DOMAIN_FIGSIZE)
    artist = plot_on_axes(fun, ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("\n".join(title_lines))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="7%", pad=0.65)
    cbar = fig.colorbar(artist, cax=cax, orientation="horizontal", format=cbar_format)
    cbar.locator = matplotlib.ticker.MaxNLocator(nbins=5)
    cbar.update_ticks()
    cax.tick_params(labelsize=12)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("wrote", path)


def save_control_pair(control_np, control_bases, path, model_name):
    fig, axes = plt.subplots(1, 2, figsize=CONTROL_PAIR_FIGSIZE)
    plot_boundary_control_profiles(control_np, control_bases, axes, model_name=model_name)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("wrote", path)


def draw_samples(control_model, prior, Vh, control_np, seed, sample_count, out_prefix_param, out_prefix_sol,
                  model_label, sol_title_prefix):
    V_parameter = Vh[soupy.PARAMETER]
    import dolfin as dl
    V_parameter_scalar = dl.FunctionSpace(V_parameter.mesh(), "CG", 1)
    noise = dl.Vector(V_parameter.mesh().mpi_comm())
    prior.init_vector(noise, "noise")
    rng = hp.Random(seed=seed)

    for i in range(sample_count):
        m = prior.mean.copy()
        rng.normal(1.0, noise)
        prior.sample(noise, m)
        m_fun = scalarize_parameter_for_plot(vector_to_function(V_parameter, m), V_parameter_scalar)
        if out_prefix_param is not None:
            save_domain_field(
                m_fun, f"{out_prefix_param}{i + 1}.png",
                [f"{pretty_label(model_label)}", f"sample {i + 1} parameter"],
            )
        if control_np is not None and out_prefix_sol is not None:
            state_vec, qoi = solve_state_at_parameter_control(control_model, m, control_np)
            state_fun = state_velocity_magnitude(Vh[soupy.STATE], state_vec)
            save_domain_field(
                state_fun, f"{out_prefix_sol}{i + 1}.png",
                [f"{sol_title_prefix}", f"sample {i + 1}, |u| at sample, QoI={qoi:.3e}"],
            )


def draw_tail_samples(control_model, prior, Vh, control_np, seed, cvar_value, sample_size, sample_count,
                       out_prefix_param, out_prefix_sol):
    import dolfin as dl
    V_parameter = Vh[soupy.PARAMETER]
    V_parameter_scalar = dl.FunctionSpace(V_parameter.mesh(), "CG", 1)
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

    for i, (qoi, m, state_vec) in enumerate(tail_samples):
        m_fun = scalarize_parameter_for_plot(vector_to_function(V_parameter, m), V_parameter_scalar)
        save_domain_field(
            m_fun, f"{out_prefix_param}{i + 1}.png",
            [f"Tail sample {i + 1}", f"QoI={qoi:.2e} > CVaR={cvar_value:.2e}"],
        )
        state_fun = state_velocity_magnitude(Vh[soupy.STATE], state_vec)
        save_domain_field(
            state_fun, f"{out_prefix_sol}{i + 1}.png",
            ["|u| at tail sample"],
        )


def draw_component_pair(control_model, prior, Vh, control_np, out_prefix, title_lines_x, title_lines_y):
    _, state_vec = solve_state_at_control(control_model, prior, control_np)
    u_x, u_y = velocity_components_for_plot(Vh[soupy.STATE], state_vec)
    save_domain_field(u_x, f"{out_prefix}_vx.png", title_lines_x)
    save_domain_field(u_y, f"{out_prefix}_vy.png", title_lines_y)
    return u_x, u_y


def main():
    os.makedirs(OUT_MAIN, exist_ok=True)
    os.makedirs(OUT_SUPP, exist_ok=True)

    args = build_args()
    problem = setup_problem(args, None)
    Vh = problem["Vh"]
    control_model = problem["control_model"]
    prior = problem["prior"]
    control_bases = control_model.problem.ns_residual.control_bases

    mv_saa100 = load_control(TAYLOR_DIR, "saa_100")
    cvar_saa100 = load_control(CVAR_DIR, "saa_100")

    # ---- SM11: inflow-parameter samples + mean-variance / CVaR solutions ----
    draw_samples(
        control_model, prior, Vh, mv_saa100, seed=11, sample_count=3,
        out_prefix_param=os.path.join(OUT_SUPP, "ns_samples_param_r"),
        out_prefix_sol=os.path.join(OUT_SUPP, "ns_samples_mvsol_r"),
        model_label="saa_100", sol_title_prefix=f"Mean-variance, {pretty_label('saa_100')}",
    )
    draw_samples(
        control_model, prior, Vh, cvar_saa100, seed=11, sample_count=3,
        out_prefix_param=None,
        out_prefix_sol=os.path.join(OUT_SUPP, "ns_samples_cvarsol_r"),
        model_label="saa_100", sol_title_prefix=f"CVaR, {pretty_label('saa_100')}",
    )

    # ---- SM12: CVaR tail parameter samples + solutions ----
    tail_npz = np.load(os.path.join(CVAR_DIR, "saa_100_tail_parameter_solution_samples_data.npz"))
    draw_tail_samples(
        control_model, prior, Vh,
        control_np=tail_npz["control_np"], seed=int(tail_npz["seed"]),
        cvar_value=float(tail_npz["cvar_value"]), sample_size=100, sample_count=3,
        out_prefix_param=os.path.join(OUT_SUPP, "ns_tail_param_r"),
        out_prefix_sol=os.path.join(OUT_SUPP, "ns_tail_sol_r"),
    )

    # ---- SM13/SM14: best-mixture and max-SAA boundary controls ----
    mv_bestmixture = load_control(TAYLOR_DIR, "mixture_quadratic_hep")
    mv_maxsaa = load_control(TAYLOR_DIR, "saa_500")
    cvar_bestmixture = load_control(CVAR_DIR, "mixture_quadratic_kle")
    cvar_maxsaa = load_control(CVAR_DIR, "saa_500")

    save_control_pair(mv_bestmixture, control_bases, os.path.join(OUT_SUPP, "ns_mv_bestmixture_control.png"), pretty_label("mixture_quadratic_hep"))
    save_control_pair(mv_maxsaa, control_bases, os.path.join(OUT_SUPP, "ns_mv_maxsaa_control.png"), pretty_label("saa_500"))
    save_control_pair(cvar_bestmixture, control_bases, os.path.join(OUT_SUPP, "ns_cvar_bestmixture_control.png"), pretty_label("mixture_quadratic_kle"))
    save_control_pair(cvar_maxsaa, control_bases, os.path.join(OUT_SUPP, "ns_cvar_maxsaa_control.png"), pretty_label("saa_500"))

    # ---- SM15/SM16: optimal velocity components, best-mixture vs. max-SAA ----
    draw_component_pair(
        control_model, prior, Vh, mv_bestmixture,
        os.path.join(OUT_SUPP, "ns_mv_optcomp_bestmixture"),
        [pretty_label("mixture_quadratic_hep"), "$u_x(z^*)$"], [pretty_label("mixture_quadratic_hep"), "$u_y(z^*)$"],
    )
    draw_component_pair(
        control_model, prior, Vh, mv_maxsaa,
        os.path.join(OUT_SUPP, "ns_mv_optcomp_maxsaa"),
        [pretty_label("saa_500"), "$u_x(z^*)$"], [pretty_label("saa_500"), "$u_y(z^*)$"],
    )
    draw_component_pair(
        control_model, prior, Vh, cvar_bestmixture,
        os.path.join(OUT_SUPP, "ns_cvar_optcomp_bestmixture"),
        [pretty_label("mixture_quadratic_kle"), "$u_x(z^*)$"], [pretty_label("mixture_quadratic_kle"), "$u_y(z^*)$"],
    )
    draw_component_pair(
        control_model, prior, Vh, cvar_maxsaa,
        os.path.join(OUT_SUPP, "ns_cvar_optcomp_maxsaa"),
        [pretty_label("saa_500"), "$u_x(z^*)$"], [pretty_label("saa_500"), "$u_y(z^*)$"],
    )

    # ---- SM17/SM18: components vs. shared linear-Taylor initial state (z=0) ----
    mv_vsinit = np.load(os.path.join(TAYLOR_DIR, "optimal_vs_linear_initial_solutions_data.npz"))
    cvar_vsinit = np.load(os.path.join(CVAR_DIR, "optimal_vs_linear_initial_solutions_data.npz"))
    mv_init_control = mv_vsinit["linear_initial_control_np"]
    cvar_init_control = cvar_vsinit["linear_initial_control_np"]

    draw_component_pair(
        control_model, prior, Vh, mv_init_control,
        os.path.join(OUT_SUPP, "ns_mv_vsinit_init"),
        ["Linear-Taylor initial (z=0)", "$u_x$"], ["Linear-Taylor initial (z=0)", "$u_y$"],
    )
    draw_component_pair(
        control_model, prior, Vh, cvar_init_control,
        os.path.join(OUT_SUPP, "ns_cvar_vsinit_init"),
        ["Linear-Taylor initial (z=0)", "$u_x$"], ["Linear-Taylor initial (z=0)", "$u_y$"],
    )
    # bestmixture / maxsaa vs-init panels reuse the same fields as the optcomp
    # panels above (identical control vectors); regenerate under the vsinit
    # filenames used by the supplement so all figures share one clean style.
    draw_component_pair(
        control_model, prior, Vh, mv_bestmixture,
        os.path.join(OUT_SUPP, "ns_mv_vsinit_bestmixture"),
        [pretty_label("mixture_quadratic_hep"), "$u_x(z^*)$"], [pretty_label("mixture_quadratic_hep"), "$u_y(z^*)$"],
    )
    draw_component_pair(
        control_model, prior, Vh, mv_maxsaa,
        os.path.join(OUT_SUPP, "ns_mv_vsinit_maxsaa"),
        [pretty_label("saa_500"), "$u_x(z^*)$"], [pretty_label("saa_500"), "$u_y(z^*)$"],
    )
    draw_component_pair(
        control_model, prior, Vh, cvar_bestmixture,
        os.path.join(OUT_SUPP, "ns_cvar_vsinit_bestmixture"),
        [pretty_label("mixture_quadratic_kle"), "$u_x(z^*)$"], [pretty_label("mixture_quadratic_kle"), "$u_y(z^*)$"],
    )
    draw_component_pair(
        control_model, prior, Vh, cvar_maxsaa,
        os.path.join(OUT_SUPP, "ns_cvar_vsinit_maxsaa"),
        [pretty_label("saa_500"), "$u_x(z^*)$"], [pretty_label("saa_500"), "$u_y(z^*)$"],
    )

    # ---- Main text Figure 9: mean-variance mixture quadratic HEP fields ----
    save_control_pair_split(
        mv_bestmixture, control_bases,
        os.path.join(OUT_MAIN, "ns_mv_hep_control_top.png"),
        os.path.join(OUT_MAIN, "ns_mv_hep_control_bottom.png"),
        pretty_label("mixture_quadratic_hep"),
    )
    draw_component_pair(
        control_model, prior, Vh, mv_bestmixture,
        os.path.join(OUT_MAIN, "ns_mv_hep"),
        [pretty_label("mixture_quadratic_hep"), "$u_x(z^*)$"], [pretty_label("mixture_quadratic_hep"), "$u_y(z^*)$"],
    )


def save_control_pair_split(control_np, control_bases, path_top, path_bottom, model_name):
    # plot_boundary_control_profiles always draws both sides together given
    # two axes; draw both then save each axis separately via its own tight
    # bbox so the two panels never bleed into each other.
    fig, axes = plt.subplots(1, 2, figsize=CONTROL_PAIR_FIGSIZE)
    plot_boundary_control_profiles(control_np, control_bases, axes, model_name=model_name)
    fig.tight_layout()
    for ax, path in zip(axes, (path_top, path_bottom)):
        extent = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(path, dpi=200, bbox_inches=extent.expanded(1.02, 1.05))
        print("wrote", path)
    plt.close(fig)


if __name__ == "__main__":
    main()
