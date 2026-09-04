import os
import sys

import dolfin as dl
import ufl

sys.path.append(os.environ.get("HIPPYLIB_PATH", ""))
sys.path.append(os.environ.get("../../", ""))

import soupy

from navier_stokes_problem import build_navier_stokes_problem, defaultBluffBodySettings


class VelocityTrackingVarf:
    """
    Velocity-tracking QoI on a fixed rectangular observation window.
    """

    def __init__(self, mesh, x_min=0.6, x_max=2.0, y_min=0.0, y_max=1.0, target=(1.0, 0.0)):
        self.mesh = mesh
        self.target = dl.Constant(target)

        x = dl.SpatialCoordinate(mesh)
        in_x = ufl.And(ufl.ge(x[0], x_min), ufl.le(x[0], x_max))
        in_y = ufl.And(ufl.ge(x[1], y_min), ufl.le(x[1], y_max))
        self.chi = dl.conditional(ufl.And(in_x, in_y), dl.Constant(1.0), dl.Constant(0.0))

    def __call__(self, u, m, z):
        del m, z
        v, _ = dl.split(u)
        diff = v - self.target
        return self.chi * dl.inner(diff, diff) * dl.dx


class PhysicalControlPenalizationForm:
    """
    Penalize the physical control field phi(z)(x), not the raw coefficient vector.
    """

    def __init__(self, residual_handler, alpha=1.0):
        self.residual_handler = residual_handler
        self.alpha = float(alpha)

    def __call__(self, z):
        phi_z = self.residual_handler.control_to_obstacle_velocity(z)
        return dl.Constant(self.alpha) * dl.inner(phi_z, phi_z) * dl.dx


def navier_stokes_problem_settings():
    settings = defaultBluffBodySettings()
    settings["qoi_type"] = "velocity_tracking"
    settings["penalty_alpha"] = 1.0
    settings["tracking_region"] = {"x_min": 0.6, "x_max": 2.0, "y_min": 0.0, "y_max": 1.0}
    settings["target_velocity"] = (1.0, 0.0)
    return settings


def setup_qoi(mesh, Vh, settings):
    if settings["qoi_type"] != "velocity_tracking":
        raise ValueError(f"Unsupported Navier-Stokes QoI type: {settings['qoi_type']}")

    region = settings["tracking_region"]
    form_handler = VelocityTrackingVarf(
        mesh,
        x_min=region["x_min"],
        x_max=region["x_max"],
        y_min=region["y_min"],
        y_max=region["y_max"],
        target=settings["target_velocity"],
    )
    return soupy.VariationalControlQoI(Vh, form_handler)


def setup_penalty(Vh, residual_handler, settings):
    alpha = settings.get("penalty_alpha", 1.0)
    form_handler = PhysicalControlPenalizationForm(residual_handler, alpha=alpha)
    return soupy.VariationalPenalization(Vh, form_handler)


def setup_navier_stokes_problem(settings=None):
    if settings is None:
        settings = navier_stokes_problem_settings()

    mesh, Vh, ns_problem, prior, basis_all, geometry, geo_specs = build_navier_stokes_problem(settings)
    qoi = setup_qoi(mesh, Vh, settings)
    penalty = setup_penalty(Vh, ns_problem.ns_residual, settings)
    control_model = soupy.ControlModel(ns_problem, qoi)

    return {
        "settings": settings,
        "mesh": mesh,
        "Vh": Vh,
        "problem": ns_problem,
        "control_model": control_model,
        "prior": prior,
        "penalty": penalty,
        "basis_all": basis_all,
        "geometry": geometry,
        "geo_specs": geo_specs,
    }
