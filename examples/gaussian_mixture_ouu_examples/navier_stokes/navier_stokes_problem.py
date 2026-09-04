import pickle

import numpy as np
import dolfin as dl
import matplotlib.pyplot as plt
from mpi4py import MPI 

import hippylib as hp
import soupy

from navier_stokes_pde import NavierStokesBluffBodyHandler, NavierStokesBluffBodyControlProblem, \
    VectorSplineFunction, setup_b_splines
from obstacle_geometry import ObstacleGeometry

def defaultBluffBodySettings():
    settings = dict()
    settings["nu"] = 5e-3
    settings["continuation"] = False
    settings["stabilization"] = False
    settings["nitche"] = True
    settings["gamma"] = 1.
    settings["delta"] = 5.
    settings["mean_velocity"] = 1.
    settings["control_sd"] = 1.0
    settings["control_max"] = 2.0
    settings["control_min"] = -2.0
    settings["mesh_base_directory"] = "./"
    settings["mesh_resolution"] = "medium"
    settings["mesh_format"] = "xdmf"
    return settings


def generate_centers(control_edge, n):
    """
    Generate linearly spaced centers along the control edge
    """

    x_min = control_edge[0]
    x_max = control_edge[1]

    y_min = control_edge[2]
    y_max = control_edge[3]

    points_x = np.linspace(x_min, x_max, n+1)
    points_y = np.linspace(y_min, y_max, n+1)

    centers = []
    radii = []

    for i in range(n):
        dx = points_x[i+1] - points_x[i]
        dy = points_y[i+1] - points_y[i]

        cx = points_x[i] + dx/2
        cy = points_y[i] + dy/2
        ri = np.sqrt((dx/2)**2 + (dy/2)**2)

        centers.append([cx, cy])
        radii.append(ri)

    return centers, radii

def setup_prior(Vh_PARAMETER, mean_velocity, gamma, delta):
    mean = dl.interpolate(dl.Constant(np.log(mean_velocity)), Vh_PARAMETER).vector()
    # return hp.BiLaplacianPrior(Vh_PARAMETER, gamma, delta, mean=mean, robin_bc=True)
    return hp.BiLaplacianPrior(Vh_PARAMETER, gamma, delta, mean=mean, robin_bc=False)

def build_navier_stokes_problem(settings):
    case_name = "%s/pellet/pellet_%s_r0.1_angle0" %(settings["mesh_base_directory"], settings["mesh_resolution"])
    geo_name = case_name + "/geometry.p"
    mesh_name = case_name + "/mesh.%s" %(settings["mesh_format"])
    with open(geo_name, "rb") as geo_file:
        geo_specs = pickle.load(geo_file)

    use_stabilization = settings["stabilization"]
    use_nitche = settings["nitche"]
    if not use_nitche:
        print("Warning: this is only used for testing purposes")

    nu = settings["nu"]
    lx = geo_specs["lx"]
    ly = geo_specs["ly"]

    gamma = settings["gamma"]
    delta = settings["delta"]
    mean_velocity = settings["mean_velocity"]

    # Load control edges
    control_edge_top = geo_specs["control_edge_top"]
    dx_top = control_edge_top[1] - control_edge_top[0]
    dy_top = control_edge_top[3] - control_edge_top[2]
    v_top = np.array([dy_top, -dx_top])

    control_edge_bot = geo_specs["control_edge_bot"]
    dx_bot = control_edge_bot[1] - control_edge_bot[0]
    dy_bot = control_edge_bot[3] - control_edge_bot[2]
    v_bot = np.array([-dy_bot, dx_bot])

    x_start_top = [control_edge_top[0], control_edge_top[2]]
    x_end_top = [control_edge_top[1], control_edge_top[3]]
    x_start_bot = [control_edge_bot[0], control_edge_bot[2]]
    x_end_bot = [control_edge_bot[1], control_edge_bot[3]]
    splines = setup_b_splines()
    basis_top = [VectorSplineFunction(spline, x_start_top, x_end_top, v_top) for spline in splines]
    basis_bot = [VectorSplineFunction(spline, x_start_bot, x_end_bot, v_bot) for spline in splines]
    basis_all = basis_top + basis_bot


    dim_control = len(basis_all)
    comm_mesh = MPI.COMM_SELF 
    if settings['mesh_format'] == 'xdmf':
        mesh = dl.Mesh(comm_mesh)
        mesh_file = dl.XDMFFile(comm_mesh, mesh_name)
        mesh_file.read(mesh)
    else:
        mesh = dl.Mesh(comm_mesh, mesh_name)


    P2 = dl.VectorElement("CG", mesh.ufl_cell(), 2)
    P2_1d = dl.FiniteElement("CG", mesh.ufl_cell(), 2)
    P1 = dl.FiniteElement("CG", mesh.ufl_cell(), 1)
    ME = P2 * P1  # Taylor hood

    Vh_STATE = dl.FunctionSpace(mesh, ME)
    Vh_PARAMETER = dl.FunctionSpace(mesh, P2_1d)
    Vh_CONTROL = dl.VectorFunctionSpace(mesh, "R", 0, dim=dim_control)
    Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE, Vh_CONTROL]

    body_box = geo_specs["box"]
    geometry = ObstacleGeometry(mesh, lx, ly, body_box)
    ns_residual = NavierStokesBluffBodyHandler(Vh_STATE, mesh, geometry, nu, basis_all, use_stabilization=use_stabilization, use_nitche=use_nitche)

    if settings["continuation"]:
        ns_problem = NavierStokesBluffBodyControlProblem(Vh, ns_residual, [], [])
    else:
        ns_problem = soupy.NonlinearPDEControlProblem(Vh, ns_residual, [], [])

    prior = setup_prior(Vh_PARAMETER, mean_velocity, gamma, delta)

    return mesh, Vh, ns_problem, prior, basis_all, geometry, geo_specs


