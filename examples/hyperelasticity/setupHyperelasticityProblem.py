# Copyright (c) 2023, The University of Texas at Austin 
# & Georgia Institute of Technology
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the SOUPy package. For more information see
# https://github.com/hippylib/soupy/
#
# SOUPy is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 3.0 dated June 1991.

import pickle
import sys, os
from mpi4py import MPI 

import numpy as np
import matplotlib.pyplot as plt 
import dolfin as dl

sys.path.append( os.environ.get('HIPPYLIB_PATH'))
sys.path.append( os.environ.get('../../'))

import hippylib as hp 
import soupy

dl.set_log_active(False)

from hyperelasticityControlPDE import BeamGeometry, HyperelasticityControlPDE, \
        HyperelasticityVarfHandler


class LocalDisplacementVarf:
    """
    Variational form for local displacement QoI using a mollified dirac 
    to localize the displacement observation 
    """
    def __init__(self, obs_location, obs_width):
        self.obs_location = obs_location 
        self.obs_width = obs_width

    def __call__(self, u, m, z):
        mollified_dirac = dl.Expression("a*exp(-pow(x[0] - loc, 2)/(2*b*b))", loc=self.obs_location, 
                a=1/(np.sqrt(2*np.pi)*self.obs_width), 
                b=self.obs_width, 
                degree=4)
        return dl.inner(u, u) * mollified_dirac * dl.dx  


class Stiffnessvarf:
    """
    Variational form for the stiffness QoI (H1-seminorm)
    """
    def __call__(self, u, m, z):
        return dl.inner(dl.grad(u), dl.grad(u)) * dl.dx


def hyperelasticity_problem_settings():
    """
    Defines the settings for the control problem
    """
    settings = dict()
    settings["load"] = {"center" : 1.0, "width" : 0.2, "strength" : 1}
    settings["obs"] = {"location" : 0.5, "width" : 0.1}
    settings["qoi_type"] = "stiffness"
    settings["geometry"] = {"lx" : 2.0, "ly" : 0.5, "lz" : 0.25, "dim" : 2}
    settings["mesh"] = {"nx" : 96, "ny" : 24, "nz" : 12}
    settings["solver"] = {"backtrack" : True, "load_steps" : [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.9375, 1.0]}
    settings["uncertainty"] = {"gamma" : 0.1, "delta" : 1, "robin_bc" : True}
    return settings


def setup_prior(Vh, settings):
    """
    Setup the prior for the problem
    """
    prior = hp.BiLaplacianPrior(Vh[soupy.PARAMETER], 
            gamma=settings["uncertainty"]["gamma"], 
            delta=settings["uncertainty"]["delta"], 
            robin_bc=settings["uncertainty"]["robin_bc"]
            )
    return prior 


def setup_qoi(mesh, Vh, settings):
    """
    Setup the optimization QoI
    """
    if settings["qoi_type"] == "all":
        u0 = dl.Function(Vh[soupy.STATE]).vector()
        qoi = soupy.L2MisfitControlQoI(mesh, Vh, u0)
    elif settings["qoi_type"] == "stiffness":
        stiffness = Stiffnessvarf()
        qoi = soupy.VariationalControlQoI(mesh, Vh, stiffness)
    elif settings["qoi_type"] == "point":
        local_displacement = LocalDisplacementVarf(settings["obs"]["location"], settings["obs"]["width"])
        qoi = soupy.VariationalControlQoI(mesh, Vh, local_displacement)
    else:
        raise ValueError("Settings qoi type not available")

    return qoi 


def setup_hyperelasticity_problem(settings, comm_mesh=MPI.COMM_WORLD):
    """
    Setup the components for the hyperelasticity optimal design problem
    """
    
    # A mollified point load 
    T_nominal = dl.Expression(("0.0", "-a * exp(-pow(x[0] - center, 2)/(2*b*b))"), 
            a=settings["load"]["strength"]/(np.sqrt(2*np.pi)*settings["load"]["width"]),
            b=settings["load"]["width"],
            center=settings["load"]["center"],
            degree=5)

    # Define mesh, default is 2D 
    if settings["geometry"]["dim"] == 2: 
        mesh = dl.RectangleMesh(comm_mesh, 
                dl.Point(0.0, 0.0), 
                dl.Point(settings["geometry"]["lx"], settings["geometry"]["ly"]), 
                settings["mesh"]["nx"], settings["mesh"]["ny"]
        )
        left_disp = dl.Constant((0.0, 0.0))
        right_disp = dl.Constant((0.0, 0.0))

    elif settings["geometry"]["dim"] == 3:
        mesh = dl.RectangleMesh(comm_mesh, 
                dl.Point(0.0, 0.0), 
                dl.Point(settings["geometry"]["lx"], settings["geometry"]["ly"], settings["geometry"]["lz"]), 
                settings["mesh"]["nx"], settings["mesh"]["ny"], settings["mesh"]["nx"])
        left_disp = dl.Constant((0.0, 0.0, 0.0))
        right_disp = dl.Constant((0.0, 0.0, 0.0))

    else:
        mesh = dl.RectangleMesh(comm_mesh,
                dl.Point(0.0, 0.0), dl.Point(settings["geometry"]["lx"], settings["geometry"]["ly"]), 
                settings["mesh"]["nx"], settings["mesh"]["ny"])
        left_disp = dl.Constant((0.0, 0.0))
        right_disp = dl.Constant((0.0, 0.0))
    
    # Make function spaces 
    Vh_STATE = dl.VectorFunctionSpace(mesh, "CG", 1)
    Vh_PARAMETER = dl.FunctionSpace(mesh, "CG", 1)
    Vh_CONTROL = dl.FunctionSpace(mesh, "CG", 1)
    Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE, Vh_CONTROL]

    # Mark left and right boundaries for zero displacement conditions
    left_boundary =  dl.CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
    right_boundary = dl.CompiledSubDomain("near(x[0], side) && on_boundary", side = settings["geometry"]["lx"])

    bcl = dl.DirichletBC(Vh_STATE, left_disp, left_boundary)
    bcr = dl.DirichletBC(Vh_STATE, right_disp, right_boundary)
    bcs = [bcl, bcr]
    
    geometry = BeamGeometry(mesh, settings["geometry"]["lx"], settings["geometry"]["ly"])

    # Make the PDE problem, qoi, and control model
    hyperelasticity_varf = HyperelasticityVarfHandler(Vh, geometry, T_nominal, 
            spatial_dim=settings["geometry"]["dim"])

    pde = HyperelasticityControlPDE(Vh, hyperelasticity_varf, bcs, bcs, 
        load_steps=settings["solver"]["load_steps"], backtrack=settings['solver']['backtrack'])

    prior = setup_prior(Vh, settings)
    qoi = setup_qoi(mesh, Vh, settings)
    control_model = soupy.ControlModel(pde, qoi)

    return mesh, Vh, hyperelasticity_varf, control_model, prior 


