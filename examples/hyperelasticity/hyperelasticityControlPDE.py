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

import numpy as np
import dolfin as dl
from mpi4py import MPI 

sys.path.append(os.environ.get('HIPPYLIB_PATH'))
sys.path.append(os.environ.get('../../'))
import hippylib as hp 
import soupy

TOP = 1 

class BeamGeometry:
    def __init__(self, mesh, Lx, Ly):
        self.mesh = mesh
        self.Lx = Lx 
        self.Ly = Ly 
        self.TOP = TOP
        self.ds = create_labels(mesh, Lx, Ly)


def create_labels(mesh, Lx, Ly):
    """
    Mark boundaries of a domain corresponding to a channel domain
    """
    boundaries = dl.MeshFunction("size_t", mesh, mesh.geometry().dim()-1)
    boundaries.set_all(0)
    top = Top(Lx, Ly)
    top.mark(boundaries, TOP)
    ds = dl.Measure("ds", domain=mesh, subdomain_data=boundaries)
    return ds 


class Top(dl.SubDomain):
    def __init__(self, Lx, Ly, **kwargs):
        self.Lx = Lx 
        self.Ly = Ly 
        super().__init__(**kwargs)

    def inside(self, x, on_boundary):
        return dl.near(x[1], self.Ly)


def hyperElasticityPDESettings():
    settings = dict() 
    settings["E0"] = 20
    settings["E1"] = 200
    settings["nu"] = 0.3 
    return settings


class HyperelasticityVarfHandler:
    def __init__(self, Vh, geometry, t_nominal, settings=hyperElasticityPDESettings(), spatial_dim=2):
        assert spatial_dim == 2 or spatial_dim == 3 
        self.Vh = Vh 
        self.geometry = geometry
        self.E0 = dl.Constant(settings["E0"])
        self.E1 = dl.Constant(settings["E1"])
        self.nu = dl.Constant(settings["nu"])
        self.t_nominal = t_nominal 
        self.spatial_dim = spatial_dim
        self.z_dim = self.Vh[soupy.CONTROL].dim()

    def parameterControl2Modulus(self, m, z):
        """
        Control variable to elastic modulus form
        """
        percentage = z**3
        E = self.E0 + percentage * (self.E1 - self.E0) + m 
        return E 

    def parameterControl2Traction(self, m, z):
        return self.t_nominal

    def energy(self, u, m, p, z, load_step=1.0):
        """
        Strain energy form
        """
        traction = self.parameterControl2Traction(m, z) * dl.Constant(load_step)
        d = u.geometric_dimension()
        I = dl.Identity(d) # Identity tensor
        F = I + dl.grad(u) # Deformation gradient
        C = F.T*F # Right Cauchy-Green tensor
        
        # Invariants
        Ic = dl.tr(C)
        J  = dl.det(F)
    
        E = self.parameterControl2Modulus(m, z)
        mu, lmbda = E/(2*(1 + self.nu)), E*self.nu/((1 + self.nu)*(1 - 2*self.nu))
        psi = (mu/2)*(Ic - 3) - mu*dl.ln(J) + (lmbda/2)*(dl.ln(J))**2
        Pi = psi*dl.dx - dl.dot(traction, u) * self.geometry.ds(TOP)
        return Pi

    def __call__(self, u, m, p, z, load_step=1.0):
        """
        Residual form
        """
        Pi = self.energy(u, m, p, z,load_step=load_step)
        F = dl.derivative(Pi, u, p)
        return F 



class HyperelasticityControlPDE(soupy.PDEVariationalControlProblem):
    def __init__(self, Vh, varf_handler, bc, bc0, load_steps=[1.0], max_newton_iter=10, max_newton_iter_increment=15, backtrack=True):
        super().__init__(Vh, varf_handler, bc, bc0, is_fwd_linear=False)
        self.load_steps = load_steps 
        self.n_linear_solves = 0 
        self.max_newton_iter = max_newton_iter
        self.max_newton_iter_increment = max_newton_iter_increment
        self.backtrack = backtrack
        
        default_solver_parameters = {'newton_solver': {'linear_solver' : 'lu', 'maximum_iterations': self.max_newton_iter}}
        self.default_newton_solver = soupy.NonlinearVariationalSolver(default_solver_parameters)

        self.backtrack_newton_solver = soupy.NewtonBacktrackSolver()
        self.backtrack_newton_solver.parameters["maximum_iterations"] = self.max_newton_iter_increment

    def set_load_steps(self, load_steps):
        self.load_steps = load_steps

    def solveFwd(self, state, x):
        """
        Solve the forward problem using default Newton. \
            Do load increments using backtracking newton if full solve fails
        """
        self.n_calls["forward"] += 1
        try:
            print("Process %d: Try immediate solve" %(MPI.COMM_WORLD.Get_rank()))
            # Make functions 
            u = hp.vector2Function(x[soupy.STATE], self.Vh[soupy.STATE])
            m = hp.vector2Function(x[soupy.PARAMETER], self.Vh[soupy.PARAMETER])
            p = dl.TestFunction(self.Vh[soupy.ADJOINT])
            z = hp.vector2Function(x[soupy.CONTROL], self.Vh[soupy.CONTROL])

            # Define full nonlinear problem
            res_form = self.varf_handler(u, m, p, z)
            jacobian_form = dl.derivative(res_form, u) 

            num_iters, converged = self.default_newton_solver.solve(res_form, u, self.bc, jacobian_form)
            self.n_linear_solves += num_iters

            if converged:
                print("Process %d: Immediate solve succeeded" %(MPI.COMM_WORLD.Get_rank()))

        except:
            print("Process %d: Increment load instead starting form zero with backtracking" %(MPI.COMM_WORLD.Get_rank()))

            # Start with zero initial guess
            u = dl.Function(self.Vh[soupy.STATE])
            m = hp.vector2Function(x[soupy.PARAMETER], self.Vh[soupy.PARAMETER])
            p = dl.TestFunction(self.Vh[soupy.ADJOINT])
            z = hp.vector2Function(x[soupy.CONTROL], self.Vh[soupy.CONTROL])

            for load_step in self.load_steps:
                print("Process %d: Load step: " %(MPI.COMM_WORLD.Get_rank()), load_step)

                res_form = self.varf_handler(u, m, p, z, load_step=load_step)
                jacobian_form = dl.derivative(res_form, u) 
                energy_form = self.varf_handler.energy(u, m, p, z, load_step=load_step)
                
                # num_iters, converged = solver.solve(res_form, u, self.bc, jacobian_form, energy_form)
                num_iters, converged = self.backtrack_newton_solver.solve(res_form, u, self.bc, jacobian_form)
                self.n_linear_solves += num_iters
        state.zero()
        state.axpy(1., u.vector())



