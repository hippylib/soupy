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

import unittest
import matplotlib.pyplot as plt
import dolfin as dl

import sys, os 
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

sys.path.append('../../')
from soupy import PDEVariationalControlProblem, NewtonBacktrack, NonlinearPDEControlProblem, \
    UniformDistribution, STATE, PARAMETER, ADJOINT, CONTROL

from poissonControlProblem import PoissonVarfHandler, NonlinearPoissonVarfHandler, poisson_control_settings

def u_boundary(x, on_boundary):
    return on_boundary and (x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)


def setupNonlinearPDEProblem(nx, ny, n_wells_per_side):
    settings = poisson_control_settings()
    settings['nx'] = nx
    settings['ny'] = ny
    settings['STRENGTH_LOWER'] = -1.
    settings['STRENGTH_UPPER'] = 2.
    settings['N_WELLS_PER_SIDE'] = n_wells_per_side
    settings['LINEAR'] = False
    n_control = n_wells_per_side ** 2 
    
    # 1. Make spaces
    mesh = dl.UnitSquareMesh(nx, ny)
    Vh_STATE = dl.FunctionSpace(mesh, "CG", 1)
    Vh_PARAMETER = dl.FunctionSpace(mesh, "CG", 1)
    Vh_CONTROL = dl.VectorFunctionSpace(mesh, "R", degree=0, dim=n_control)
    Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE, Vh_CONTROL]

    bc = dl.DirichletBC(Vh[STATE], dl.Expression("x[1]", degree=1), u_boundary)
    bc0 = dl.DirichletBC(Vh[STATE], dl.Constant(0.0), u_boundary)
    poisson_varf = NonlinearPoissonVarfHandler(Vh, settings=settings)
    newton_solver = NewtonBacktrack()
    pde = NonlinearPDEControlProblem(Vh, poisson_varf, [bc], [bc0], newton_solver)
    
    return Vh, pde, settings


def setupDistributions(Vh, settings):
    # 2. Setting up prior
    anis_diff = dl.CompiledExpression(hp.ExpressionModule.AnisTensor2D(), degree = 1)
    anis_diff.set(settings['THETA0'], settings['THETA1'], settings['ALPHA'])
    m_mean_fun = dl.Function(Vh[PARAMETER])
    m_mean_fun.interpolate(dl.Constant(0.0))
    prior = hp.BiLaplacianPrior(Vh[PARAMETER], settings['GAMMA'], settings['DELTA'],\
                                anis_diff, mean=m_mean_fun.vector(), robin_bc=True)
    
    control_dist = UniformDistribution(Vh[CONTROL], 
            settings['STRENGTH_LOWER'],
            settings['STRENGTH_UPPER'])
    
    return prior, control_dist


def setupVariationalPDEControlProblem(nx, ny, n_wells_per_side, is_fwd_linear):
    settings = poisson_control_settings()
    settings['nx'] = nx
    settings['ny'] = ny
    settings['STRENGTH_LOWER'] = -1.
    settings['STRENGTH_UPPER'] = 2.
    settings['N_WELLS_PER_SIDE'] = n_wells_per_side
    settings['LINEAR'] = is_fwd_linear
    n_control = n_wells_per_side ** 2 
    
    # 1. Make spaces
    mesh = dl.UnitSquareMesh(nx, ny)
    Vh_STATE = dl.FunctionSpace(mesh, "CG", 1)
    Vh_PARAMETER = dl.FunctionSpace(mesh, "CG", 1)
    Vh_CONTROL = dl.VectorFunctionSpace(mesh, "R", degree=0, dim=n_control)
    Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE, Vh_CONTROL]

    bc = dl.DirichletBC(Vh[STATE], dl.Expression("x[1]", degree=1), u_boundary)
    bc0 = dl.DirichletBC(Vh[STATE], dl.Constant(0.0), u_boundary)
    poisson_varf = PoissonVarfHandler(Vh, settings=settings)
    pde = PDEVariationalControlProblem(Vh, poisson_varf, bc, bc0, 
            is_fwd_linear=settings["LINEAR"])
    
    return Vh, pde, settings

class TestPoissonPDEControlProblem(unittest.TestCase):
    def setUp(self):
        self.reltol = 1e-3
        self.fdtol = 1e-3
        self.delta = 1e-4
        self.n_wells_per_side = 5
        self.nx = 20
        self.ny = 20
        self.n_control = self.n_wells_per_side**2
        
        
    def countPDEIterations(self, Vh, pde, prior, control_dist, is_fwd_linear):
        noise = dl.Vector()
        prior.init_vector(noise, "noise")

        u0 = dl.Function(Vh[STATE]).vector()
        m0 = dl.Function(Vh[PARAMETER]).vector()
        p0 = dl.Function(Vh[ADJOINT]).vector()
        z0 = dl.Function(Vh[CONTROL]).vector()
        
        # Base point 
        x = [u0, m0, p0, z0]
        control_dist.sample(z0)
        m0.axpy(1.0, prior.mean)
        pde.solveFwd(u0, x)
        rhs = pde.generate_state()
        print(z0.get_local())

        if is_fwd_linear:
            print("Linear PDE")
        else:
            print("Nonlinear PDE")

        print("Total linear solves after solveFwd: ", pde.n_linear_solves)

        if is_fwd_linear:
            self.assertEqual(pde.n_linear_solves, 1)
        else:
            self.assertGreater(pde.n_linear_solves, 1)
        n_solves = pde.n_linear_solves

        pde.solveAdj(p0, x, rhs)
        self.assertEqual(pde.n_linear_solves, n_solves + 1)
        n_solves = pde.n_linear_solves
        print("Total linear solves after solveAdj: ", pde.n_linear_solves)

        pde.setLinearizationPoint(x, gauss_newton_approx=False)

        pde.solveIncremental(u0, rhs, is_adj=False)
        self.assertEqual(pde.n_linear_solves, n_solves + 1)
        n_solves = pde.n_linear_solves
        print("Total linear solves after solveIncremental (for): ", pde.n_linear_solves)


        pde.solveIncremental(p0, rhs, is_adj=True)
        self.assertEqual(pde.n_linear_solves, n_solves + 1)
        n_solves = pde.n_linear_solves
        print("Total linear solves after solveIncremental (adj): ", pde.n_linear_solves)
    
        control_dist.sample(z0)
        pde.solveFwd(u0, x)
        self.assertGreater(pde.n_linear_solves, n_solves)
        print("Total linear solves after new solveFwd: ", pde.n_linear_solves)

    def nonlinearPDEIterations(self): 
        is_fwd_linear = False
        Vh, pde, settings = setupNonlinearPDEProblem(self.nx, self.ny, self.n_wells_per_side)
        prior, control_dist = setupDistributions(Vh, settings)
        self.countPDEIterations(Vh, pde, prior, control_dist, is_fwd_linear)


    def variationalPDEIterations(self, is_fwd_linear=True):
        Vh, pde, settings = setupVariationalPDEControlProblem(self.nx, self.ny, self.n_wells_per_side, is_fwd_linear)
        prior, control_dist = setupDistributions(Vh, settings)
        self.countPDEIterations(Vh, pde, prior, control_dist, is_fwd_linear)

    def testVariationPDEIterations(self):
        self.variationalPDEIterations(True)
        self.variationalPDEIterations(False)
        self.nonlinearPDEIterations()

if __name__ == "__main__":
    unittest.main()


