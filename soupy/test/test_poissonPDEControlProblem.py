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
# Software Foundation) version 3.0 dated June 2007.

import unittest

import numpy as np
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
from soupy import PDEVariationalControlProblem, \
    STATE, ADJOINT, PARAMETER, CONTROL

from setupPoissonControlProblem import poisson_control_settings, \
    PoissonVarfHandler, UniformDistribution, setupPoissonPDEProblem


def u_boundary(x, on_boundary):
    return on_boundary and (x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

class TestPoissonPDEControlProblem(unittest.TestCase):
    def setUp(self):
        self.reltol = 1e-3
        self.fdtol = 1e-3
        self.delta = 1e-4
        self.n_wells_per_side = 5
        self.nx = 20
        self.ny = 20
        self.n_control = self.n_wells_per_side**2
        
        # Make spaces
        mesh = dl.UnitSquareMesh(self.nx, self.ny)
        Vh_STATE = dl.FunctionSpace(mesh, "CG", 1)
        Vh_PARAMETER = dl.FunctionSpace(mesh, "CG", 1)
        Vh_CONTROL = dl.VectorFunctionSpace(mesh, "R", degree=0, dim=self.n_control)
        self.Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE, Vh_CONTROL]
        

    def testControlDistribution(self):
        settings = poisson_control_settings()
        settings['nx'] = self.nx
        settings['ny'] = self.ny
        settings['N_WELLS_PER_SIDE'] = self.n_wells_per_side

        control_dist = UniformDistribution(self.Vh[CONTROL], 
                settings['STRENGTH_LOWER'],
                settings['STRENGTH_UPPER'])
    
        z = dl.Vector()
        control_dist.init_vector(z)
        control_dist.sample(z)
        z_vec = z.get_local()
        self.assertTrue(np.max(z_vec) <= settings['STRENGTH_UPPER']) 
        self.assertTrue(np.min(z_vec) >= settings['STRENGTH_LOWER']) 
        self.assertEqual(len(z_vec), self.n_control)


    def testPDEGenerateVector(self):
        settings = poisson_control_settings()
        settings['N_WELLS_PER_SIDE'] = self.n_wells_per_side
        bc = dl.DirichletBC(self.Vh[STATE], dl.Constant(0.0), u_boundary)
        bc0 = dl.DirichletBC(self.Vh[STATE], dl.Constant(0.0), u_boundary)
        poisson_varf = PoissonVarfHandler(self.Vh, settings=settings)
        pde = PDEVariationalControlProblem(self.Vh, poisson_varf, bc, bc0, 
                is_fwd_linear=settings["LINEAR"])

        z = pde.generate_control()
        z_vec = z.get_local()
        self.assertEqual(len(z_vec), self.n_control)

        z = dl.Vector()
        pde.init_control(z)
        z_vec = z.get_local()
        self.assertEqual(len(z_vec), self.n_control)


    def testLinearFwdAdjSolve(self):
        settings = poisson_control_settings()
        settings['nx'] = self.nx
        settings['ny'] = self.ny
        settings['N_WELLS_PER_SIDE'] = self.n_wells_per_side
        settings['LINEAR'] = True

        _, prior, control_dist = setupPoissonPDEProblem(self.Vh, settings)

        noise = dl.Vector()
        prior.init_vector(noise, "noise")
        hp.parRandom.normal(1.0, noise)

        bc = dl.DirichletBC(self.Vh[STATE], dl.Constant(0.0), u_boundary)
        bc0 = dl.DirichletBC(self.Vh[STATE], dl.Constant(0.0), u_boundary)
        poisson_varf = PoissonVarfHandler(self.Vh, settings=settings)
        pde = PDEVariationalControlProblem(self.Vh, poisson_varf, bc, bc0, 
                is_fwd_linear=settings["LINEAR"])        


        u_fun = dl.Function(self.Vh[STATE])
        p_fun = dl.Function(self.Vh[ADJOINT])
        z_fun = dl.Function(self.Vh[CONTROL])
        f_fun = dl.Function(self.Vh[STATE])
        m_fun = dl.Function(self.Vh[PARAMETER])
        zero_fun = dl.Function(self.Vh[STATE])

        prior.sample(noise, m_fun.vector()) 

        x = [u_fun.vector(), m_fun.vector(), p_fun.vector(), z_fun.vector()]
        control_dist.sample(z_fun.vector()) 

        u_trial = dl.TrialFunction(self.Vh[STATE])
        u_test = dl.TestFunction(self.Vh[STATE])
        b = dl.assemble(poisson_varf(zero_fun, m_fun, u_test, z_fun))
        M = dl.assemble(u_trial*u_test*dl.dx)
        M_solver = hp.PETScLUSolver(dl.MPI.comm_world)
        M_solver.set_operator(M)
        M_solver.solve(f_fun.vector(), -b)

        pde.solveFwd(u_fun.vector(), x)
        pde.solveAdj(p_fun.vector(), x, -b)

        u = u_fun.vector()
        p = p_fun.vector()
        diff = u - p

        u_norm = np.sqrt(u.inner(M*u))
        diff_norm = np.sqrt(diff.inner(M*diff))
        print("Difference between forward and adjoint solves %g" %(diff_norm/u_norm))
        self.assertTrue(diff_norm/u_norm < self.reltol)

    def finiteDifferenceCheck(self, is_fwd_linear=True):
        settings = poisson_control_settings()
        settings['nx'] = self.nx
        settings['ny'] = self.ny
        settings['N_WELLS_PER_SIDE'] = self.n_wells_per_side
        settings['LINEAR'] = is_fwd_linear

        pde, prior, control_dist = setupPoissonPDEProblem(self.Vh, settings)
        
        u0 = dl.Function(self.Vh[STATE]).vector()
        m0 = dl.Function(self.Vh[PARAMETER]).vector()
        p0 = dl.Function(self.Vh[ADJOINT]).vector()
        z0 = dl.Function(self.Vh[CONTROL]).vector()

        um1 = dl.Function(self.Vh[STATE]).vector()
        uz1 = dl.Function(self.Vh[STATE]).vector()
        m1 = dl.Function(self.Vh[PARAMETER]).vector()
        z1 = dl.Function(self.Vh[CONTROL]).vector()

        dudm = dl.Function(self.Vh[STATE]).vector()
        dudz = dl.Function(self.Vh[STATE]).vector()
        dm = dl.Function(self.Vh[PARAMETER]).vector()
        dz = dl.Function(self.Vh[CONTROL]).vector()
        
        # Base point 
        x = [u0, m0, p0, z0]
        control_dist.sample(z0)

        noise = dl.Vector()
        prior.init_vector(noise, "noise")
        hp.parRandom.normal(1.0, noise)
        prior.sample(noise, m0)

        pde.solveFwd(u0, x)
        pde.setLinearizationPoint(x, False)
        
        hp.parRandom.normal(1., noise)
        prior.sample(noise, dm)
        control_dist.sample(dz)

        rhs = dl.Vector()
        # pde.Cm.mult(dm, rhs)
        pde.apply_ij(ADJOINT, PARAMETER, dm, rhs)
        pde.solveIncremental(dudm, -rhs, False)

        rhs = dl.Vector()
        # pde.Cz.mult(dz, rhs)
        pde.apply_ij(ADJOINT, CONTROL, dz, rhs)
        pde.solveIncremental(dudz, -rhs, False)

        m1.axpy(1.0, m0)
        m1.axpy(self.delta, dm)

        z1.axpy(1.0, z0)
        z1.axpy(self.delta, dz)

        x = [u0, m1, p0, z0]
        pde.solveFwd(um1, x)

        x = [u0, m0, p0, z1]
        pde.solveFwd(uz1, x)

        gradm_fd = (um1.get_local() - u0.get_local())/self.delta
        gradz_fd = (uz1.get_local() - u0.get_local())/self.delta

        gradm_ad = dudm.get_local()
        gradz_ad = dudz.get_local()

        err_m = np.linalg.norm(gradm_fd - gradm_ad)/np.linalg.norm(u0.get_local())
        err_z = np.linalg.norm(gradz_fd - gradz_ad)/np.linalg.norm(u0.get_local())

        if is_fwd_linear:
            print("(Linear PDE) Relative error in du/dm: %g" %(err_m))
            print("(Linear PDE) Relative error in du/dz: %g" %(err_z))
        else:
            print("(Nonlinear PDE) Relative error in du/dm: %g" %(err_m))
            print("(Nonlinear PDE) Relative error in du/dz: %g" %(err_z))
        self.assertTrue(err_m < self.fdtol)
        self.assertTrue(err_z < self.fdtol)

    def testFiniteDifference(self):
        self.finiteDifferenceCheck(True)
        self.finiteDifferenceCheck(False)


    def testNonlinearSolverParams(self):
        settings = poisson_control_settings()
        settings['nx'] = self.nx
        settings['ny'] = self.ny
        settings['N_WELLS_PER_SIDE'] = self.n_wells_per_side
        settings['LINEAR'] = False

        _, prior, control_dist = setupPoissonPDEProblem(self.Vh, settings)

        noise = dl.Vector()
        prior.init_vector(noise, "noise")

        bc = dl.DirichletBC(self.Vh[STATE], dl.Constant(0.0), u_boundary)
        bc0 = dl.DirichletBC(self.Vh[STATE], dl.Constant(0.0), u_boundary)
        poisson_varf = PoissonVarfHandler(self.Vh, settings=settings)

        pde_updated_params = PDEVariationalControlProblem(self.Vh, poisson_varf, bc, bc0, 
                is_fwd_linear=settings["LINEAR"])        
        params = {'nonlinear_solver': 'snes',
            'snes_solver':
            {
                'linear_solver'           : 'mumps',
                'absolute_tolerance'      : 1e-10,
                'relative_tolerance'      : 1e-10,
                'maximum_iterations'      : 20,
             }
         }
        pde_updated_params.set_nonlinear_solver_parameters(params)
        pde = PDEVariationalControlProblem(self.Vh, poisson_varf, bc, bc0, 
                is_fwd_linear=settings["LINEAR"])        

        
        u0 = pde.generate_state()
        u1 = pde.generate_state()
        m = pde.generate_parameter()
        z = pde.generate_control()
        x = [u0, m, None, z]
    
        
        hp.parRandom.normal(1.0, noise)
        prior.sample(noise, m)
        hp.parRandom.normal(1.0, z)

        pde_updated_params.solveFwd(u0, x)
        pde.solveFwd(u1, x)
        
        rel_err = np.linalg.norm(u0.get_local() - u1.get_local())/np.linalg.norm(u1.get_local())
        print("Relative error between default and SNES nonlinear solver: %g" %(rel_err))
        self.assertTrue(rel_err < self.reltol)


if __name__ == "__main__":
    unittest.main()
            
