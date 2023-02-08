from statistics import mean
import unittest
import dolfin as dl
import numpy as np 
import matplotlib.pyplot as plt 

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

import os, sys
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp

sys.path.append('../../')
from hippycontrol import VariationalControlQoI, ControlModel, meanVarRiskMeasureSettings, MeanVarRiskMeasure,\
                        PDEVariationalControlProblem, UniformDistribution

from poissonControlProblem import *

def u_boundary(x, on_boundary):
    return on_boundary and (x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

class TestControlCostFunctional(unittest.TestCase):
    def setUp(self):
        self.reltol = 1e-3
        self.fdtol = 1e-1
        self.delta = 1e-3
        self.n_wells_per_side = 3
        self.nx = 20
        self.ny = 20
        self.n_control = self.n_wells_per_side**2
        
        # Make spaces
        self.mesh = dl.UnitSquareMesh(self.nx, self.ny)
        Vh_STATE = dl.FunctionSpace(self.mesh, "CG", 1)
        Vh_PARAMETER = dl.FunctionSpace(self.mesh, "CG", 1)
        Vh_CONTROL = dl.VectorFunctionSpace(self.mesh, "R", degree=0, dim=self.n_control)
        self.Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE, Vh_CONTROL]


    def testCostValue(self):
        settings = poisson_control_settings()
        settings['nx'] = self.nx
        settings['ny'] = self.ny
        settings['STRENGTH_LOWER'] = -1.
        settings['STRENGTH_UPPER'] = 2.
        settings['N_WELLS_PER_SIDE'] = self.n_wells_per_side
        settings['LINEAR'] = True

        # 2. Setting up prior
        anis_diff = dl.CompiledExpression(hp.ExpressionModule.AnisTensor2D(), degree = 1)
        anis_diff.set(settings['THETA0'], settings['THETA1'], settings['ALPHA'])
        m_mean_fun = dl.Function(self.Vh[PARAMETER])
        m_mean_fun.interpolate(dl.Constant(1.0))
        prior = hp.BiLaplacianPrior(self.Vh[PARAMETER], settings['GAMMA'], settings['DELTA'],\
                                    anis_diff, mean=m_mean_fun.vector(), robin_bc=True)

        bc = dl.DirichletBC(self.Vh[STATE], dl.Expression("x[1]", degree=1), u_boundary)
        bc0 = dl.DirichletBC(self.Vh[STATE], dl.Constant(0.0), u_boundary)
        poisson_varf = PoissonVarfHandler(self.Vh, settings=settings)
        pde = PDEVariationalControlProblem(self.Vh, poisson_varf, bc, bc0, 
                is_fwd_linear=settings["LINEAR"])

        def l2norm(u,m,z):
            return u**2*dl.dx + (m - dl.Constant(1.0))**2*dl.dx 

        qoi = VariationalControlQoI(self.mesh, self.Vh, l2norm)
        model = ControlModel(pde, qoi)

        risk = MeanVarRiskMeasure(model, prior)
        z = model.generate_vector(CONTROL)
        c_val = risk.cost()
        print("Before computing: ", c_val)
        risk.computeComponents(z)
        c_val = risk.cost()
        print("After computing: ", c_val)
        

    def finiteDifferenceCheck(self, sample_size, is_fwd_linear=True):
        # 1. Settings for PDE
        settings = poisson_control_settings()
        settings['nx'] = self.nx
        settings['ny'] = self.ny
        settings['STRENGTH_LOWER'] = -1.
        settings['STRENGTH_UPPER'] = 2.
        settings['N_WELLS_PER_SIDE'] = self.n_wells_per_side
        settings['LINEAR'] = is_fwd_linear
        settings['GAMMA'] = 10
        settings['DELTA'] = 20

        # 2. Setting up prior
        anis_diff = dl.CompiledExpression(hp.ExpressionModule.AnisTensor2D(), degree = 1)
        anis_diff.set(settings['THETA0'], settings['THETA1'], settings['ALPHA'])
        m_mean_fun = dl.Function(self.Vh[PARAMETER])
        m_mean_fun.interpolate(dl.Constant(0.0))
        prior = hp.BiLaplacianPrior(self.Vh[PARAMETER], settings['GAMMA'], settings['DELTA'],\
                                    anis_diff, mean=m_mean_fun.vector(), robin_bc=True)

        noise = dl.Vector()
        prior.init_vector(noise, "noise")

        control_dist = UniformDistribution(self.Vh[CONTROL], 
                settings['STRENGTH_LOWER'],
                settings['STRENGTH_UPPER'])
            
        # 3. Setting up PDE
        bc = dl.DirichletBC(self.Vh[STATE], dl.Expression("x[1]", degree=1), u_boundary)
        bc0 = dl.DirichletBC(self.Vh[STATE], dl.Constant(0.0), u_boundary)
        poisson_varf = PoissonVarfHandler(self.Vh, settings=settings)
        pde = PDEVariationalControlProblem(self.Vh, poisson_varf, bc, bc0, 
                is_fwd_linear=settings["LINEAR"])
    
        # 4. Setting up QoI, model, and risk measure
        def l2norm(u,m,z):
            return u**2*dl.dx

        qoi = VariationalControlQoI(self.mesh, self.Vh, l2norm)
        model = ControlModel(pde, qoi)

        rm_settings = meanVarRiskMeasureSettings()
        rm_settings['beta'] = 0.5

        risk = MeanVarRiskMeasure(model, prior, rm_settings)

        z0 = model.generate_vector(CONTROL)
        dz = model.generate_vector(CONTROL)
        z1 = model.generate_vector(CONTROL)
        g0 = model.generate_vector(CONTROL)

        # np.random.seed(1)
        control_dist.sample(z0)
        control_dist.sample(dz)
        z1.axpy(1.0, z0)
        z1.axpy(self.delta, dz)

        rng = hp.Random()        
        risk.computeComponents(z0, order=1, sample_size=sample_size, rng=rng)


        c0 = risk.cost()
        risk.costGrad(g0)

        rng = hp.Random()        
        risk.computeComponents(z1, order=0, sample_size=sample_size, rng=rng)
        c1 = risk.cost()

        dcdz_fd = (c1 - c0)/self.delta
        dcdz_ad = g0.inner(dz)
        print("Initial cost: ", c0)
        print("New cost: ", c1)
        print("Finite difference derivative: %g" %(dcdz_fd))
        print("Adjoint derivative: %g" %(dcdz_ad))
        self.assertTrue(abs((dcdz_fd - dcdz_ad)/dcdz_ad) < self.fdtol)

        # Hdz_fd = (g1.get_local() - g0.get_local())/self.delta
        # Hdz_ad = Hdz.get_local()
        # print("Finite difference Hessian action ", Hdz_fd)
        # print("Adjoint Hessian action ", Hdz_ad)
        # err_hess = np.linalg.norm(Hdz_fd - Hdz_ad)
        # print("Norm error: %g" %(err_hess))
        # self.assertTrue(err_hess/np.linalg.norm(Hdz_ad) < self.fdtol)

    def testFiniteDifference(self):
        self.finiteDifferenceCheck(1, True)
        self.finiteDifferenceCheck(10, True)
        self.finiteDifferenceCheck(1, False)
        self.finiteDifferenceCheck(10, False)


if __name__ == "__main__":
    unittest.main()
