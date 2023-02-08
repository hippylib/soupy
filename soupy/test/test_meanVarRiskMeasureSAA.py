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
from hippycontrol import VariationalControlQoI, ControlModel, \
                        meanVarRiskMeasureSAASettings, MeanVarRiskMeasureSAA,\
                        PDEVariationalControlProblem, UniformDistribution

from poissonControlProblem import *

def u_boundary(x, on_boundary):
    return on_boundary and (x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)


def setupProblem(Vh, settings):
    # 2. Setting up prior
    anis_diff = dl.CompiledExpression(hp.ExpressionModule.AnisTensor2D(), degree = 1)
    anis_diff.set(settings['THETA0'], settings['THETA1'], settings['ALPHA'])
    m_mean_fun = dl.Function(Vh[PARAMETER])
    m_mean_fun.interpolate(dl.Constant(1.0))
    prior = hp.BiLaplacianPrior(Vh[PARAMETER], settings['GAMMA'], settings['DELTA'],\
                                anis_diff, mean=m_mean_fun.vector(), robin_bc=True)

    bc = dl.DirichletBC(Vh[STATE], dl.Expression("x[1]", degree=1), u_boundary)
    bc0 = dl.DirichletBC(Vh[STATE], dl.Constant(0.0), u_boundary)
    poisson_varf = PoissonVarfHandler(Vh, settings=settings)
    pde = PDEVariationalControlProblem(Vh, poisson_varf, bc, bc0, 
            is_fwd_linear=settings["LINEAR"])
    
    control_dist = UniformDistribution(Vh[CONTROL], 
            settings['STRENGTH_LOWER'],
            settings['STRENGTH_UPPER'])
    return pde, prior, control_dist



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

        pde, prior, _ = setupProblem(self.Vh, settings)
        def l2norm(u,m,z):
            return u**2*dl.dx + (m - dl.Constant(1.0))**2*dl.dx 

        qoi = VariationalControlQoI(self.mesh, self.Vh, l2norm)
        model = ControlModel(pde, qoi)

        risk = MeanVarRiskMeasureSAA(model, prior)
        z = model.generate_vector(CONTROL)
        c_val = risk.cost()
        print("Before computing: ", c_val)
        risk.computeComponents(z)
        c_val = risk.cost()
        print("After computing: ", c_val)
        
    def testSavedSolution(self):
        # 1. Settings for PDE
        settings = poisson_control_settings()
        settings['nx'] = self.nx
        settings['ny'] = self.ny
        settings['STRENGTH_LOWER'] = -1.
        settings['STRENGTH_UPPER'] = 2.
        settings['N_WELLS_PER_SIDE'] = self.n_wells_per_side
        settings['LINEAR'] = False
        settings['GAMMA'] = 10
        settings['DELTA'] = 20
        
        # 2. Setup problem
        pde, prior, control_dist = setupProblem(self.Vh, settings)

        # 3. Setting up problem
        pde, prior, control_dist = setupProblem(self.Vh, settings)
        noise = dl.Vector()
        prior.init_vector(noise, "noise")
    
        # 3. Setting up QoI, model, and risk measure
        def l2norm(u,m,z):
            return u**2*dl.dx

        qoi = VariationalControlQoI(self.mesh, self.Vh, l2norm)
        model = ControlModel(pde, qoi)

        rm_settings = meanVarRiskMeasureSAASettings()
        rm_settings['sample_size'] = 5
        rm_settings['beta'] = 0.5
        risk = MeanVarRiskMeasureSAA(model, prior, rm_settings)

        z0 = model.generate_vector(CONTROL)
        z1 = model.generate_vector(CONTROL)

        # np.random.seed(1)
        control_dist.sample(z0)
        control_dist.sample(z1)
        
        print("Test if correct solution and adjoint solves are being stored")
        
        # initial cost
        risk.computeComponents(z0, order=0)
        self.assertFalse(risk.has_adjoint_solve)
        c0 = risk.cost()
        
        # same place, check same 
        risk.computeComponents(z0, order=0)
        self.assertFalse(risk.has_adjoint_solve)
        self.assertAlmostEqual(c0, risk.cost())
    
        # now with gradient 
        risk.computeComponents(z0, order=1)
        self.assertAlmostEqual(c0, risk.cost())
        self.assertTrue(risk.has_adjoint_solve)
        
        # new cost, no gradient 
        risk.computeComponents(z1, order=0)
        c1 = risk.cost()
        self.assertFalse(risk.has_adjoint_solve)
        
        # back to old cost
        risk.computeComponents(z0, order=0)
        self.assertAlmostEqual(c0, risk.cost())
        self.assertFalse(risk.has_adjoint_solve)

        # new cost, with gradient
        risk.computeComponents(z1, order=1)
        self.assertTrue(risk.has_adjoint_solve)
        self.assertAlmostEqual(c1, risk.cost())

        # new cost, no gradient 
        risk.computeComponents(z1, order=0)
        self.assertTrue(risk.has_adjoint_solve)
        self.assertAlmostEqual(c1, risk.cost())

        # old cost, no gradient 
        risk.computeComponents(z0, order=1)
        self.assertAlmostEqual(c0, risk.cost())
        self.assertTrue(risk.has_adjoint_solve)




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

        # 2. Setting up problem
        pde, prior, control_dist = setupProblem(self.Vh, settings)
        noise = dl.Vector()
        prior.init_vector(noise, "noise")
    
        # 3. Setting up QoI, model, and risk measure
        def l2norm(u,m,z):
            return u**2*dl.dx

        qoi = VariationalControlQoI(self.mesh, self.Vh, l2norm)
        model = ControlModel(pde, qoi)

        rm_settings = meanVarRiskMeasureSAASettings()
        rm_settings['sample_size'] = sample_size
        rm_settings['beta'] = 0.5

        risk = MeanVarRiskMeasureSAA(model, prior, rm_settings)

        z0 = model.generate_vector(CONTROL)
        dz = model.generate_vector(CONTROL)
        z1 = model.generate_vector(CONTROL)
        g0 = model.generate_vector(CONTROL)

        # np.random.seed(1)
        control_dist.sample(z0)
        control_dist.sample(dz)
        z1.axpy(1.0, z0)
        z1.axpy(self.delta, dz)

        risk.computeComponents(z0, order=1)

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


    def testFiniteDifference(self):
        self.finiteDifferenceCheck(1, True)
        self.finiteDifferenceCheck(10, True)
        self.finiteDifferenceCheck(1, False)
        self.finiteDifferenceCheck(10, False)


if __name__ == "__main__":
    unittest.main()
