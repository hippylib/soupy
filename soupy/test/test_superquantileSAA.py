import unittest
import dolfin as dl
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.stats
import math

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

import os, sys
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp

sys.path.append('../../')
from soupy import VariationalControlQoI, ControlModel, \
                        superquantileRiskMeasureSAASettings, SuperquantileRiskMeasureSAA_MPI, \
                        PDEVariationalControlProblem, UniformDistribution, \
                        STATE, PARAMETER, ADJOINT, CONTROL

from poissonControlProblem import poisson_control_settings, PoissonVarfHandler

def u_boundary(x, on_boundary):
    return on_boundary and (x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)


def standardNormalSuperquantile(beta):
    quantile = scipy.stats.norm.ppf(beta)
    return np.exp(-quantile**2/2)/(1-beta)/np.sqrt(2*math.pi)


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
    
    return pde, prior



class TestSuperquantileSAA(unittest.TestCase):
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
        
        # 2. Setting up problem
        pde, prior = setupProblem(self.Vh, settings)
    
        # 3. Setting up QoI, model, and risk measure
        def l2norm(u,m,z):
            return u**2*dl.dx

        qoi = VariationalControlQoI(self.mesh, self.Vh, l2norm)
        model = ControlModel(pde, qoi)

        rm_settings = superquantileRiskMeasureSAASettings()
        rm_settings['sample_size'] = 5
        rm_settings['beta'] = 0.5
        risk = SuperquantileRiskMeasureSAA_MPI(model, prior, rm_settings)

        z0 = risk.generate_vector(CONTROL)
        z1 = risk.generate_vector(CONTROL)

        # np.random.seed(1)
        z0.set_local(np.random.randn(len(z0.get_local())))
        z1.set_local(np.random.randn(len(z1.get_local())))
        
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




    def finiteDifferenceCheck(self, sample_size, smoothplus_type, is_fwd_linear=True):
        # 1. Settings for PDE
        settings = poisson_control_settings()
        settings['nx'] = self.nx
        settings['ny'] = self.ny
        settings['N_WELLS_PER_SIDE'] = self.n_wells_per_side
        settings['LINEAR'] = is_fwd_linear

        # 2. Setting up problem
        pde, prior = setupProblem(self.Vh, settings)
    
        # 3. Setting up QoI, model, and risk measure
        def l2norm(u,m,z):
            return u**2*dl.dx

        qoi = VariationalControlQoI(self.mesh, self.Vh, l2norm)
        model = ControlModel(pde, qoi)

        rm_settings = superquantileRiskMeasureSAASettings()
        rm_settings['sample_size'] = sample_size
        rm_settings['beta'] = 0.5
        rm_settings['smoothplus_type'] = smoothplus_type

        risk = SuperquantileRiskMeasureSAA_MPI(model, prior, rm_settings)

        z0 = risk.generate_vector(CONTROL)
        dz = risk.generate_vector(CONTROL)
        z1 = risk.generate_vector(CONTROL)
        g0 = risk.generate_vector(CONTROL)

        # np.random.seed(1)
        z0.set_local(np.random.randn(len(z0.get_local())))
        dz.set_local(np.random.randn(len(dz.get_local())))
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
        print(40*"-")
        if is_fwd_linear:
            print("Linear, %d Samples, Smooth plus type: %s" %(sample_size, smoothplus_type))
        else:
            print("Nonlinear, %d Samples, Smooth plus type: %s" %(sample_size, smoothplus_type))
        
        print("Q: ", risk.q_samples)
        print("Initial cost: ", c0)
        print("New cost: ", c1)
        print("Finite difference derivative: %g" %(dcdz_fd))
        print("Adjoint derivative: %g" %(dcdz_ad))
        print(40*"-")
        self.assertTrue(abs((dcdz_fd - dcdz_ad)/dcdz_ad) < self.fdtol)


    def testFiniteDifference(self):
        sample_sizes = [1, 10]
        linearities = [True, False]
        smoothpluses = ["softplus", "quartic"]
        smoothpluses = ["quartic"]
        
        for sample_size in sample_sizes:
            for linearity in linearities:
                for smoothplus in smoothpluses: 
                    self.finiteDifferenceCheck(sample_size, smoothplus, linearity)

    def computeSuperquantileValue(self, sample_size, beta):
        settings = poisson_control_settings()
        settings['N_WELLS_PER_SIDE'] = self.n_wells_per_side

        # 2. Setting up problem
        pde, prior = setupProblem(self.Vh, settings)
    
        # 3. Setting up QoI, model, and risk measure
        def l2norm(u,m,z):
            return u**2*dl.dx

        qoi = VariationalControlQoI(self.mesh, self.Vh, l2norm)
        model = ControlModel(pde, qoi)

        rm_settings = superquantileRiskMeasureSAASettings()
        rm_settings['sample_size'] = sample_size
        rm_settings['beta'] = beta
        risk = SuperquantileRiskMeasureSAA_MPI(model, prior, rm_settings)
        np.random.seed(1)
        risk.q_samples = np.random.randn(len(risk.q_samples))
        sq = risk.superquantile()
        return sq
    

    def testSuperquantileValue(self):
        sample_size = 10000
        beta = 0.2
        sq_normal = standardNormalSuperquantile(beta)
        sq = self.computeSuperquantileValue(sample_size, beta)
        print("Computed superquantile: ", sq)
        print("Analytic superquantile: ", sq_normal)
        tol = 1e-2
        self.assertTrue(np.abs(sq_normal - sq) < tol)
        
    
    def testCostValue(self):
        beta = 0.2
        sample_size = 100
        smoothplus_types = ["softplus", "quartic"]
        settings = poisson_control_settings()
        settings['N_WELLS_PER_SIDE'] = self.n_wells_per_side

        tol = 5e-2
        def l2norm(u,m,z):
            return u**2*dl.dx

        for smoothplus_type in smoothplus_types:
            rm_settings = superquantileRiskMeasureSAASettings()
            rm_settings['sample_size'] = sample_size
            rm_settings['beta'] = beta
            rm_settings['smoothplus_type'] = smoothplus_type

            pde, prior = setupProblem(self.Vh, settings)
            qoi = VariationalControlQoI(self.mesh, self.Vh, l2norm)
            model = ControlModel(pde, qoi)
            risk = SuperquantileRiskMeasureSAA_MPI(model, prior, rm_settings)

            zt = risk.generate_vector(CONTROL)
            z = zt.get_vector()
            np.random.seed(1)
            z.set_local(np.random.randn(len(z.get_local())))

            risk.computeComponents(zt, order=0)
            quantile = np.quantile(risk.q_samples, beta)
            zt.set_scalar(quantile)
            risk.computeComponents(zt, order=0)
            sq_by_cost = risk.cost()
            sq_by_samples = risk.superquantile()

            print("Superquantile from cost: ", sq_by_cost)
            print("Superquantile from samples: ", sq_by_samples)
            err = np.abs(sq_by_cost - sq_by_samples)
            self.assertTrue(err/np.abs(sq_by_samples) < tol)





if __name__ == "__main__":
    unittest.main()
