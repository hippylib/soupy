import unittest

import numpy as np
import dolfin as dl

import sys, os 
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)

sys.path.append('../../')
from soupy import STATE, PARAMETER, ADJOINT, CONTROL, L2Penalization

class TestPenalization(unittest.TestCase):
    def setUp(self):
        self.mesh = dl.UnitSquareMesh(32, 32)

    def finiteDifferenceGradient(self, penalty, z, dz, delta=1e-5, grad_tol=1e-3):
        grad = dl.Vector()
        penalty.init_vector(grad, 0)

        p0 = penalty.cost(z)
        penalty.grad(z, grad)

        z.axpy(delta, dz)
        p1 = penalty.cost(z)

        finite_diff = (p1 - p0)/delta
        symbolic_derivative = grad.inner(dz)
        print("Finite difference derivative: %g" %(finite_diff))
        print("Implemented derivative: %g" %(symbolic_derivative))
        self.assertTrue(abs(finite_diff - symbolic_derivative) < grad_tol)

    def finiteDifferenceHessian(self, penalty, z, zhat, delta=1e-5, hess_tol=1e-3):
        grad0 = dl.Vector()
        grad1 = dl.Vector()
        Hzhat = dl.Vector()

        penalty.init_vector(grad0, 0)
        penalty.init_vector(grad1, 0)
        penalty.init_vector(Hzhat, 0)

        penalty.grad(z, grad0)
        penalty.hessian(z, zhat, Hzhat)

        z.axpy(delta, zhat)
        penalty.grad(z, grad1)

        Hzhat_fd = (grad1 - grad0)/delta
        hess_diff = Hzhat_fd - Hzhat

        if np.linalg.norm(Hzhat.get_local()) > 0:
            hess_err = np.sqrt(hess_diff.inner(hess_diff))/np.linalg.norm(Hzhat.get_local())
        else:
            hess_err = np.sqrt(hess_diff.inner(hess_diff))

        print("Hessian difference: %g" %(hess_err))
        self.assertTrue(hess_err < hess_tol)


    def compareFunctionValue(self, penalty, z, p_true, tol=1e-3):
        p = penalty.cost(z)
        p_err = abs(p - p_true)
        print("Evaluation: %g" %(p))
        print("Truth: %g" %(p_true))
        self.assertTrue(p_err < tol)


    def testL2Penalization(self):
        print("Test function L2 Penalization")
        Vh_CONTROL = dl.FunctionSpace(self.mesh, "CG", 2)
        Vh = [None, None, None, Vh_CONTROL]

        alpha = 2.0
        L2 = L2Penalization(Vh, alpha)
        z_fun = dl.Function(Vh_CONTROL)
        z = z_fun.vector()
        dz_fun = dl.Function(Vh_CONTROL)
        dz = dz_fun.vector()




        z_fun.interpolate(dl.Constant(1.0))
        P_TRUE = 1.0 * alpha

        self.compareFunctionValue(L2, z, P_TRUE, tol=1e-3)


        z_fun.interpolate(dl.Expression("cos(x[0])*x[1] + exp(-pow(x[0]-0.5, 2) - pow(x[1]-0.5, 2))", degree=2))
        dz_fun.interpolate(dl.Expression("cos(2.88*x[0])*x[1] + exp(-pow(x[0]-0.5, 2) - pow(x[1]-0.5, 2))",
            degree=5))

        self.finiteDifferenceGradient(L2, z, dz)
        self.finiteDifferenceHessian(L2, z, dz)

    def testVectorL2Penalization(self):
        print("Test vector l2 Penalization")
        dim = 10
        Vh_CONTROL = dl.VectorFunctionSpace(self.mesh, "R", 0, dim=dim)
        Vh = [None, None, None, Vh_CONTROL]

        alpha = 2.0
        L2 = L2Penalization(Vh, alpha)
        z = dl.Function(Vh_CONTROL).vector()
        z.set_local(np.ones(dim))
        P_TRUE = dim * alpha
        self.compareFunctionValue(L2, z, P_TRUE, tol=1e-3)

        dz = dl.Function(Vh_CONTROL).vector()
        dz.set_local(np.random.randn(dim))
        self.finiteDifferenceGradient(L2, z, dz)
        self.finiteDifferenceHessian(L2, z, dz)





if __name__ == "__main__":
    unittest.main()
