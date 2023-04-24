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
import math 

import numpy as np
import dolfin as dl

import os, sys
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

sys.path.append('../../')
from soupy import controlQoI, L2MisfitVarfHandler, \
    VariationalControlQoI, L2MisfitControlQoI, \
    STATE, PARAMETER, ADJOINT, CONTROL




def boundary_varf(u, m, z):
    return dl.exp(m) * u**3 * z**3 * dl.dx

def nonlinear_varf(u, m, z):
    return dl.exp(m) * u**3 * z**3 * dl.dx


class TestControlQoI(unittest.TestCase):

    def setUp(self):
        dl.dx = dl.dx(metadata={'quadrature_degree':2, "representation":'uflacs'})
        self.mesh = dl.UnitSquareMesh(32, 32)
        Vh_STATE = dl.FunctionSpace(self.mesh, "CG", 2)
        Vh_PARAMETER = dl.FunctionSpace(self.mesh, "CG", 1)
        Vh_CONTROL = dl.FunctionSpace(self.mesh, "CG", 1)
        self.Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE, Vh_CONTROL]

    
    def finiteDifferenceGradient(self, nonlinear_qoi, i):
        delta = 1e-5
        grad_tol = 1e-3

        # ud = dl.Function(self.Vh[STATE])
        # ud.interpolate(dl.Expression("sin(k*x[0])*cos(k*x[1])", k=math.pi, degree=2))
        # nonlinear_qoi = VariationalControlQoI(self.mesh, self.Vh, varf)


        u_fun = dl.Function(self.Vh[STATE])
        p_fun = dl.Function(self.Vh[ADJOINT])
        m_fun = dl.Function(self.Vh[PARAMETER])
        z_fun = dl.Function(self.Vh[CONTROL])
        dfun = dl.Function(self.Vh[i])

        u_fun.interpolate(dl.Expression("sin(x[0])*x[1] + x[0]*x[1]", degree=2))
        m_fun.interpolate(dl.Expression("cos(x[0])*x[1]*x[1] + x[0]", degree=2))
        z_fun.interpolate(dl.Expression("cos(x[0])*x[1] + exp(-pow(x[0]-0.5, 2) - pow(x[1]-0.5, 2))", degree=2))
        dfun.interpolate(dl.Expression("cos(2.88*x[0])*x[1] + exp(-pow(x[0]-0.5, 2) - pow(x[1]-0.5, 2))",
            degree=5))

        x = [u_fun.vector(), m_fun.vector(), p_fun.vector(), z_fun.vector()]
        grad = dl.Function(self.Vh[i]).vector()
        q0 = nonlinear_qoi.cost(x)
        nonlinear_qoi.grad(i, x, grad)

        x[i].axpy(delta, dfun.vector())
        q1 = nonlinear_qoi.cost(x)

        finite_diff = (q1 - q0)/delta
        symbolic_derivative = grad.inner(dfun.vector())
        print("Finite difference derivative (%d): %g" %(i, finite_diff))
        print("Symbolic derivative (%d): %g" %(i, symbolic_derivative))
        self.assertTrue(abs(finite_diff - symbolic_derivative) < grad_tol)

    def finiteDifferenceHessian(self, nonlinear_qoi, i, j):
        delta = 1e-5
        hess_tol = 1e-3


        u_fun = dl.Function(self.Vh[STATE])
        p_fun = dl.Function(self.Vh[ADJOINT])
        m_fun = dl.Function(self.Vh[PARAMETER])
        z_fun = dl.Function(self.Vh[CONTROL])
        dfun = dl.Function(self.Vh[j])

        u_fun.interpolate(dl.Expression("sin(x[0])*x[1] + x[0]*x[1]", degree=2))
        m_fun.interpolate(dl.Expression("cos(x[0])*x[1]*x[1] + x[0]", degree=2))
        z_fun.interpolate(dl.Expression("cos(x[0])*x[1] + exp(-pow(x[0]-0.5, 2) - pow(x[1]-0.5, 2))", degree=2))
        dfun.interpolate(dl.Expression("cos(2.88*x[0])*x[1] + exp(-pow(x[0]-0.5, 2) - pow(x[1]-0.5, 2))",
            degree=5))


        x = [u_fun.vector(), m_fun.vector(), p_fun.vector(), z_fun.vector()]
        grad = dl.Function(self.Vh[i]).vector()
        Hd = dl.Function(self.Vh[i]).vector()

        q0 = nonlinear_qoi.cost(x)
        nonlinear_qoi.grad(i, x, grad)
        nonlinear_qoi.setLinearizationPoint(x)
        nonlinear_qoi.apply_ij(i, j, dfun.vector(), Hd)

        x[j].axpy(delta, dfun.vector())

        q1 = nonlinear_qoi.cost(x)
        grad1 = dl.Function(self.Vh[i]).vector()
        nonlinear_qoi.grad(i, x, grad1)

        Hd_fd = (grad1 - grad)/delta
        hess_diff = Hd_fd - Hd 

        if np.linalg.norm(Hd.get_local()) > 0:
            hess_err = np.sqrt(hess_diff.inner(hess_diff))/np.linalg.norm(Hd.get_local())
        else:
            hess_err = np.sqrt(hess_diff.inner(hess_diff))

        print("Hessian difference (%d, %d): %g" %(i,j,hess_err))
        self.assertTrue(hess_err < hess_tol)


    def testFiniteDifference(self):
        varfs = [boundary_varf, nonlinear_varf]
        ud = dl.Function(self.Vh[STATE])
        ud.interpolate(dl.Expression("sin(k*x[0])*cos(k*x[1])", k=math.pi, degree=2))
        for varf in varfs:
            nonlinear_qoi = VariationalControlQoI(self.mesh, self.Vh, varf)
            for i in [STATE, PARAMETER, CONTROL]:
                self.finiteDifferenceGradient(nonlinear_qoi, i)
                for j in [STATE, PARAMETER, CONTROL]:
                    self.finiteDifferenceHessian(nonlinear_qoi, i, j)


    def testVariationalControlQoI(self):
        tol = 1e-3
        ud = dl.Function(self.Vh[STATE])
        ud.interpolate(dl.Expression("sin(k*x[0])*cos(k*x[1])", k=math.pi, degree=5))
        chi = dl.Expression("0.5", degree=5)
        l2varf = L2MisfitVarfHandler(ud, chi=chi)
        l2qoi = VariationalControlQoI(self.mesh, self.Vh, l2varf)
        u = dl.Function(self.Vh[STATE])
        m = dl.Function(self.Vh[PARAMETER])
        z = dl.Function(self.Vh[CONTROL])
        x = [u.vector(), m.vector(), None, z.vector()]

        exact_q = 0.125
        q = l2qoi.cost(x)
        print("Evaluated QoI: %g"%(q))
        print("Exact QoI: %g"%(exact_q))
        assert(abs(q - exact_q) < tol)


    def testL2MisfitQoI(self):
        ud = dl.Function(self.Vh[STATE])
        ud.interpolate(dl.Expression("sin(k*x[0])*cos(k*x[1])", k=math.pi, degree=2))
        nonlinear_qoi = L2MisfitControlQoI(self.mesh, self.Vh, ud.vector())
        for i in [STATE, PARAMETER, CONTROL]:
            self.finiteDifferenceGradient(nonlinear_qoi, i)
            for j in [STATE, PARAMETER, CONTROL]:
                self.finiteDifferenceHessian(nonlinear_qoi, i, j)


if __name__ == "__main__":
    unittest.main()

