# Copyright (c) 2023, The University of Texas at Austin 
# & Georgia Institute of Technology

"""Reduced Hessian operator used by Taylor quadratic approximations."""

import dolfin as dl
import hippylib as hp
import time

from .variables import STATE, PARAMETER, ADJOINT, CONTROL


class ReducedHessianSVD:
    """Linear operator representing the reduced Hessian for Taylor expansions."""

    def __init__(self, pde, qoi, tol=1e-9):
        self.pde = pde
        self.qoi = qoi
        self.tol = tol

        self.rhs_fwd = pde.generate_state()
        self.rhs_adj = pde.generate_state()
        self.rhs_adj2 = pde.generate_state()
        self.rhs_adj3 = pde.generate_state()
        self.rhs_adj4 = pde.generate_state()
        self.mhelp = pde.generate_parameter()
        self.Hmhat1 = pde.generate_parameter()

    def init_vector(self, vec, dim):
        self.pde.init_parameter(vec)

    def mult(self, direction, out):
        value = self._apply(direction)
        out.zero()
        out.axpy(1.0, value)

    # This is the core routine that applies the reduced Hessian to a parameter increment; it solves incremental state and adjoint problems
    def _apply(self, mhat):
        xhat = self.pde.generate_state() # Initialize state increment
        yhat = self.pde.generate_state() # Initialize adjoint increment
        self.pde.apply_ij(ADJOINT, PARAMETER, mhat, self.rhs_fwd) # Compute RHS of forward increment (\partial_{vm} \bar{r} \hat{m})
        self.pde.solveIncremental(xhat, -self.rhs_fwd, False) # Solve for forward increment (false indicates that this is not incremental adjoint)

        self.pde.apply_ij(STATE, STATE, xhat, self.rhs_adj) # Compute RHS of adjoint increment (\partial_{uu} \bar{r} \hat{u})
        self.pde.apply_ij(STATE, PARAMETER, mhat, self.rhs_adj2) # Compute contribution from \partial_{um} \bar{r} \hat{m}
        self.rhs_adj.axpy(1.0, self.rhs_adj2)
        self.qoi.apply_ij(STATE, STATE, xhat, self.rhs_adj3) # contribution from Q_{uu}\hat{u}
        self.rhs_adj.axpy(1.0, self.rhs_adj3)
        self.qoi.apply_ij(STATE, PARAMETER, mhat, self.rhs_adj4) # Should be zero if Q doesn't explicitly depend on the parameter
        self.rhs_adj.axpy(1.0, self.rhs_adj4)

        self.pde.solveIncremental(yhat, -self.rhs_adj, True)

        self.pde.apply_ij(PARAMETER, PARAMETER, mhat, self.Hmhat1) # \partial_{mm} \bar{r} \hat{m}
        self.pde.apply_ij(PARAMETER, ADJOINT, yhat, self.mhelp) # \partial_{vm} \bar{r}^* \hat{v}
        self.Hmhat1.axpy(1.0, self.mhelp)
        self.pde.apply_ij(PARAMETER, STATE, xhat, self.mhelp) # \partial_{um} \bar{r}^* \hat{u}
        self.Hmhat1.axpy(1.0, self.mhelp)
        # The following four lines deal with the case when QoI explicitly depends on the parameter
        self.qoi.apply_ij(PARAMETER, ADJOINT, yhat, self.mhelp) 
        self.Hmhat1.axpy(1.0, self.mhelp)
        self.qoi.apply_ij(PARAMETER, STATE, xhat, self.mhelp)
        self.Hmhat1.axpy(1.0, self.mhelp)

        return self.Hmhat1.copy()

    def HessianInner(self, mhat1, mhat2):
        applied = self._apply(mhat1)
        return mhat2.inner(applied)
