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

import dolfin as dl
from .variables import STATE, CONTROL
from .augmentedVector import AugmentedVector


class Penalization:
    def init_vector(self, z):
        raise NotImplementedError("Child class should implement init_vector")

    def cost(self, z):
        raise NotImplementedError("Child class should implement cost")

    def grad(self, z, out):
        raise NotImplementedError("Child class should implement grad")

    def hessian(self, z, zhat, out):
        raise NotImplementedError("Child class should implement hessian")


class MultiPenalization(Penalization):
    """
    Class for a sum of penalization terms 
    """
    def __init__(self, Vh, penalization_list, alpha_list=None):
        """
        - :code: `Vh` function space for STATE, PARAMETER, ADJOINT, CONTROL
        - :code: `penalization_list` a list of Penalization objects
        - :code: `alpha_list` optional list of weights, assumed to all be 1 
            if None is given
        """
        self.Vh = Vh 
        self.helper = dl.Function(Vh[CONTROL]).vector()
        self.penalization_list = penalization_list

        if alpha_list is not None:
            assert len(alpha_list) == len(penalization_list)
            self.alpha_list = alpha_list
        else:
            self.alpha_list = [1.0] * len(self.penalization_list)

    
    def cost(self, z):
        cost = 0
        for alpha, penalization in zip(self.alpha_list, self.penalization_list):
            cost += alpha * penalization.cost(z)
        return cost 

    def grad(self, z, out):
        out.zero()
        for alpha, penalization in zip(self.alpha_list, self.penalization_list):
            penalization.grad(z, self.helper)
            out.axpy(alpha, self.helper)

    def hessian(self, z, zhat, out):
        out.zero()
        for alpha, penalization in zip(self.alpha_list, self.penalization_list):
            penalization.hessian(z, zhat, self.helper)
            out.axpy(alpha, self.helper)


class L2Penalization(Penalization):
    """
    L2 integral over the domain
        P(z) = \alpha \int_{\Omega) |z|^2 dx 

    For finite dimensional controls `z`, this amounts to a little \ell_2 norm
    In this case, `Vh[soupy.CONTROL]` needs to be a 
    `dolfin.VectorFunctionSpace` of reals
    """
    def __init__(self, Vh, alpha):
        """
        - :code: `Vh` function space for STATE, PARAMETER, ADJOINT, CONTROL
        - :code: `alpha` weighting factor
        """
        self.Vh = Vh
        self.alpha = alpha

        z_trial = dl.TrialFunction(self.Vh[CONTROL])
        z_test = dl.TestFunction(self.Vh[CONTROL])

        # Do we need a backend type?
        self.M = dl.assemble(dl.inner(z_trial, z_test) * dl.dx)
        self.Mz = dl.Function(self.Vh[CONTROL]).vector()

    def init_vector(self, v, dim=0):
        if isinstance(v, AugmentedVector):
            pass 
        else:
            self.M.init_vector(v, dim)


    def cost(self, z):
        if isinstance(z, AugmentedVector):
            z_sub = z.get_vector()
            self.M.mult(z_sub, self.Mz)
            return self.alpha * z_sub.inner(self.Mz)
        else:
            self.M.mult(z, self.Mz)
            return self.alpha * z.inner(self.Mz)

    def grad(self, z, out):
        out.zero()
        if isinstance(z, AugmentedVector):
            out_sub = out.get_vector()
            z_sub = z.get_vector() 
            self.M.mult(z_sub, self.Mz)
            out_sub.axpy(2.0*self.alpha, self.Mz)
        else:
            self.M.mult(z, self.Mz)
            out.axpy(2.0*self.alpha, self.Mz)

    def hessian(self, z, zhat, out):
        out.zero()
        if isinstance(z, AugmentedVector):
            out_sub = out.get_vector()
            zhat_sub = zhat.get_vector() 
            self.M.mult(zhat_sub, self.Mz)
            out_sub.axpy(2.0*self.alpha, self.Mz)
        else:
            self.M.mult(zhat, self.Mz)
            out.axpy(2.0*self.alpha, self.Mz)



class WeightedL2Penalization(Penalization):
    """
    A weighted L2 norm penalization 
        P(z) = z^T M z 
    where M is some symmetric positive definite weight matrix
    """
    def __init__(self, Vh, M, alpha):
        """
        - :code: `Vh` function space for STATE, PARAMETER, ADJOINT, CONTROL
        - :code: `M` weighting matrix with method `mult`
        - :code: `alpha` weighting factor
        """
        self.Vh = Vh
        self.M = M 
        self.alpha = alpha
        self.Mz = dl.Function(self.Vh[CONTROL]).vector()

    def init_vector(self, v, dim=0):
        if isinstance(v, AugmentedVector):
            pass 
        else:
            self.M.init_vector(v, dim)


    def cost(self, z):
        if isinstance(z, AugmentedVector):
            z_sub = z.get_vector()
            self.M.mult(z_sub, self.Mz)
            return self.alpha * z_sub.inner(self.Mz)
        else:
            self.M.mult(z, self.Mz)
            return self.alpha * z.inner(self.Mz)

    def grad(self, z, out):
        out.zero()
        if isinstance(z, AugmentedVector):
            out_sub = out.get_vector()
            z_sub = z.get_vector() 
            self.M.mult(z_sub, self.Mz)
            out_sub.axpy(2.0*self.alpha, self.Mz)
        else:
            self.M.mult(z, self.Mz)
            out.axpy(2.0*self.alpha, self.Mz)

    def hessian(self, z, zhat, out):
        out.zero()
        if isinstance(z, AugmentedVector):
            out_sub = out.get_vector()
            zhat_sub = zhat.get_vector() 
            self.M.mult(zhat_sub, self.Mz)
            out_sub.axpy(2.0*self.alpha, self.Mz)
        else:
            self.M.mult(zhat, self.Mz)
            out.axpy(2.0*self.alpha, self.Mz)

