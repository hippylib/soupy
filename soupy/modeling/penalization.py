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

import dolfin as dl
from .variables import STATE, CONTROL
from .augmentedVector import AugmentedVector


class Penalization:
    """
    Abstract class for the penalization on the control variable :math:`z`
    """
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
    Penalization term for the sum of individual penalties

        .. math:: P(z) = \sum_{i=1}^{n} \\alpha_i P_i(z)

    """
    def __init__(self, Vh, penalization_list, alpha_list=None):
        """
        Constructor 

        :param Vh: List of function spaces the state, parameter, adjoint, and control
        :type Vh: list of :py:class:`dolfin.FunctionSpace`        
        :param penalization_list: List of penalization objects
        :type penalization_list: list of :py:class:`Penalization` 
        :param alpha_list: List of weights for each penalization term
        :type alpha_list: list of floats
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
        """
        Compute the penalization at given control :math:`z`

        :param z: The control variable
        :type z: :py:class:`dolfin.Vector` or :py:class:`soupy.AugmentedVector`
        """
        cost = 0
        for alpha, penalization in zip(self.alpha_list, self.penalization_list):
            cost += alpha * penalization.cost(z)
        return cost 

    def grad(self, z, out):
        """
        Compute the gradient of the penalty at given control :math:`z`

        :param z: The control variable
        :type z: :py:class:`dolfin.Vector` or :py:class:`soupy.AugmentedVector`
        :param out: The assembled gradient vector
        :type out: :py:class:`dolfin.Vector` or :py:class:`soupy.AugmentedVector`
        """
        out.zero()
        for alpha, penalization in zip(self.alpha_list, self.penalization_list):
            penalization.grad(z, self.helper)
            out.axpy(alpha, self.helper)

    def hessian(self, z, zhat, out):
        """
        Compute the Hessian of the penalty at given control :math:`z`

        :param z: The control variable
        :type z: :py:class:`dolfin.Vector` or :py:class:`soupy.AugmentedVector`
        :param zhat: The direction for Hessian action
        :type zhat: :py:class:`dolfin.Vector` or :py:class:`soupy.AugmentedVector`
        :param out: The assembled Hessian action vector
        :type out: :py:class:`dolfin.Vector` or :py:class:`soupy.AugmentedVector`
        """
        out.zero()
        for alpha, penalization in zip(self.alpha_list, self.penalization_list):
            penalization.hessian(z, zhat, self.helper)
            out.axpy(alpha, self.helper)


class L2Penalization(Penalization):
    """
    :math:`L^2(\Omega)` integral over the domain

        .. math:: P(z) = \\alpha \int_{\Omega} |z|^2 dx 

    For finite dimensional controls `z`, this amounts to a little \
        :math:`\ell_2` norm \
        In this case, :code:`Vh[soupy.CONTROL]` needs to be a \
        :code:`dolfin.VectorFunctionSpace` of reals
    """
    def __init__(self, Vh, alpha):
        """
        Constructor 

        :param Vh: List of function spaces the state, parameter, adjoint, and control
        :type Vh: list of :py:class:`dolfin.FunctionSpace`        
        :param alpha: Weight for the penalization term
        :type alpha: float
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
        """
        Compute the penalization at given control :math:`z`

        :param z: The control variable
        :type z: :py:class:`dolfin.Vector` or :py:class:`AugmentedVector`
        """
        if isinstance(z, AugmentedVector):
            z_sub = z.get_vector()
            self.M.mult(z_sub, self.Mz)
            return self.alpha * z_sub.inner(self.Mz)
        else:
            self.M.mult(z, self.Mz)
            return self.alpha * z.inner(self.Mz)

    def grad(self, z, out):
        """
        Compute the gradient of the penalty at given control :math:`z`

        :param z: The control variable
        :type z: :py:class:`dolfin.Vector` or :py:class:`AugmentedVector`
        :param out: The assembled gradient vector
        :type out: :py:class:`dolfin.Vector` or :py:class:`AugmentedVector`
        """
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
        """
        Compute the Hessian of the penalty at given control :math:`z`

        :param z: The control variable
        :type z: :py:class:`dolfin.Vector` or :py:class:`soupy.AugmentedVector`
        :param zhat: The direction for Hessian action
        :type zhat: :py:class:`dolfin.Vector` or :py:class:`soupy.AugmentedVector`
        :param out: The assembled Hessian action vector
        :type out: :py:class:`dolfin.Vector` or :py:class:`soupy.AugmentedVector`
        """
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
        .. math:: P(z) = z^T M z 

    where :math:`M` is a symmetric positive definite weight matrix
    """
    def __init__(self, Vh, M, alpha):
        """
        Constructor: 

        :param Vh: List of function spaces the state, parameter, adjoint, and control
        :type Vh: list of :py:class:`dolfin.FunctionSpace`        
        :param M: Weighting matrix with method :code:`mult`
        :type M: :py:class:`dolfin.Matrix` like
        :param alpha: Weight for the penalization term
        :type alpha: float
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
        """
        Compute the penalization at given control :math:`z`

        :param z: The control variable
        :type z: :py:class:`dolfin.Vector` or :py:class:`soupy.AugmentedVector`
        """
        if isinstance(z, AugmentedVector):
            z_sub = z.get_vector()
            self.M.mult(z_sub, self.Mz)
            return self.alpha * z_sub.inner(self.Mz)
        else:
            self.M.mult(z, self.Mz)
            return self.alpha * z.inner(self.Mz)

    def grad(self, z, out):
        """
        Compute the gradient of the penalty at given control :math:`z`

        :param z: The control variable
        :type z: :py:class:`dolfin.Vector` or :py:class:`soupy.AugmentedVector`
        :param out: The assembled gradient vector
        :type out: :py:class:`dolfin.Vector` or :py:class:`soupy.AugmentedVector`
        """
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
        """
        Compute the Hessian of the penalty at given control :math:`z`

        :param z: The control variable
        :type z: :py:class:`dolfin.Vector` or :py:class:`soupy.AugmentedVector`
        :param zhat: The direction for Hessian action
        :type zhat: :py:class:`dolfin.Vector` or :py:class:`soupy.AugmentedVector`
        :param out: The assembled Hessian action vector
        :type out: :py:class:`dolfin.Vector` or :py:class:`soupy.AugmentedVector`
        """
        out.zero()
        if isinstance(z, AugmentedVector):
            out_sub = out.get_vector()
            zhat_sub = zhat.get_vector() 
            self.M.mult(zhat_sub, self.Mz)
            out_sub.axpy(2.0*self.alpha, self.Mz)
        else:
            self.M.mult(zhat, self.Mz)
            out.axpy(2.0*self.alpha, self.Mz)

