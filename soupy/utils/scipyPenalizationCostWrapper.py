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

from mpi4py import MPI
from ..modeling import STATE, PARAMETER, ADJOINT, CONTROL, AugmentedVector

class ScipyPenalizationCostWrapper:
    """
    Class to interface the :py:class:`soupy.Penalization` with a \
        scipy optimizer. Converts inputs to functions taking \
        and returning :code:`numpy` arrays
    """
    def __init__(self, penalization, is_augmented_vector=False, verbose=False):
        """
        Constructor

        :param penalization: The penalization to wrap
        :type penalization: :py:class:`Penalization`
        """
        self.penalization = penalization 
        self.z_help = self.penalization.generate_control()
        self.z_hat_help = self.penalization.generate_control()
        self.z_out_help = self.penalization.generate_control()
        self.verbose = verbose
        self.is_augmented_vector = is_augmented_vector

        if self.is_augmented_vector:
            self.z_help = AugmentedVector(self.z_help)
            self.z_hat_help = AugmentedVector(self.z_hat_help)
            self.z_out_help = AugmentedVector(self.z_out_help)

        self.n_func = 0
        self.n_grad = 0
        self.n_hess = 0 

    def cost(self, z_np):
        """
        Evaluates the cost functional at given control 

        :param z_np: The control as a numpy array
        :type z_np: :py:class:`numpy.ndarray`

        :returns: The value of the cost functional
        :return type: float
        """
        self.z_help.set_local(z_np)
        self.z_help.apply("")
        cost_value = self.cost_functional.cost(self.z_help, order=0)
        self.n_func += 1

        if self.verbose and MPI.COMM_WORLD.Get_rank() == 0:
            print("Cost: ", cost_value)

        return cost_value

    def grad(self, z_np):
        """
        Evaluates the gradient of the cost functional at given control 

        :param z_np: The control as a numpy array
        :type z_np: :py:class:`numpy.ndarray`

        :returns: The gradient 
        :return type: :py:class:`numpy.ndarray`
        """

        self.z_help.set_local(z_np)
        self.z_help.apply("")
        self.cost_functional.cost(self.z_help, order=1)
        self.cost_functional.grad(self.z_out_help)
        if self.verbose and MPI.COMM_WORLD.Get_rank() == 0:
            print("Gradient evaluation")
        self.n_grad += 1

        return self.z_out_help.get_local()

    def hessian(self, z_np, zhat_np):
        self.z_help.set_local(z_np)
        self.z_help.apply("")

        self.z_hat_help.set_local(zhat_np)
        self.z_hat_help.apply("")

        self.cost_functional.cost(self.z_help, order=2)
        self.cost_functional.hessian(self.z_hat_help, self.z_out_help)
        if self.verbose and MPI.COMM_WORLD.Get_rank() == 0:
            print("Hessian action evaluation")
        self.n_hess += 1

        return self.z_out_help.get_local()


    def function(self):
        """
        :returns: A function that evaluates the cost functional at a given control
        """
        return lambda x : self.cost(x)

    def jac(self):
        """
        :returns: A function that evaluates the gradient of the \
            cost functional at a given control
        """
        return lambda x : self.grad(x)

    def hessp(self):
        """
        :returns: A function that evaluates the Hessian of the \
            cost functional at a given control :code:`x`: applied in a direction :code:`p`
        """
        return lambda x, p : self.hessian(x,p)

