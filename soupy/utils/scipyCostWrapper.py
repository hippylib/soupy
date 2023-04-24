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

from mpi4py import MPI
from ..modeling import STATE, PARAMETER, ADJOINT, CONTROL

class ScipyCostWrapper:
    """
    Class to interface the controlCostFunctional with a
    scipy optimizer. Converts inputs to functions taking
    and returning numpy arrays
    """
    def __init__(self, controlCostFunctional, verbose=False):
        self.cost_functional = controlCostFunctional
        self.z_help = self.cost_functional.generate_vector(CONTROL)
        self.g_help = self.cost_functional.generate_vector(CONTROL)
        self.verbose = verbose
        self.n_func = 0
        self.n_grad = 0

    def cost(self, z_np):
        self.z_help.set_local(z_np)
        self.z_help.apply("")
        cost_value = self.cost_functional.cost(self.z_help, order=0)
        self.n_func += 1

        if self.verbose and MPI.COMM_WORLD.Get_rank() == 0:
            print("Cost: ", cost_value)

        return cost_value

    def grad(self, z_np):
        self.z_help.set_local(z_np)
        self.z_help.apply("")
        self.cost_functional.cost(self.z_help, order=1)
        self.cost_functional.costGrad(self.z_help, self.g_help)
        if self.verbose and MPI.COMM_WORLD.Get_rank() == 0:
            print("Gradient evaluation")
        self.n_grad += 1

        return self.g_help.get_local()

    def costHessian(self, z_np, zhat_np):
        pass

    def function(self):
        return lambda x : self.cost(x)

    def jac(self):
        return lambda x : self.grad(x)



