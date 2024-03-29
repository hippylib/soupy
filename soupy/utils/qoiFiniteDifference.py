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
import numpy as np 
from ..modeling.variables import STATE, PARAMETER, ADJOINT, CONTROL

try:
    import matplotlib.pyplot as plt 
except:
    pass

def qoiFiniteDifference(Vh, qoi, x, du, order=1, delta=1e-4, plotting=False):
    """
    Finite difference checks the gradient of the quantity of interest with respect to
    the state variable, as well as the parameter and control variables

    :param Vh: List of function spaces for the state, parameter, adjoint, and control variables
    :type Vh: list of :py:class:`dolfin.FunctionSpace`
    :param qoi: Quantity of interest to check
    :type qoi: :py:class:`soupy.ControlQoI`
    :param x: The list of state, parameter, adjoint, and control variables
    :type x: list of :py:class:`dolfin.Vector` or similar
    :param du: The perturbation to the state variable
    :type du: :py:class:`dolfin.Vector` or similar
    :param order: Order of derivative to check. 1 for gradient, 2 for Hessian
    :type order: int
    :param delta: The finite difference step size
    :type delta: float
    :param plotting: If :code:`true`, plots the finite difference Hessian and analytic Hessian
    :type plotting: bool
    """
    x2 = [None, None, None, None]
    x2[STATE] = dl.Vector(x[STATE])
    for i in [PARAMETER, ADJOINT, CONTROL]:
        x2[i] = x[i]
    x2[STATE].axpy(delta, du)

    q1 = qoi.cost(x)
    q2 = qoi.cost(x2)
    fd_grad = (q2 - q1)/delta
    gu = dl.Function(Vh[STATE]).vector()
    qoi.grad(STATE, x, gu)

    exact_grad = gu.inner(du)
    print("\nFinite difference checking gradients")
    print("Analytic state derivative: %g" %exact_grad)
    print("Finite diff state derivative: %g" %fd_grad)

    gm = dl.Function(Vh[PARAMETER]).vector()
    gz = dl.Function(Vh[CONTROL]).vector()
    qoi.grad(PARAMETER, x, gm)
    qoi.grad(CONTROL, x, gz)
    print("Analytic parameter derivative: %g" %gm.inner(gm))
    print("Analytic control derivative: %g" %gm.inner(gm))
    print("")

    if order == 2:
        print("Finite difference checking Hessians")
        Hdu = dl.Function(Vh[STATE]).vector()
        g2 = dl.Function(Vh[STATE]).vector()
        qoi.setLinearizationPoint(x)
        qoi.apply_ij(STATE, STATE, du, Hdu)
        qoi.grad(STATE, x2, g2)
        
        Hdu_np = Hdu.get_local()
        Hdiff_np = (g2.get_local() - gu.get_local())/delta
        print("Hdu diff norm: %g" %(np.linalg.norm(Hdu_np - Hdiff_np)))
        print("")
        
        if plotting:
            plt.figure()
            plt.plot(Hdu_np)
            plt.plot(Hdiff_np)
            


