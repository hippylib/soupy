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
import numpy as np 
from ..modeling.variables import STATE, PARAMETER, ADJOINT, CONTROL

try:
    import matplotlib.pyplot as plt 
except:
    pass

def penalizationFiniteDifference(Vh, penalization, z, dz, order=1, delta=1e-4, plotting=False):
    """
    Finite difference checks the gradient of the penalization 

    :param Vh: List of function spaces for the state, parameter, adjoint, and control variables
    :type Vh: list of :py:class:`dolfin.FunctionSpace`
    :param penalization: Penalization term to check
    :type penalization: :py:class:`soupy.Penalization`
    :param z: The control variable
    :type z: :py:class:`dolfin.Vector` or similar 
    :param dz: The perturbation to the control variable
    :type dz: :py:class:`dolfin.Vector` or similar 
    :param delta: The finite difference step size
    :type delta: float
    :plotting: If :code:`true`, plots the finite difference Hessian and analytic Hessian
    :type plotting: bool
    """

    z1 = dl.Vector(z)
    z2 = dl.Vector(z)
    z2.axpy(delta, dz)

    p1 = penalization.cost(z1)
    p2 = penalization.cost(z2)
    fd_grad = (p2 - p1)/delta
    g1 = dl.Function(Vh[CONTROL]).vector()
    penalization.grad(z, g1)
    exact_grad = g1.inner(dz)
    print("\nFinite difference checking gradients")
    print("Analytic control derivative: %g" %exact_grad)
    print("Finite diff control derivative: %g" %fd_grad)

    if order == 2:
        print("Finite difference checking Hessians")
        Hdz = dl.Function(Vh[CONTROL]).vector()
        g2 = dl.Function(Vh[CONTROL]).vector()
        penalization.grad(z2, g2)
        penalization.hessian(z1, dz, Hdz)
        Hdz_np = Hdz.get_local()
        Hdiff_np = (g2.get_local() - g1.get_local())/delta
        print("Hdz diff norm: %g" %(np.linalg.norm(Hdz_np - Hdiff_np)))
        print("")
        
        if plotting:
            plt.figure()
            plt.plot(Hdz_np)
            plt.plot(Hdiff_np, '--')
            


