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

import numpy as np 

class SmoothPlusApproximationSoftplus:
    """
    Implements the smooth approximation to the maximum function \
        :math:`\\max(0, t)` using the softplus function 
    """
    def __init__(self, epsilon=1e-2):
        self.epsilon = epsilon 

    def __call__(self, t):
        return self.epsilon * np.log(1 + np.exp(t/self.epsilon))

    def grad(self, t):
        return 1/(1 + np.exp(-t/self.epsilon))

    def hessian(self, t):
        return 1/(2*self.epsilon) * 1/(np.cosh(t/self.epsilon) + 1)


class SmoothPlusApproximationQuartic:
    """
    Implements the smooth approximation to the maximum function \
        :math:`\\max(0, t)` a piecewise quartic function
    """

    def __init__(self, epsilon=1e-2):
        self.epsilon = epsilon 

    def __call__(self, t):
        # return np.where(t >= 0, 1, 0) * np.where(t < self.epsilon, 1, 0) * (t**3/self.epsilon**2 - t**4/(2*self.epsilon**3)) \
        #         + np.where(t >= self.epsilon, 1, 0) * (t - self.epsilon/2)
        return np.where(t >= 0, 1, 0) * np.where(t < self.epsilon, t**3/self.epsilon**2 - t**4/(2*self.epsilon**3), 0) \
                + np.where(t >= self.epsilon, 1, 0) * (t - self.epsilon/2)

    def grad(self, t):
        return np.where(t >= 0, 1, 0) * np.where(t < self.epsilon, 3*t**2/self.epsilon**2 - 2*t**3/self.epsilon**3, 0) \
                + np.where(t >= self.epsilon, 1, 0)

    def hessian(self, t):
        return np.where(t >= 0, 1, 0) * np.where(t < self.epsilon, 6*t/self.epsilon**2 - 6*t**2/self.epsilon**3, 0)


def finiteDifferenceCheckApproximation(approx, t, dt, delta=1e-3):
    f0 = approx(t)
    f1 = approx(t + delta * dt)
    g0 = approx.grad(t) 
    g1 = approx.grad(t + delta * dt)
    h0 = approx.hessian(t)

    # print("Analytic gradient: ", g0)
    # print("FD gradient: ", (f1 - f0)/delta)
    print("Gradient error: ", np.linalg.norm(g0 - 1/delta * (f1-f0)))
    # print("Analytic hessian: ", h0)
    # print("FD hessian: ", (g1 - g0)/delta)
    print("Hessian error: ", np.linalg.norm(h0 - 1/delta * (g1-g0)))



if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    softplus = SmoothPlusApproximationSoftplus()
    quartic = SmoothPlusApproximationQuartic()
    
    t = np.linspace(-5, 5, 200)
    plt.figure()
    plt.plot(t, softplus(t), '-',label="Softplus")
    plt.plot(t, quartic(t), '--', label="Quartic")
    plt.legend()
    plt.show()

    np.random.seed(1)

    print("t is scalar")
    t = np.random.randn()
    dt = 1.0
    print("FD check softplus")
    finiteDifferenceCheckApproximation(softplus, t, dt)
    print("FD check quartic")
    finiteDifferenceCheckApproximation(quartic, t, dt)

    
    print("t is numpy array")
    t = np.linspace(-5, 5, 200)
    dt = np.ones_like(t)
    print("FD check softplus")
    finiteDifferenceCheckApproximation(softplus, t, dt)
    print("FD check quartic")
    finiteDifferenceCheckApproximation(quartic, t, dt)


