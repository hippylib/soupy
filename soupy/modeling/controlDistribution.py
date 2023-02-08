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

import sys, os
import hippylib as hp 
import dolfin as dl 
import numpy as np

class UniformDistribution:
    """
    Class for sampling from a uniform distribution to `dl.Vector`
    """
    def __init__(self, Vh, a, b):
        """ 
        Constructor:
            :code: `Vh`: Function space for sample vectors
            :code: `a`: Lower bound
            :code: `b`: Upper bound
            :code: `ndim`: Dimension of sample vectors
        """
        self.Vh = Vh
        self.a = a
        self.b = b
        self.ndim = self.Vh.dim()
        self._dummy = dl.Function(Vh).vector()

    def init_vector(self, v):
        v.init( self._dummy.local_range() )

    def sample(self, out):
        assert out.mpi_comm().Get_size() == 1
        v = np.random.rand(self.ndim) * (self.b-self.a) + self.a
        out.set_local(v)

if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    DIM = 4
    NS = 1000
    mesh = dl.UnitSquareMesh(20, 20)
    Vh = dl.VectorFunctionSpace(mesh, "R", degree=0, dim=DIM)

    a = -1
    b = 2
    control_dist = UniformDistribution(Vh, a, b)

    z = dl.Vector()
    print(z)
    control_dist.init_vector(z)
    
    samples = []
    for i in range(NS):
        control_dist.sample(z)
        samples.append(z.get_local())
    samples = np.array(samples)
    
    plt.figure()
    for i in range(DIM):
        plt.subplot(221+i)
        plt.hist(samples[:,i])
    plt.show()
