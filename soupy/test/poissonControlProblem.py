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
import math
import numpy as np
import hippylib as hp 
from soupy import STATE, PARAMETER, ADJOINT, CONTROL, PDEVariationalControlProblem


class UniformDistribution:
    """
    Class for sampling from a uniform distribution to `dl.Vector`
    Used only for tests 
    """
    def __init__(self, Vh, a, b):
        """ 
        Constructor:
            :code: `Vh`: Function space for sample vectors
            :code: `a`: Lower bound
            :code: `b`: Upper bound
            :code: `ndim`: Dimension of sample vectors
        """
        assert Vh.mesh().mpi_comm().size == 1
        self.Vh = Vh
        self.a = a
        self.b = b
        self.ndim = self.Vh.dim()
        self._dummy = dl.Function(Vh).vector()

    def init_vector(self, v):
        v.init( self._dummy.local_range() )

    def sample(self, out):
        v = np.random.rand(self.ndim) * (self.b-self.a) + self.a
        out.set_local(v)
        

def u_boundary(x, on_boundary):
    return on_boundary and (x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)



def setupPoissonPDEProblem(Vh, settings):
    anis_diff = dl.CompiledExpression(hp.ExpressionModule.AnisTensor2D(), degree = 1)
    anis_diff.set(settings['THETA0'], settings['THETA1'], settings['ALPHA'])
    m_mean_fun = dl.Function(Vh[PARAMETER])
    m_mean_fun.interpolate(dl.Constant(1.0))
    prior = hp.BiLaplacianPrior(Vh[PARAMETER], settings['GAMMA'], settings['DELTA'],\
                                anis_diff, mean=m_mean_fun.vector(), robin_bc=True)

    bc = dl.DirichletBC(Vh[STATE], dl.Expression("x[1]", degree=1), u_boundary)
    bc0 = dl.DirichletBC(Vh[STATE], dl.Constant(0.0), u_boundary)
    poisson_varf = PoissonVarfHandler(Vh, settings=settings)
    pde = PDEVariationalControlProblem(Vh, poisson_varf, bc, bc0, 
            is_fwd_linear=settings["LINEAR"])

    control_dist = UniformDistribution(Vh[CONTROL], 
            settings['STRENGTH_LOWER'],
            settings['STRENGTH_UPPER'])

    return pde, prior, control_dist



def poisson_control_settings():
    settings = {}
    settings['nx'] = 20
    settings['ny'] = 20

    settings['STRENGTH_UPPER'] = 1.
    settings['STRENGTH_LOWER'] = -1.
    settings['LINEAR'] = True


    settings['N_WELLS_PER_SIDE'] = 5
    settings['LOC_LOWER'] = 0.25
    settings['LOC_UPPER'] = 0.75
    settings['WELL_WIDTH'] = 0.1

    settings['GAMMA'] = 1.0
    settings['DELTA'] = 20.0
    settings['THETA0'] = 2.0
    settings['THETA1'] = 0.5
    settings['ALPHA'] = math.pi/4

    return settings


class PoissonVarfHandler:
    """
    """
    def __init__(self,Vh,settings = poisson_control_settings()):
        """
        """
        self.linear = settings['LINEAR']

        # Set up the control right hand side
        well_grid = np.linspace(settings['LOC_LOWER'],settings['LOC_UPPER'],settings['N_WELLS_PER_SIDE'])
        well_grid_x, well_grid_y = np.meshgrid(well_grid, well_grid)
        mollifier_list = [] 

        for i in range(settings['N_WELLS_PER_SIDE']):
            for j in range(settings['N_WELLS_PER_SIDE']):
                mollifier_list.append(
                        dl.Expression("a*exp(-(pow(x[0]-xi,2)+pow(x[1]-yj,2))/(pow(b,2)))", 
                            xi=well_grid[i], 
                            yj=well_grid[j], 
                            a=1/(2*math.pi*settings['WELL_WIDTH']**2),
                            b=settings['WELL_WIDTH'],
                            degree=2)
                        )

        self.mollifiers = dl.as_vector(mollifier_list)

        assert Vh[CONTROL].dim() == len(mollifier_list)


    def __call__(self,u,m,p,z):
        if self.linear:
            return dl.exp(m)*dl.inner(dl.grad(u),dl.grad(p))*dl.dx - dl.inner(self.mollifiers,z)*p*dl.dx
        else:
            return dl.exp(m)*dl.inner(dl.grad(u),dl.grad(p))*dl.dx + u**3*p*dl.dx  - dl.inner(self.mollifiers,z)*p*dl.dx



