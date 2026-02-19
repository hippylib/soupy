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

import sys, os
import math 
import dolfin as dl 
import numpy as np 
from mpi4py import MPI

import hippylib as hp
import soupy 


def bilaplacian_2D(Vh_parameter,gamma = 0.1,delta = 0.1, theta0 = 2.0, theta1 = 0.5,\
             alpha = np.pi/4,mean = None,robin_bc = False):
    """
    Return 2D BiLaplacian prior given function space and coefficients for Matern covariance
    """
    anis_diff = dl.CompiledExpression(hp.ExpressionModule.AnisTensor2D(), degree = 1)
    anis_diff.theta0 = theta0
    anis_diff.theta1 = theta1
    anis_diff.alpha = alpha

    return hp.BiLaplacianPrior(Vh_parameter, gamma, delta, anis_diff,mean = mean,robin_bc = robin_bc)



def semilinear_elliptic_control_settings():
	settings = {}
	settings['nx'] = 64
	settings['ny'] = 64
	settings['state_order'] = 1
	settings['parameter_order'] = 1

	settings['strength_upper'] = 4.
	settings['strength_lower'] = -4.
	settings['linear'] = False 

	settings['n_wells_per_side'] = 7
	settings['loc_lower'] = 0.1
	settings['loc_upper'] = 0.9
	settings['well_width'] = 0.08

	settings['mean'] = -1.0
	settings['gamma'] = 0.1
	settings['delta'] = 5.0
	settings['theta0'] = 2.0
	settings['theta1'] = 0.5
	settings['alpha'] = math.pi/4

	return settings


class SemilinearEllipticVarfHandler:
	"""
    Variational form handler for the semilinear elliptic PDE
	"""
	def __init__(self,Vh,settings = semilinear_elliptic_control_settings()):
		"""
        Constructor:
        
        :param Vh: List of function spaces
        :param settings: Settings from :code:`semilinear_elliptic_control_settings`

		"""
		self.linear = settings['linear']

		# Set up the control right hand side
		well_grid = np.linspace(settings['loc_lower'],settings['loc_upper'],settings['n_wells_per_side'])
		well_grid_x, well_grid_y = np.meshgrid(well_grid, well_grid)
		mollifier_list = [] 

		for i in range(settings['n_wells_per_side']):
			for j in range(settings['n_wells_per_side']):
				mollifier_list.append(
						dl.interpolate(dl.Expression("a*exp(-(pow(x[0]-xi,2)+pow(x[1]-yj,2))/(pow(b,2)))", 
                                    xi=well_grid[i], 
                                    yj=well_grid[j], 
                                    a=1/(2*math.pi*settings['well_width']**2),
                                    b=settings['well_width'],
                                    mpi_comm=Vh[soupy.STATE].mesh().mpi_comm(),
                                    degree=2),
                                Vh[soupy.STATE]
                            )
						)

		self.mollifiers = dl.as_vector(mollifier_list)
		assert Vh[soupy.CONTROL].dim() == len(mollifier_list), "Control dimension and number of sources do not match"


	def __call__(self,u,m,p,z):
		if self.linear:
			return dl.exp(m)*dl.inner(dl.grad(u),dl.grad(p))*dl.dx - dl.inner(self.mollifiers,z)*p*dl.dx
		else:
			return dl.exp(m)*dl.inner(dl.grad(u),dl.grad(p))*dl.dx + u**3*p*dl.dx  - dl.inner(self.mollifiers,z)*p*dl.dx


def setup_semilinear_elliptic_pde(settings, comm_mesh=MPI.COMM_WORLD):
    """
    Setup the semilinear elliptic PDE Problem

    :param settings: Problem settinsg from :code:`semilinear_elliptic_control_settings`
    :param comm_mesh: MPI communicator for the mesh. Set to :code:`MPI.COMM_SELF` 
        if doing sample parallel

    """
    assert comm_mesh.Get_size() == 1
    mesh = dl.UnitSquareMesh(comm_mesh, settings['nx'], settings['ny'])
    Vh_STATE = dl.FunctionSpace(mesh, 'Lagrange', settings["state_order"])
    Vh_PARAMETER = dl.FunctionSpace(mesh, 'Lagrange', settings["parameter_order"])
    n_control = settings['n_wells_per_side']**2
    Vh_CONTROL = dl.VectorFunctionSpace(mesh, "R", degree=0, dim=n_control)

    Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE, Vh_CONTROL]

    def u_boundary(x, on_boundary):
        return on_boundary 

    u_bdr = dl.Constant(0.0)
    u_bdr0 = dl.Constant(0.0)
    bc = dl.DirichletBC(Vh[soupy.STATE], u_bdr, u_boundary)
    bc0 = dl.DirichletBC(Vh[soupy.STATE], u_bdr0, u_boundary)
    pde_varf = SemilinearEllipticVarfHandler(Vh, settings = settings)
    pde = soupy.PDEVariationalControlProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=settings['linear'], lu_method="default")
    
    m_mean_fun = dl.Function(Vh[hp.PARAMETER])
    m_mean_fun.interpolate(dl.Constant(settings['mean']))
    prior = bilaplacian_2D(Vh[hp.PARAMETER], mean = m_mean_fun.vector(),
        gamma = settings['gamma'],
        delta = settings['delta'],
        theta0 = 2.0, theta1 = 0.5,alpha = np.pi/4)

    return mesh, pde, Vh, prior
