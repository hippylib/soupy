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

import logging 
import numpy as np 
import dolfin as dl 
import mpi4py

import sys, os
from hippylib import ParameterList, Random

from .riskMeasure import RiskMeasure
from .variables import STATE, PARAMETER, ADJOINT, CONTROL
from .controlModelHessian import ControlModelHessian

from ..collectives import NullCollective, MultipleSamePartitioningPDEsCollective, \
        MultipleSerialPDEsCollective, allocate_process_sample_sizes


def meanVarRiskMeasureSAASettings(data = {}):
    data['sample_size'] = [100,'Number of Monte Carlo samples']
    data['beta'] = [0,'Weighting factor for variance']
    data['seed'] = [1, 'rng seed for sampling']

    return ParameterList(data)

class MeanVarRiskMeasureSAA(RiskMeasure):
    """
    Mean + variance risk measure using sample average approximation
        
        .. math:: \\rho[Q](z) = \mathbb{E}_m[Q(m,z)] + \\beta \mathbb{V}_m[Q(m,z)]

    with sample parallelism using MPI

    .. note:: currently does not support simultaneous sample and mesh partition parallelism 
    """

    def __init__(self, control_model, prior, settings = meanVarRiskMeasureSAASettings(), comm_sampler=mpi4py.MPI.COMM_WORLD):
        """
        Constructor:

        :param control_model: control model containing the :code:`soupy.PDEVariationalControlProblem`
            and :code:`soupy.ControlQoI`
        :type control_model: :py:class:`soupy.ControlModel`
        :param prior: The prior distribution for the random parameter
        :type prior: :py:class:`hippylib.Prior`
        :param settings: additional settings
        :type settings: :py:class:`hippylib.ParameterList`
        :param comm_sampler: MPI communicator for sample parallelism 
        :type comm_sampler: :py:class:`mpi4py.MPI.Comm`
        """
        self.model = control_model
        self.prior = prior
        self.settings = settings
        self.sample_size = self.settings['sample_size']
        self.beta = settings['beta']

        assert self.sample_size >= comm_sampler.Get_size(), \
            "Total samples %d needs to be greater than MPI size %d" %(self.sample_size, comm_sampler.Get_size())
        self.comm_sampler = comm_sampler 
        self.sample_size_allprocs = allocate_process_sample_sizes(self.sample_size, self.comm_sampler)
        self.sample_size_proc = self.sample_size_allprocs[self.comm_sampler.rank]

        if comm_sampler.Get_size() == 1:
            self.collective = NullCollective()
        else:
            self.collective = MultipleSerialPDEsCollective(self.comm_sampler)

        # Within process 
        self.q_samples = np.zeros(self.sample_size_proc)

        # self.q_bar_proc = np.array([0.])
        # self.g_bar_proc = self.model.generate_vector(CONTROL)
        # self.qg_bar_proc = self.model.generate_vector(CONTROL)

        # Aggregate components for computing cost, grad, hess
        self.q_bar = 0
        self.q2_bar = 0 
        self.g_bar = self.model.generate_vector(CONTROL)
        self.qg_bar = self.model.generate_vector(CONTROL)

        # For sampling
        self.noise = dl.Vector(self.model.problem.Vh[STATE].mesh().mpi_comm()) # use the mesh mpi comm 
        self.prior.init_vector(self.noise, "noise")
        rng = Random(seed=self.settings['seed'])

        # Generate samples for m 
        self.x_mc = [] 
        self.g_mc = [] 
        self.z = self.model.generate_vector(CONTROL)

        if mpi4py.MPI.COMM_WORLD.rank == 0:
            logging.info("Initial sampling of stochastic parameter")

        # Burn in for parallel sampling  
        n_burn = int(np.sum(self.sample_size_allprocs[:self.comm_sampler.rank]))
        for i in range(n_burn):
            rng.normal(1.0, self.noise)
        
        # Actual sampling 
        for i in range(self.sample_size_proc):
            ui = self.model.generate_vector(STATE)
            mi = self.model.generate_vector(PARAMETER)
            pi = self.model.generate_vector(ADJOINT) 
            x = [ui, mi, pi, self.z]
            rng.normal(1.0, self.noise)
            self.prior.sample(self.noise, mi)
            self.x_mc.append(x)

            g = self.model.generate_vector(CONTROL)
            self.g_mc.append(g)

        if mpi4py.MPI.COMM_WORLD.rank == 0:
            logging.info("Done sampling")

        # Helper variable 
        self.diff_helper= self.model.generate_vector(CONTROL)
        self.has_adjoint_solve = False
        self.has_forward_solve = False

        # For Hessian computations 
        self.control_model_hessian = None 
        self.Hi_zhat = self.model.generate_vector(CONTROL)


    def generate_vector(self, component = "ALL"):
        """
        If :code:`components` is :code:`STATE`, :code:`PARAMETER`, :code:`ADJOINT`, \
            or :code:`CONTROL`, return a vector corresponding to that function space. \
            If :code:`components` is :code:`"ALL"`, \
            generate the list of vectors :code:`x = [u,m,p,z]`
        """
        return self.model.generate_vector(component)

    def computeComponents(self, z, order=0, **kwargs):
        """
        Computes the components for the evaluation of the risk measure

        :param z: the control variable
        :type z: :py:class:`dolfin.Vector`
        :param order: Order of the derivatives needed.
            :code:`0` for cost, :code:`1` for gradient, :code:`2` for Hessian
        :type order: int 
        :param kwargs: dummy keyword arguments for compatibility 
        """
        # Check if a new control variable is used 
        new_forward_solve = False 
        self.diff_helper.zero()
        self.diff_helper.axpy(1.0, self.z)
        self.diff_helper.axpy(-1.0, z)
        diff_norm = np.sqrt(self.diff_helper.inner(self.diff_helper))
        
        # Check if new forward solve is needed 
        if diff_norm > dl.DOLFIN_EPS or not self.has_forward_solve:
            # Update control variable (changes all samples)
            # Ask that new forward and adjoint solves are computed 
            self.z.zero()
            self.z.axpy(1.0, z)
            new_forward_solve = True
            logging.info("Using new forward solve")

        # Check if a new adjoint solve is needed 
        if new_forward_solve:
            # Reset the averages for new solve 
            self.q_bar = 0 
            self.q2_bar = 0 

            # Make sure previous adjoint solve is no longer valid 
            self.has_adjoint_solve = False 
            new_adjoint_solve = True 
            # print("Solving new adjoint")
            self.g_bar.zero()
            self.qg_bar.zero()

        elif self.has_adjoint_solve:
            # If it's not a new forward solve, check if we already have an adjoint 
            new_adjoint_solve = False 
        else:
            # If we don't already have an adjoint, compute a new one 
            # print("Solving new adjoint")
            new_adjoint_solve = True 
            self.g_bar.zero()
            self.qg_bar.zero()
            # In this case, self.has_adjoint_solve is already False

        # Now actually compute the solves
        if new_forward_solve or new_adjoint_solve:
            for i in range(self.sample_size_proc):
                x = self.x_mc[i]

                if new_forward_solve:
                    # Solve state           
                    self.model.solveFwd(x[STATE], x)
                    
                    # Compute the averages
                    qi = self.model.cost(x) 
                    self.q_samples[i] = qi 
                    self.q_bar += qi/self.sample_size
                    self.q2_bar += qi**2/self.sample_size

                if order >= 1 and new_adjoint_solve:
                    self.model.solveAdj(x[ADJOINT], x)
                    self.model.evalGradientControl(x, self.g_mc[i])
                    self.g_bar.axpy(1/self.sample_size, self.g_mc[i])
                    self.qg_bar.axpy(self.q_samples[i]/self.sample_size, self.g_mc[i])
            
            if new_forward_solve:
                self.q_bar = self.collective.allReduce(self.q_bar, "SUM")
                self.q2_bar = self.collective.allReduce(self.q2_bar, "SUM")
            
            if order >= 1 and new_adjoint_solve:
                self.collective.allReduce(self.g_bar, "SUM")
                self.collective.allReduce(self.qg_bar, "SUM")

                # We have computed a new adjoint solve 
                # Don't need to alter if it's any other case 
                self.has_adjoint_solve = True 

            # Will always be true after solving 
            self.has_forward_solve = True 
    
    def cost(self):
        """
        Evaluates the value of the risk measure once components have been computed

        :return: Value of the cost functional

        .. note:: Assumes :code:`computeComponents` has been called with :code:`order>=0`
        """
        return self.q_bar + self.beta * (self.q2_bar - self.q_bar**2)

    def grad(self, g):
        """
        Evaluates the gradient of the risk measure once components have been computed

        :param g: (Dual of) the gradient of the risk measure to store result in
        :type g: :py:class:`dolfin.Vector`

        .. note:: Assumes :code:`self.computeComponents` has been called with :code:`order >= 1`
        """
        g.zero()
        g.axpy(1.0, self.g_bar)
        g.axpy(2*self.beta, self.qg_bar)
        g.axpy(-2*self.beta*self.q_bar, self.g_bar)


    def hessian(self, zhat, Hzhat):
        """
        Apply the hessian of the risk measure once components have been computed \
            in the direction :code:`zhat`
        
        :param zhat: Direction for application of Hessian action of the risk measure
        :type zhat: :py:class:`dolfin.Vector`
        :param Hzhat: (Dual of) Result of the Hessian action of the risk measure 
            to store the result in
        :type Hzhat: :py:class:`dolfin.Vector`

        .. note:: Assumes :code:`self.computeComponents` has been called with :code:`order >= 1`
        """
        if self.control_model_hessian is None:
            self.control_model_hessian = ControlModelHessian(self.model)



        Hzhat.zero()
        for i in range(self.sample_size_proc):
            xi = self.x_mc[i] 
            gi = self.g_mc[i]
            qi = self.q_samples[i] 
            self.model.setLinearizationPoint(xi)
            self.control_model_hessian.mult(zhat, self.Hi_zhat)

            hessian_scale_factor = 1 + 2*self.beta*qi - 2*self.beta*self.q_bar
            
            # Apply :math:`H_i z` the qoi hessian at sample
            Hzhat.axpy(hessian_scale_factor/self.sample_size, self.Hi_zhat)

            # Apply :math:`g_i g_i^T`for qoi gradients at sample
            gradient_scale_factor = gi.inner(zhat) * 2*self.beta
            Hzhat.axpy(gradient_scale_factor/self.sample_size, gi)
        

        self.collective.allReduce(Hzhat, "SUM")

        # Apply :math:`\bar{g} \bar{g}^T` for mean qoi gradients
        mean_gradient_scale_factor = self.g_bar.inner(zhat) * 2*self.beta
        Hzhat.axpy(-mean_gradient_scale_factor, self.g_bar)


    def gather_samples(self):
        """
        Gather the QoI samples from all processes

        :return: An array of the sample QoI values
        :return type: :py:class:`numpy.ndarray`
        """
        q_all = np.zeros(self.sample_size) 
        self.comm_sampler.Allgatherv(self.q_samples, [q_all, self.sample_size_allprocs])
        return q_all 


