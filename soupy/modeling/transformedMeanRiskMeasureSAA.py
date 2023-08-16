
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

from hippylib import ParameterList, Random

from .riskMeasure import RiskMeasure
from .variables import STATE, PARAMETER, ADJOINT, CONTROL
from .controlModelHessian import ControlModelHessian

from ..collectives import NullCollective, MultipleSamePartitioningPDEsCollective, \
        MultipleSerialPDEsCollective, allocate_process_sample_sizes


def transformedMeanRiskMeasureSAASettings(data = {}):
    data['sample_size'] = [100,'Number of Monte Carlo samples']
    data['seed'] = [1, 'rng seed for sampling']
    data['inner_function'] = [IdentityFunction(), 'Transformation of Q inside the expectation']
    data['outer_function'] = [IdentityFunction(), 'Transformation of Q outside the expectation']
    return ParameterList(data)

class IdentityFunction:
    def __call__(self, x):
        return x

    def grad(self, x):
        return 1

    def hessian(self, x):
        return 0


class FunctionWrapper:
    """
    Wrapper for a function that can be called with a single argument \
        Will use finite difference for gradient and Hessian if not provided
    """
    def __init__(self, function, grad=None, hessian=None, delta=1e-4):
        self._function = function 
        self._grad = grad
        self._hessian = hessian 
        self._delta = delta

    def __call__(self, x):
        return self._function(x)

    def grad(self, x):
        if self._grad is None:
            f0 = self._function(x)            
            f1 = self._function(x + self._delta)
            return (f1 - f0)/self._delta
        else:
            return self._grad(x)

    def hessian(self, x):
        if self._hessian is None:
            g0 = self.grad(x)
            g1 = self.grad(x + self._delta)
            return (g1 - g0)/self._delta
        else:
            return self._hessian(x)



class TransformedMeanRiskMeasureSAA(RiskMeasure):
    """
    Sample average approximation for a risk measure of type
        
        .. math:: \\rho[Q](z) = g \left( \mathbb{E}_m[f(Q(m,z))] \\right) 

    with sample parallelism using MPI, where the inner and outer functions, 
    :math:`f` and :math:`g` 
    can be supplied by the user 

    .. note:: currently does not support simultaneous sample and mesh partition parallelism 
    """

    def __init__(self, control_model, prior, settings = transformedMeanRiskMeasureSAASettings(), comm_sampler=mpi4py.MPI.COMM_WORLD):
        """
        Constructor:

        :param control_model: control model containing the :code:`soupy.PDEVariationalControlProblem`
            and :code:`soupy.ControlQoI`
        :type control_model: :py:class:`soupy.ControlModel`
        :param prior: The prior distribution for the random parameter
        :type prior: :py:class:`hippylib.Prior`
        :param settings: Additional settings
        :type settings: :py:class:`hippylib.ParameterList`
        :param comm_sampler: MPI communicator for sample parallelism 
        :type comm_sampler: :py:class:`mpi4py.MPI.Comm`
        """
        self.model = control_model
        self.prior = prior
        self.settings = settings
        self.sample_size = self.settings['sample_size']
        self.inner_function = self.settings['inner_function']
        self.outer_function = self.settings['outer_function']

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


        # Aggregate components for computing cost, grad, hess
        self.fq_bar = 0 

        self.f_prime_g_bar = self.model.generate_vector(CONTROL)

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
        If :code:`component` is :code:`STATE`, :code:`PARAMETER`, :code:`ADJOINT`, \
            or :code:`CONTROL`, return a vector corresponding to that function space. \
            If :code:`component` is :code:`"ALL"`, \
            Generate the list of vectors :code:`x = [u,m,p,z]`
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
            self.fq_bar = 0 

            # Make sure previous adjoint solve is no longer valid 
            self.has_adjoint_solve = False 
            new_adjoint_solve = True 
            # print("Solving new adjoint")
            self.f_prime_g_bar.zero()

        elif self.has_adjoint_solve:
            # If it's not a new forward solve, check if we already have an adjoint 
            new_adjoint_solve = False 
        else:
            # If we don't already have an adjoint, compute a new one 
            # print("Solving new adjoint")
            new_adjoint_solve = True 
            self.f_prime_g_bar.zero()
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
                    self.fq_bar += self.inner_function(qi)/self.sample_size

                if order >= 1 and new_adjoint_solve:
                    self.model.solveAdj(x[ADJOINT], x)
                    self.model.evalGradientControl(x, self.g_mc[i])
                    self.f_prime_g_bar.axpy(self.inner_function.grad(self.q_samples[i])/self.sample_size, self.g_mc[i])
            
            if new_forward_solve:
                self.fq_bar = self.collective.allReduce(self.fq_bar, "SUM")
            
            if order >= 1 and new_adjoint_solve:
                self.collective.allReduce(self.f_prime_g_bar, "SUM")

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
        return self.outer_function(self.fq_bar)

    def grad(self, g):
        """
        Evaluates the gradient of the risk measure once components have been computed

        :param g: (Dual of) the gradient of the risk measure to store result in
        :type g: :py:class:`dolfin.Vector`

        .. note:: Assumes :code:`self.computeComponents` has been called with :code:`order >= 1`
        """
        g.zero()
        g.axpy(self.outer_function.grad(self.fq_bar), self.f_prime_g_bar)


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

            # Apply :math:`H_i z` the qoi hessian at sample
            hessian_scale_factor = self.inner_function.grad(qi) * self.outer_function.grad(self.fq_bar)
            Hzhat.axpy(hessian_scale_factor/self.sample_size, self.Hi_zhat) 

            # Apply :math:`g_i g_i^T`for qoi gradients at sample
            gradient_scale_factor = gi.inner(zhat) * self.inner_function.hessian(qi) * self.outer_function.grad(self.fq_bar)
            Hzhat.axpy(gradient_scale_factor/self.sample_size, gi)
        
        self.collective.allReduce(Hzhat, "SUM")

        # Apply :math:`\bar{g} \bar{g}^T` for mean qoi gradients
        mean_gradient_scale_factor = self.f_prime_g_bar.inner(zhat) * self.outer_function.hessian(self.fq_bar)
        Hzhat.axpy(mean_gradient_scale_factor, self.f_prime_g_bar)


    def gather_samples(self):
        """
        Gather the QoI samples from all processes

        :return: An array of the sample QoI values
        :return type: :py:class:`numpy.ndarray`
        """
        q_all = np.zeros(self.sample_size) 
        self.comm_sampler.Allgatherv(self.q_samples, [q_all, self.sample_size_allprocs])
        return q_all 


