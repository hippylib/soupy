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
import scipy.optimize

from hippylib import ParameterList, Random

from .riskMeasure import RiskMeasure
from .variables import STATE, PARAMETER, ADJOINT, CONTROL
from .controlModelHessian import ControlModelHessian

from ..collectives import NullCollective, MultipleSamePartitioningPDEsCollective, \
        MultipleSerialPDEsCollective, allocate_process_sample_sizes
from .smoothPlusApproximation import SmoothPlusApproximationQuartic, SmoothPlusApproximationSoftplus
from .augmentedVector import AugmentedVector



def sample_superquantile(samples, beta):
    """
    Evaluate superquantile from samples 
    """ 
    quantile = np.percentile(samples, beta * 100) 
    return quantile + np.mean(np.maximum(samples - quantile, 0))/(1-beta)
    

def sample_superquantile_by_minimization(samples, beta, epsilon=1e-2):
    """
    Evaluate superquantile from samples by minimization 
    """ 
    quantile = np.percentile(samples, beta * 100) 
    smoothPlus = SmoothPlusApproximationQuartic(epsilon=epsilon)
    cvar_obj = lambda t : t + np.mean(smoothPlus(samples - t))/(1-beta)
    minimum = scipy.optimize.fmin(cvar_obj, quantile)
    quantile = minimum[0]
    superquantile = cvar_obj(quantile)
    return quantile, superquantile


def superquantileRiskMeasureSAASettings(data = {}):
    data['sample_size'] = [100,'Number of Monte Carlo samples']
    data['beta'] = [0.95, 'Quantile value for superquantile']
    data['epsilon'] = [0.01, 'Sharpness of smooth plus approximation']
    data['seed'] = [1, 'rng seed for sampling']
    data['smoothplus_type'] = ['quartic', 'approximation type for smooth plus function']
    return ParameterList(data)


class SuperquantileRiskMeasureSAA(RiskMeasure):
    """
    Risk measure for the sample average approximation of the superquantile risk measure (CVaR) 
    with sample parallelism using MPI 
    """

    def __init__(self, control_model, prior, settings = superquantileRiskMeasureSAASettings(), comm_sampler=mpi4py.MPI.COMM_WORLD):
        """
        Constructor:

        :param control_model: control model containing the :code:`soupy.PDEVariationalControlProblem`
            and :code:`soupy.ControlQoI`
        :type control_model: :py:class:`soupy.ControlModel`
        :param prior: The prior distribution for the random parameter
        :type prior: :py:class:`hippylib.Prior`
        :param settings: additional settings given in :code:`superquantileRiskMeasureSAASettings()`
        :type settings: :py:class:`hippylib.ParameterList`
        :param comm_sampler: MPI communicator for sample parallelism 
        :type comm_sampler: :py:class:`mpi4py.MPI.Comm`
        """
        self.model = control_model
        self.prior = prior
        self.settings = settings
        self.sample_size = self.settings['sample_size']
        self.beta = settings['beta']
        if self.settings["smoothplus_type"] == "softplus":
            self.smoothplus = SmoothPlusApproximationSoftplus(self.settings["epsilon"])
        elif self.settings["smoothplus_type"] == "quartic":
            self.smoothplus = SmoothPlusApproximationQuartic(self.settings["epsilon"])
        else:
            # Default case 
            self.smoothplus = SmoothPlusApproximationQuartic(self.settings["epsilon"])

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
        self.s_bar = 0 
        self.sprime_bar = 0 
        self.sprime_g_bar = self.model.generate_vector(CONTROL)

        # For sampling
        self.noise = dl.Vector(self.model.problem.Vh[STATE].mesh().mpi_comm()) # use the mesh mpi comm 
        self.prior.init_vector(self.noise, "noise")
        rng = Random(seed=self.settings['seed'])

        # Generate samples for m 
        self.x_mc = [] 
        self.z = self.model.generate_vector(CONTROL)
        self.zt = AugmentedVector(self.z, copy_vector=False)

        self.g_mc = []

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

        self.Hi_zhat = self.model.generate_vector(CONTROL)
        self.control_model_hessian = None 


    def generate_vector(self, component = "ALL"):
        """
        If :code:`component` is :code:`STATE`, :code:`PARAMETER`, :code:`ADJOINT`, \
            return a vector corresponding to that function space. \
            If :code:`component == CONTROL`, return an :py:class:`soupy.AugmentedVector` \
            that augments the control variable :code:`z` with a scalar that can be used \
            for optimization 
            
            If :code:`component == "ALL"`, \
            Generate the list of vectors :code:`x = [u,m,p,z]`. \
            Note that in this case, the :code:`CONTROL` variable will not be augmented \
            with the scalar, and can be used directly for methods like :code:`solveFwd`.
        """
        if component == CONTROL:
            return AugmentedVector(self.model.generate_vector(CONTROL), copy_vector=False)
        else:
            return self.model.generate_vector(component)

    def computeComponents(self, zt, order=0, **kwargs):
        """
        Computes the components for the evaluation of the risk measure

        :param zt: the control variable with the scalar :code:`t` appended
        :type zt: :py:class:`soupy.AugmentedVector`
        :param order: Order of the derivatives needed.
            :code:`0` for cost, :code:`1` for gradient, :code:`2` for Hessian
        :type order: int 
        :param kwargs: dummy keyword arguments for compatibility 
        """
        z = zt.get_vector()
        t = zt.get_scalar()
        new_forward_solve = False 

        self.s_bar = 0 
        self.sprime_bar = 0 
        self.sprime_g_bar.zero()

        # Check if a new control variable is used 
        self.diff_helper.zero()
        self.diff_helper.axpy(1.0, self.z)
        self.diff_helper.axpy(-1.0, z)
        diff_norm = np.sqrt(self.diff_helper.inner(self.diff_helper))
        
        # Store zt in class 
        self.zt.zero()
        self.zt.axpy(1.0, zt)

        # Check if new forward solve is needed 
        if diff_norm > dl.DOLFIN_EPS or not self.has_forward_solve:
            # Update control variable (changes all samples)
            # Ask that new forward and adjoint solves are computed 
            new_forward_solve = True
            logging.info("Using new forward solve")

        # Check if a new adjoint solve is needed 
        if new_forward_solve:
            # Reset the averages for new solve 
            # Make sure previous adjoint solve is no longer valid 
            self.has_adjoint_solve = False 
            new_adjoint_solve = True 

        elif self.has_adjoint_solve:
            # If it's not a new forward solve, check if we already have an adjoint 
            new_adjoint_solve = False 
            # self.sprime_bar = 0 
            # self.sprime_g_bar.zero()
        else:
            # If we don't already have an adjoint, compute a new one 
            new_adjoint_solve = True 
            # self.sprime_bar = 0 
            # self.sprime_g_bar.zero()
            # In this case, self.has_adjoint_solve is already False

        # Now actually compute the solves
        for i in range(self.sample_size_proc):
            x = self.x_mc[i]

            if new_forward_solve:
                # Solve state           
                self.model.solveFwd(x[STATE], x)
                
            # Compute the averages
            qi = self.model.cost(x) 
            self.q_samples[i] = qi 
            self.s_bar += self.smoothplus(qi - t)/self.sample_size

            if order >= 1:
                if new_adjoint_solve:
                    self.model.solveAdj(x[ADJOINT], x)
                self.model.evalGradientControl(x, self.g_mc[i])
                self.sprime_bar += self.smoothplus.grad(qi - t)/self.sample_size
                self.sprime_g_bar.axpy(self.smoothplus.grad(qi - t)/self.sample_size, self.g_mc[i])
        
        self.s_bar = self.collective.allReduce(self.s_bar, "SUM")
        
        # if order >= 1 and new_adjoint_solve:
        if order >= 1:
            self.collective.allReduce(self.sprime_g_bar, "SUM")
            self.sprime_bar = self.collective.allReduce(self.sprime_bar, "SUM")

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
        t = self.zt.get_scalar()
        return  t + 1/(1-self.beta) * self.s_bar

    def grad(self, gt):
        """
        Evaluates the gradient of the risk measure once components have been computed

        :param g: (Dual of) the gradient of the risk measure to store result in
        :type g: :py:class:`dolfin.Vector`

        .. note:: Assumes :code:`self.computeComponents` has been called with :code:`order>=1`
        """
        # print("(proc %d) q_bar = %g" %(self.comm_sampler.Get_rank(), self.q_bar))
        dzJ_np = self.sprime_g_bar.get_local()/(1-self.beta)
        dtJ_np = 1 - self.sprime_bar/(1-self.beta)
        dz_np = np.append(dzJ_np, dtJ_np)
        gt.set_local(dz_np) 


    def hessian(self, zt_hat, Hzt_hat):
        """
        Apply the hessian of the risk measure once components have been computed \
            in the direction :code:`zhat`
        
        :param zt_hat: Direction for application of Hessian action of the risk measure
        :type zt_hat: :py:class:`soupy.AugmentedVector`
        :param Hzt_hat: (Dual of) Result of the Hessian action of the risk measure 
            to store the result in
        :type Hzt_hat: :py:class:`soupy.AugmentedVector`

        .. note:: Assumes :code:`self.computeComponents` has been called with :code:`order >= 1`
        """
        if self.control_model_hessian is None:
            self.control_model_hessian = ControlModelHessian(self.model)

        Hzt_hat.zero()

        Hz_hat = Hzt_hat.get_vector()
        Ht_hat = 0.0

        z_hat = zt_hat.get_vector()
        t_hat = zt_hat.get_scalar()

        t = self.zt.get_scalar()



        for i in range(self.sample_size_proc):
            xi = self.x_mc[i] 
            gi = self.g_mc[i]
            qi = self.q_samples[i] 
            
            # Derivatives of the smoothed max approximation
            sprime_i = self.smoothplus.grad(qi - t)
            sprimeprime_i = self.smoothplus.hessian(qi - t)

            self.model.setLinearizationPoint(xi)
            self.control_model_hessian.mult(z_hat, self.Hi_zhat)
            gi_inner_zhat = gi.inner(z_hat)
            
            # Output of Hessian in z
            zz_hessian_scale_factor = sprime_i/(1-self.beta)/self.sample_size
            zz_gradient_scale_factor = sprimeprime_i * gi_inner_zhat/(1-self.beta)/self.sample_size 
            zt_gradient_scale_factor = -sprimeprime_i * t_hat/(1-self.beta)/self.sample_size

            Hz_hat.axpy(zz_hessian_scale_factor, self.Hi_zhat)
            Hz_hat.axpy(zz_gradient_scale_factor + zt_gradient_scale_factor, gi)

            # Output of Hessian in t 
            Ht_hat += sprimeprime_i/(1-self.beta)/self.sample_size * (t_hat - gi_inner_zhat)
            
        self.collective.allReduce(Hz_hat, "SUM")
        Ht_hat_all = self.collective.allReduce(Ht_hat, "SUM")
        Hzt_hat.set_scalar(Ht_hat_all)


    def gather_samples(self):
        """
        Gather the QoI samples from all processes

        :return: An array of the sample QoI values
        :return type: :py:class:`numpy.ndarray`
        """
        q_all = np.zeros(self.sample_size) 
        self.comm_sampler.Allgatherv(self.q_samples, [q_all, self.sample_size_allprocs])
        return q_all 


    def superquantile(self):
        """ 
        Evaluate the superquantile using the computed samples 

        :return: Value of the superquantile by sampling
    
        .. note:: Assumes :code:`computeComponents` has been called with :code:`order>=0`
        """
        q_all = self.gather_samples()
        value = sample_superquantile(q_all, self.beta)
        return value 


