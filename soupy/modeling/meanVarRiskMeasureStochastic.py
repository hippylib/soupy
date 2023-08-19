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

import numpy as np 
import dolfin as dl 
import logging 

import sys, os
from hippylib import ParameterList, parRandom

from .riskMeasure import RiskMeasure

from .variables import STATE, PARAMETER, ADJOINT, CONTROL


def meanVarRiskMeasureStochasticSettings(data = {}):
    data['beta'] = [0,'Weighting factor for variance']

    return ParameterList(data)

class MeanVarRiskMeasureStochastic(RiskMeasure):
    """
    Class for memory efficient evaluation of the Mean + Variance risk measure 

        .. math:: \\rho[Q](z) = \mathbb{E}_m[Q(m,z)] + \\beta \mathbb{V}_m[Q(m,z)]

    that allows for stochastic approximations and SGD 
    """

    def __init__(self, control_model, prior, settings = meanVarRiskMeasureStochasticSettings()):
        """
        Constructor:

        :param control_model: control model containing the :code:`soupy.PDEVariationalControlProblem`
            and :code:`soupy.ControlQoI`
        :type control_model: :py:class:`soupy.ControlModel`
        :param prior: The prior distribution for the random parameter
        :type prior: :py:class:`hippylib.Prior`
        :param settings: additional settings
        :type settings: :py:class:`hippylib.ParameterList`
        """
        self.model = control_model
        self.prior = prior
        self.settings = settings
        self.settings.showMe()
        # self.n_samples = self.settings['nsamples']
        self.beta = settings['beta']


        # Aggregate components for computing cost, grad, hess
        self.x = self.model.generate_vector()
        self.g = self.model.generate_vector(CONTROL)
        self.q_samples = np.zeros(1)

        self.uhat = self.model.generate_vector(STATE)
        self.phat = self.model.generate_vector(STATE)
        self.zhelp = self.model.generate_vector(CONTROL)
        self.rhs_fwd = self.model.generate_vector(STATE)
        self.rhs_adj = self.model.generate_vector(STATE)
        self.rhs_adj2 = self.model.generate_vector(STATE)

        self.q_bar = 0
        self.g_bar = self.model.generate_vector(CONTROL)
        self.qg_bar = self.model.generate_vector(CONTROL)

        # For sampling
        self.noise = dl.Vector()
        self.prior.init_vector(self.noise, "noise")

    def generate_vector(self, component = "ALL"):
        """
        If :code:`components` is :code:`STATE`, :code:`PARAMETER`, :code:`ADJOINT`, \
            or :code:`CONTROL`, return a vector corresponding to that function space. \
            If :code:`components` is :code:`"ALL"`, \
            Generate the list of vectors :code:`x = [u,m,p,z]`
        """
        return self.model.generate_vector(component)

    def computeComponents(self, z, order=0, sample_size=100, rng=None):
        """
        Computes the components for the evaluation of the risk measure

        :param z: the control variable
        :type z: :py:class:`dolfin.Vector`
        :param order: Order of the derivatives needed.
            :code:`0` for cost, :code:`1` for gradient, :code:`2` for Hessian
        :type order: int 
        :param rng: The random number generator used for sampling. If :code:`None` the default
            uses :py:class:`hippylib.parRandom`
        :type rng: :py:class:`hippylib.Random`
        """

        # Check if a new control variable is used 
        self.q_samples = np.zeros(sample_size)
        if order >= 1:
            self.g_bar.zero()
            self.qg_bar.zero()

        for i in range(sample_size):
            # Assign control
            self.x[CONTROL].zero()
            self.x[CONTROL].axpy(1.0, z)

            # Sample parameter
            if rng is None:
                parRandom.normal(1.0, self.noise)
            else:
                rng.normal(1.0, self.noise)
            self.prior.sample(self.noise, self.x[PARAMETER])

            # Solve state 			
            self.model.solveFwd(self.x[STATE], self.x)
            self.q_samples[i] = self.model.cost(self.x)

            if order >= 1:
                self.model.solveAdj(self.x[ADJOINT], self.x)
                self.model.evalGradientControl(self.x, self.g)
                self.g_bar.axpy(1/sample_size, self.g)
                self.qg_bar.axpy(self.q_samples[i]/sample_size, self.g)

            # Still need Hessian code	
            # if i % 10 == 0:
            # 	print(i)

        self.q_bar = np.mean(self.q_samples)
    
    
    def cost(self):
        """
        Evaluates the value of the risk measure once components have been computed

        :return: Value of the cost functional

        .. note:: Assumes :code:`computeComponents` has been called with :code:`order>=0`
        """
        return self.q_bar + self.beta * np.std(self.q_samples)**2

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
        logging.warning("Hessian not implemented for MeanVarRiskMeasureStochastic")
