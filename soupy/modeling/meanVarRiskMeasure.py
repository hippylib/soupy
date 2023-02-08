import numpy as np 
import dolfin as dl 

import sys, os
sys.path.append( os.environ.get('HIPPYLIB_PATH'))
from hippylib import ParameterList, parRandom

from .riskMeasure import RiskMeasure

from .variables import STATE, PARAMETER, ADJOINT, CONTROL


def meanVarRiskMeasureSettings(data = {}):
	# This should be a Parameter
	# data['nsamples'] = [100,'Number of Monte Carlo samples']
	data['beta'] = [0,'Weighting factor for variance']

	return ParameterList(data)

class MeanVarRiskMeasure(RiskMeasure):
	"""
	Class for memory efficient evaluation of the Mean + Variance risk measure 
	E[X] + beta Var[X]. 
	"""

	def __init__(self, control_model, prior, settings = meanVarRiskMeasureSettings()):
		"""
		Parameters
			- :code: `control_model` control model of problem 
			- :code: `prior` prior for uncertain parameter
			- :code: `settings` additional settings
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
		return self.model.generate_vector(component)

	def computeComponents(self, z, order=0, sample_size=100, rng=None):
		"""
		Computes the components for the stochastic approximation of the cost
		Parameters:
			- :code: `z` the control variable 
			- :code: `order` the order of derivatives needed. 
					0 for cost. 1 for grad. 2 for Hessian
			- :code: `sample_size` number of samples for sample average
			- :code: `rng` rng for the sampling (optional)
		"""
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
		Evaluates the cost given by the risk measure
		Assumes :code: `computeComponents` has been called
		"""
		return self.q_bar + self.beta * np.std(self.q_samples)**2

	def costGrad(self, g):
		"""
		Evaluates the gradient by the risk measure
		Assumes :code: `computeComponents` has been called with :code: `order>=1`
		Parameters
			- :code: `g` output vector for the gradient
		"""
		g.zero()
		g.axpy(1.0, self.g_bar)
		g.axpy(2*self.beta, self.qg_bar)
		g.axpy(-2*self.beta*self.q_bar, self.g_bar)

	def costHessian(self, zhat, Hzhat):
		self.model.setPointForHessianEvaluations(self.x)
		self.model.applyCz(zhat, self.rhs_fwd)
		self.model.solveFwdIncremental(self.uhat, self.rhs_fwd)
		self.model.applyWuu(self.uhat, self.rhs_adj)
		self.model.applyWuz(zhat, self.rhs_adj2)
		self.rhs_adj.axpy(-1., self.rhs_adj2)
		self.model.solveAdjIncremental(self.phat, self.rhs_adj)
		self.model.applyWzz(zhat, Hzhat)

		self.model.applyCzt(self.phat, self.zhelp)
		Hzhat.axpy(1., self.zhelp)
		self.model.applyWzu(self.uhat, self.zhelp)
		Hzhat.axpy(-1., self.zhelp)