![Build Status](https://github.com/hippylib/soupy/actions/workflows/ci.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/hippylibsoupy/badge/?version=latest)](https://hippylibsoupy.readthedocs.io/en/latest/?badge=latest)

# Stochastic Optimization under high-dimensional Uncertainty in Python

**S**tochastic **O**ptimization under high-dimensional **U**ncertainty in **Py**thon&mdash;**SOUPy**, 
is implements scalable algorithms for the optimization of risk measures such as mean, variance, and superquantile/condition-value-at-risk, subject to PDE constraints. 
SOUPy enables efficient PDE-constrained optimization under uncertainty through parallel computation of risk measures and their derivatives (gradients and Hessians).
The library provides built-in implementations of large-scale optimization algorithms (e.g. BFGS, Inexact Newton CG), as well as an interface to the `scipy.optimize` module in [SciPy](https://scipy.org/).

SOUPy is built on the open-source [hIPPYlib library](https://hippylib.github.io/), which provides adjoint-based methods for deterministic and Bayesian inverse problems governed by PDEs, and makes use of [FEniCS](https://fenicsproject.org/) for the high-level formulation, discretization, and solution of PDEs.

SOUPy is in active development to incorporate advanced approximation algorithms and capabilities, including:
- Taylor expansion-based approximations for risk measure evaluation
- High-dimensional quadrature methods such as sparse grids and quasi Monte Carlo
- Decomposition of uncertain parameter spaces by mixture models
- Multi-fidelity methods and control variates 
- Interfaces with Bayesian inverse problems 
