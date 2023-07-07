# soupy
**S**tochastic **O**ptimization under high-dimensional **U**ncertainty in **Py**thon.

SOUPy implements scalable algorithms to solve problems of PDE-constrained optimization under uncertainty, with the computational complexity measured in terms of PDE solves independent of the uncertain parameter dimension and optimization variable dimension.

SOUPy is built on the open-source hIPPYlib library, which provides state-of-the-art scalable adjoint-based methods for deterministic and Bayesian inverse problems governed by PDEs, which in turn makes use of the FEniCS library for high-level formulation, discretization, and scalable solution of PDEs.

SOUPy implements algorithm for optimization of probability/risk measures for risk-averse optimization, such as mean, variance, and superquantile/condition-value-at-risk. 
Sample based estimators can be used through SOUPy, which leverages MPI for rapid parallel sampling.
Numerical optimization algorithms can be called from SciPy or using our built-in optimization algorithms. The evaluation of 

SOUPy has been developed and in active development to incorporate advanced approximation algorithms and capabilities, including

- PDE-constrained operator/tensor/matrix products,
- symbolic differentiation (of appropriate Lagrangians) for the derivation of high order mixed derivatives (via the FEniCS interface),
- randomized algorithms for matrix and high order tensor decomposition,
- decomposition of uncertain parameter spaces by mixture models,
- Taylor expansion-based high-dimensional control variates, and
- product convolution approximations,
- common interfaces for random fields, PDE models, probability/risk measures, and control/design/inversion constraints.
