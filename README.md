![Build Status](https://github.com/hippylib/soupy/actions/workflows/ci.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/hippylibsoupy/badge/?version=latest)](https://hippylibsoupy.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/598860325.svg)](https://zenodo.org/badge/latestdoi/598860325)

# Stochastic Optimization under high-dimensional Uncertainty in Python

**S**tochastic **O**ptimization under high-dimensional **U**ncertainty in **Py**thon&mdash;**SOUPy**, implements scalable algorithms for the optimization of large-scale complex systems governed by partial differential equations (PDEs) under high-dimensional uncertainty. The library features various risk measures (such as mean, variance, and superquantile/condition-value-at-risk), probability/chance constraints, and optimization/state constraints. SOUPy enables efficient PDE-constrained optimization under uncertainty through parallel computation of the risk measures and their derivatives (gradients and Hessians). The library also provides built-in parallel implementations of optimization algorithms (e.g. BFGS, Inexact Newton CG), as well as an interface to the `scipy.optimize` module in [SciPy](https://scipy.org/). Besides the benchmark/tutorial examples in the examples folder, SOUPy has been used to solve large-scale and high-dimensional stochastic optimization problems including [optimal control of turbulence flow](https://www.sciencedirect.com/science/article/pii/S0021999119301056), optimal design of [acoustic metamaterials](https://www.sciencedirect.com/science/article/pii/S0021999121000061) and [self-assembly nanomaterials](https://www.sciencedirect.com/science/article/pii/S0021999123001961),  and [optimal management of groundwater extraction](https://epubs.siam.org/doi/abs/10.1137/20M1381381), etc.

SOUPy is built on the open-source [hIPPYlib library](https://hippylib.github.io/), which provides adjoint-based methods for deterministic and Bayesian inverse problems governed by PDEs, and makes use of [FEniCS](https://fenicsproject.org/) for the high-level formulation, discretization, and solution of PDEs.

SOUPy is in active development to incorporate advanced approximation algorithms and capabilities, including:

- Taylor expansion-based approximations for risk measure evaluation
- High-dimensional quadrature methods such as sparse grids and quasi Monte Carlo
- Decomposition of high-dimensional uncertain parameter spaces by mixture models
- Multi-fidelity methods and control variates
- Interfaces with Bayesian inverse problems

See the [SOUPy documentation](https://hippylibsoupy.readthedocs.io/en/latest/) for more information.

Please consider to cite the following papers for using SOUPy

```
@article{chen2019taylor,
  title={Taylor approximation and variance reduction for PDE-constrained optimal control under uncertainty},
  author={Chen, Peng and Villa, Umberto and Ghattas, Omar},
  journal={Journal of Computational Physics},
  volume={385},
  pages={163--186},
  year={2019},
  publisher={Elsevier}
}
```


```
@article{luo2023efficient,
  title={Efficient PDE-Constrained optimization under high-dimensional uncertainty using derivative-informed neural operators},
  author={Luo, Dingcheng and O'Leary-Roseberry, Thomas and Chen, Peng and Ghattas, Omar},
  journal={arXiv preprint arXiv:2305.20053},
  year={2023}
}
```


### Acknowledgements

This project is partially supported by NSF grants [#2012453](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2012453&HistoricalAwards=false) and [#2245674](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2245674).
