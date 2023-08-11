---
title: 'SOUPy: Stochastic PDE-constrained optimization under high-dimensional uncertainty in Python'
tags:
  - Python
  - Uncertainty Quantification
  - PDE-constrained optimization
  - Adjoint-based methods 
  - Optimal design 
authors:
  - name: Peng Chen 
    affiliation: 3
  - name: Dingcheng Luo 
    affiliation: 1
  - name: Thomas O'Leary-Roseberry 
    affiliation: 1
  - name: Umberto Villa 
    affiliation: 1
  - name: Omar Ghattas
    affiliation: "1 2"
affiliations:
 - name: Oden Institute for Computational Engineering and Sciences, University of Texas at Austin, USA 
   index: 1
 - name: Walker Department of Mechanical Engineering, University of Texas at Austin, USA 
   index: 2
 - name: School of Computational Science and Engineering, Georgia Institute of Technology, USA
   index: 3
date: 11 August 2023
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Computational models governed by partial differential equations (PDEs) 
are frequently used by engineers to optimize the performance of various physical systems 
through decisions relating to their configuration (optimal design) and operation (optimal control). 
However, the ability to make optimal choices is often hindered by uncertainty, 
such as uncertainty in model parameters (e.g. material properties) and operating conditions.
The need for making robust and risk-informed decisions thus gives rise to problems of optimization under uncertainty (OUU). 

SOUPy is a python library for solving PDE-constrained optimization problems with uncertain parameters,
in which the optimization objective is defined by a risk measure over a given quantity of interest (QoI).
Specific attention is given to the case where the uncertain parameter is formally infinite dimensional (e.g. Gaussian random fields).
The software allows users to supply the underlying PDE model, quantity of interest, and penalization terms, 
while providing implementations for commonly used risk measures, including expectation, variance, and superquantile/conditional-value-at-risk (CVaR).
as well as derivative-based optimizers. 
SOUPy leverages FEniCS for the formulation, discretization, and solution of PDEs, 
and the framework of hIPPYlib for sampling and adjoint-based derivative computation, 
while also providing interfaces to existing optimization algorithms in SciPy.


# Statement of need 

Problems of PDE-constrained optimization under uncertainty arise due to the need to make risk-informed decisions in the presence of uncertainty.
In deterministic PDE-constrained optimization, the goal is typically to minimize a quantity of interest (QoI) that is a function of the system's state and quantifies its performance, where the optimization and state variables are related through the underlying PDE model. 
Compared to this deterministic counterpart, PDE-constrained OUU represents an added layer of complexity, 
as the QoI becomes a random variable due to its dependence on the uncertain model parameters.
In OUU, the optimization problem instead aims to minimize a risk measure, which is a statistical quantity summarizing the QoI's distribution. 
A canonical example of such a risk measure is the expected value of the QoI, 
though other measures that account for the tail behavior of the distribution such as 
the variance, or superquantile/CVaR are common choices.
Computation of risk measures typically requires sampling or other forms of quadrature over the distribution of the uncertain parameter.
This results in a complex optimization problem in which each evaluation of the optimization objective requires numerous solutions of the underlying PDE.

SOUPy provides a platform to formulate and solve such PDE-constrained OUU problems using efficient derivative-based optimization methods. 
SOUPy makes use of FEniCS, an open source finite-element library, to create and solve the underlying PDEs. 
The unified form language used by FEniCS allows users to conveniently define the PDE in its weak form, 
as well as the form of the QoI and any additional penalization terms on the optimization variable. 
SOUPy is also integrated with hIPPYlib, an open source library for large-scale inverse problems, 
adopting its framework for adjoint-based computation of derivatives and efficient sampling of random fields.
At its core, SOUPy implements sample-based evaluation of risk measures and their derivatives, where parallel-in-sample computation is supported through MPI. 
The risk measures are used to define cost functions, 
which can be minimized using custom implementations of large-scale optimization algorithms such as BFGS and Inexact Newton-CG, 
or through algorithms available in SciPy using the provided interface. 

Additionally, SOUPy aims to facilitate the development and testing of novel algorithms for PDE-constrained OUU. 
For example, SOUPy has been used in the development of methods for the optimization of turbulent flows, metamaterial design, and groundwater extraction.
It has also been used to obtain baselines for the development of machine learning approaches for PDE-constrained OUU.

# Acknowledgements
This project is partially supported by NSF grants #2012453 and #2245674.


<!-- # Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project. -->

# References