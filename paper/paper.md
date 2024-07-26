---
title: 'SOUPy: Stochastic PDE-constrained optimization under high-dimensional uncertainty in Python'
tags:
  - Python
  - Uncertainty quantification
  - PDE-constrained optimization
  - Optimization under uncertainty
  - Adjoint method
authors:
  - name: Dingcheng Luo 
    affiliation: 1
  - name: Peng Chen 
    affiliation: 2
  - name: Thomas O'Leary-Roseberry 
    affiliation: 1
  - name: Umberto Villa 
    affiliation: 1
  - name: Omar Ghattas
    affiliation: "1, 3"
affiliations:
 - name: Oden Institute for Computational Engineering and Sciences, The University of Texas at Austin, USA 
   index: 1
 - name: School of Computational Science and Engineering, Georgia Institute of Technology, USA
   index: 2
 - name: Walker Department of Mechanical Engineering, The University of Texas at Austin, USA 
   index: 3
date: 22 September 2023
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
such as uncertainty in model parameters (e.g. material properties) and operating conditions (e.g. forces on a structure).
The need to account for these uncertainties in order to arrive at robust and risk-informed decisions thus gives rise to problems of optimization under uncertainty (OUU) [@KouriShapiro18].

SOUPy is a Python library for solving PDE-constrained optimization problems with uncertain parameters, where we use the term *parameters* broadly to refer to sources and initial/boundary conditions in addition to PDE coefficients.
The optimization problem is defined by a risk measure over a given quantity of interest (QoI), which is present as either an optimization objective or constraint (as in chance-constrained optimization).
Specific attention is given to the case where the uncertain parameters are formally infinite dimensional (e.g. Gaussian random fields).
The software allows users to supply the underlying PDE model, quantity of interest, and penalty terms, 
and provides implementations for commonly used risk measures, including expectation, variance, and superquantile/conditional value-at-risk (CVaR) [@RockafellarUryasev00], as well as derivative-based optimizers. 
SOUPy leverages FEniCS [@LoggMardalWells12] for the formulation, discretization, and solution of PDEs, 
and the framework of hIPPYlib [@VillaPetraGhattas18; @VillaPetraGhattas21] for sampling from random fields and automating adjoint-based derivative computation,
while also providing interfaces to existing optimization algorithms in SciPy.


# Statement of need 

Problems of PDE-constrained optimization under uncertainty arise due to the ubiquity of uncertainty in natural and engineered systems.
In deterministic PDE-constrained optimization, the goal is typically to optimize a quantity of interest (QoI) that is a function of the system's state and quantifies its performance, where the optimization and state variables are related through the underlying PDE model. 
Compared to this deterministic counterpart, PDE-constrained OUU involves an added layer of complexity, 
since the QoI becomes a random variable due to its dependence on the uncertain model parameters.
In OUU, the cost functional and/or constraints are instead given in terms of risk measures, which are statistical quantities summarizing the QoI's distribution. 
A canonical example of such a risk measure is the expected value of the QoI, 
though other measures that account for the tail behavior of the distribution such as 
the variance, or superquantile/CVaR are common choices.
Computation of risk measures typically requires sampling or quadrature methods to approximate the integral over the distribution of the uncertain parameter.
This results in a complex optimization problem in which each evaluation of the optimization objective requires numerous solutions of the underlying PDE.

Several open-source software packages such as dolfin-adjoint [@MituschFunkeDokken2019] and hIPPYlib 
provide the capabilities for solving PDE-constrained optimization problems with generic PDEs using derivatives computed by adjoint sensitivity methods. 
However, these packages largely focus on the deterministic setting. 
On the other hand, the Rapid Optimization Library (ROL) [@KouriRidzalWinckel17], released as a part of Trilinos [@trilinos-website], provides advanced algorithms for both deterministic and stochastic (risk measure) optimization, where support for PDE-constrained OUU is enabled by its interfaces with user-supplied state and adjoint PDE solvers. 

<!-- \autoref{fig:diagram} shows the key components of a PDE-constrained OUU problem and their corresponding classes in the SOUPy.

![Structure of a PDE-constrained OUU problem, illustrating the main components and their corresponding classes as implemented in SOUPy. \label{fig:diagram}](diagram.pdf) -->

SOUPy aims to provide a unified platform to formulate and solve PDE-constrained OUU problems using efficient derivative-based optimization methods. 
Users can supply the definitions for the PDE constraint, QoI, and additional penalty terms that account for the cost of design/control, and are then given the option to choose from a suite of risk measures to optimize.
To this end, SOUPy makes use of FEniCS, an open source finite element library, to create and solve the underlying PDEs. 
The unified form language (UFL) [@AlnaesMartinLoggEtAl14] used by FEniCS allows users to conveniently define the PDE, QoI, and penalty terms in their variational forms.
SOUPy is also integrated with hIPPYlib, an open source library for large-scale inverse problems, 
adopting its framework for automating adjoint-based derivative computation by leveraging the symbolic differentiation capabilities of UFL and algorithms for efficient sampling of random fields.
As a core functionality, SOUPy implements sample-based evaluation of risk measures as well as their gradients and Hessians, where parallel-in-sample computation is supported through MPI. 
The resulting cost functionals can then be minimized using SOUPy's implementations of large-scale optimization algorithms, such as L-BFGS [@LiuNocedal89] and Inexact Newton-CG [@EisenstatWalker96; @Steihaug83], 
or through algorithms available in SciPy [@2020SciPy-NMeth] using the provided interface. 

<!-- Since the problem formulation can be conveniently supplied through their variational forms in SOUPy, the library allows researchers to rapidly prototype formulations for PDE-constrained OUU problems, 
and automatically handles the risk measure evaluations and derivative computations. -->
<!-- by simply supplying the variational forms for the problem formulation, and leaving SOUPy to automatically handle the risk measure evaluations and derivative computations. -->

Thus, SOUPy allows researchers to rapidly prototype formulations for PDE-constrained OUU problems.
Additionally, SOUPy aims to facilitate the development and testing of novel algorithms.
For example, SOUPy has been used in the development of Taylor approximation-based methods for the risk-averse optimization of turbulent flows [@ChenVillaGhattas19], metamaterial design [@ChenHabermanGhattas21], and photonic nanojets [@AlghamdiChenKaramehmedovic22], as well as groundwater extraction [@ChenGhattas21] subject to chance constraints.
It has also been used to obtain baselines for the development of machine learning approaches for PDE-constrained OUU [@LuoOLearyRoseberryChenEtAl23].

# Acknowledgements
This project is partially supported by NSF grants 2012453 and 2245674 and DOE grants DE-SC0019303 and DE-SC0023171

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
