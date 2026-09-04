# Gaussian Mixture Taylor OUU Examples

This directory contains the numerical examples for Gaussian mixture Taylor
surrogates for risk-averse PDE-constrained optimization under uncertainty
(OUU). The examples follow the experiments in the accompanying paper,
`/work2/wenbo/mixture-taylor-ouu/mixture_taylor_ouu.tex`.

The main idea is to replace expensive sample-average approximation (SAA)
optimization with Taylor-based surrogate objectives. A single Taylor expansion
is accurate only near its expansion point, so these examples also use Gaussian
mixture Taylor approximations: the Gaussian random-field prior is decomposed
into several lower-variance Gaussian components along a dominant direction, and
a local Taylor approximation is built around each component mean.

The implemented comparisons include:

- single-point linear Taylor and quadratic Taylor surrogates;
- Gaussian mixture linear and quadratic Taylor surrogates;
- mixture directions based on KLE and Hessian-eigenvector information (HEP);
- SAA baselines with several sample sizes;
- mean-variance and CVaR risk measures.

## Directory layout

```text
Gaussian_Mixture_OUU_examples/
  semilinear_adr/
    semilinear_adr_problem.py
    driver_semilinear_adr_compare_models_static.py
    driver_semilinear_adr_compare_taylor_models.py
    driver_semilinear_adr_compare_cvar_models.py
    plot_semilinear_adr_taylor_z_gradient_check.py
    plot_semilinear_adr_cvar_z_gradient_check.py
    regenerate_paper_supplement_figures.py
    results_*/

  navier_stokes/
    navier_stokes_problem.py
    navier_stokes_pde.py
    setupNavierStokesProblem.py
    obstacle_geometry.py
    navier_stokes_driver_common.py
    navier_stokes_compare_utils.py
    driver_navier_stokes_compare_models_static.py
    driver_navier_stokes_compare_taylor_models.py
    driver_navier_stokes_compare_cvar_models.py
    regenerate_paper_figures.py
    pellet/
    results_*/
```

## Semilinear ADR example

The `semilinear_adr` example corresponds to the semilinear elliptic experiment
in the paper. The state is governed by a semilinear advection-diffusion-reaction
type PDE on a unit-square domain. The uncertain parameter is a discretized
Gaussian random field, and the control is represented by coefficients of
Gaussian well functions. The objective drives the state toward a target while
accounting for uncertainty through either mean-variance or CVaR.

Important scripts:

- `driver_semilinear_adr_compare_models_static.py`: compares approximation
  errors for mean, standard deviation, and CVaR at a fixed control as the number
  of mixture components or SAA samples changes.
- `driver_semilinear_adr_compare_taylor_models.py`: runs the mean-variance OUU
  optimization comparison for Taylor, mixture Taylor, and SAA models.
- `driver_semilinear_adr_compare_cvar_models.py`: runs the CVaR OUU
  optimization comparison.
- `plot_semilinear_adr_taylor_z_gradient_check.py`: finite-difference
  verification for mean-variance Taylor surrogate gradients.
- `plot_semilinear_adr_cvar_z_gradient_check.py`: finite-difference
  verification for CVaR Taylor surrogate gradients, including the auxiliary
  CVaR variable for quadratic CVaR models.
- `regenerate_paper_supplement_figures.py`: rebuilds selected supplementary
  figures from saved outputs.

Typical runs:

```bash
cd /work2/wenbo/soupy/examples/Gaussian_Mixture_OUU_examples/semilinear_adr

python driver_semilinear_adr_compare_models_static.py
python driver_semilinear_adr_compare_taylor_models.py
python driver_semilinear_adr_compare_cvar_models.py
```

The default paper-scale settings use a `64 x 64` mesh, `N_tr = 50` Hessian
modes for quadratic models, `N_mix = 11` mixture components for optimization
drivers, CVaR level `alpha = 0.95`, and large fixed SAA sample sets for
ground-truth evaluation.

## Navier-Stokes example

The `navier_stokes` example corresponds to the boundary-control experiment in
the paper. The PDE is a steady incompressible Navier-Stokes problem on a
rectangular channel containing an obstacle. The control is the normal velocity
profile on the obstacle boundary, the uncertain parameter is a log-normal inflow
field, and the QoI measures downstream velocity mismatch.

Important scripts:

- `driver_navier_stokes_compare_models_static.py`: compares static surrogate
  approximation errors at the zero control.
- `driver_navier_stokes_compare_taylor_models.py`: runs the mean-variance OUU
  optimization comparison.
- `driver_navier_stokes_compare_cvar_models.py`: runs the CVaR OUU optimization
  comparison.
- `regenerate_paper_figures.py`: regenerates selected paper figures from saved
  results.
- `pellet/pellet.py`: mesh-generation helper for the obstacle geometry.

Typical runs:

```bash
cd /work2/wenbo/soupy/examples/Gaussian_Mixture_OUU_examples/navier_stokes

python driver_navier_stokes_compare_models_static.py
python driver_navier_stokes_compare_taylor_models.py
python driver_navier_stokes_compare_cvar_models.py
```

The default settings use the medium obstacle mesh, viscosity `nu = 5e-3`,
Gaussian random-field parameters `gamma = 1.0` and `delta = 5.0`, mean inflow
velocity `1.0`, `N_tr = 50`, `N_mix = 11`, and CVaR level `alpha = 0.95`.

## Outputs

Each driver writes results into a `results_*` directory next to the script that
created it. The most common outputs are:

- `summary.csv` and `summary.json`: final objective values, errors, iteration
  counts, and timing summaries;
- `*_iteration_metrics.csv`: per-iteration objective and projected-gradient
  histories;
- `objective_per_iteration.png`, `residual_per_iteration.png`, and
  `objective_and_residual_per_iteration.png`: optimization history plots;
- `optimal_fields*.png` and `optimal_pde_solutions*.png`: optimized controls,
  states, and PDE solution visualizations;
- `terminal_output.txt`: captured terminal log for reproducibility.

The saved outputs are intended to support the figures and tables in the paper
and supplementary material. For a lightweight development run, reduce sample
counts and mesh resolution through the command-line options before running the
full paper-scale experiments.

## Dependencies and environment

Install the Conda environment for these examples from `gmm_ouu_environment.yml`:

```bash
cd /work2/wenbo/soupy/examples/gaussian_mixture_ouu_examples
conda env create -f gmm_ouu_environment.yml
conda activate soupy
```

If an environment named `soupy` already exists, update it with:

```bash
conda env update -n soupy -f gmm_ouu_environment.yml --prune
conda activate soupy
```

After activation, set `HIPPYLIB_PATH` to the local hIPPYlib checkout used by
this repository before running the examples.

# Parallel computing
Many runs are computationally expensive. Mixture Taylor and larger SAA cases are
parallel-aware and can be run with MPI when the local environment supports it,
for example:

```bash
mpirun -np 4 python driver_navier_stokes_compare_taylor_models.py
```

However, please note that the number of processors used cannot exceed the number of
clusters N_mix! 

