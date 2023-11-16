# Installation 

## Dependencies
`SOUPy` depends on [FEniCS](http://fenicsproject.org/) version 2019.1. and [hIPPYlib](https://hippylib.github.io/) version 3.0.0 or above.

`FEniCS` needs to be built with the following dependencies enabled:

 - `numpy`, `scipy`, `matplotlib`, `mpi4py`
 - `PETSc` and `petsc4py` (version 3.10.0 or above)
 - `SLEPc` and `slepc4py` (version 3.10.0 or above)
 - PETSc dependencies: `parmetis`, `scotch`, `suitesparse`, `superlu_dist`, `ml`, `hypre`
 - (optional): `gmsh`, `mshr`, `jupyter`

## Recommended installation using Anaconda 
### Installation with pip 
1. Create an environment with `FEniCS` and appropriate dependencies and activate environment
```
conda create -n soupy -c conda-forge python=3.11 fenics==2019.1.0 petsc4py==3.19 matplotlib scipy jupyter
conda activate soupy
```

2. Install `hIPPYlib` via pip
```
pip install hippylib
```

3. Install `SOUPy` via pip 
```
pip install soupy@git+https://github.com/hippylib/soupy
```

4. Clone the `SOUPy` directory to access examples 
```
git clone https://github.com/hippylib/soupy.git
```

Examples in the `applications` directory can now be run. We refer to the full `FEniCS` [installation instructions](https://hippylib.readthedocs.io/en/3.0.0/installation.html) from `hIPPYlib` for more detail. 

### Installation for developers 
1. Create an environment with `FEniCS` with appropriate dependencies
```
conda create -n soupy -c conda-forge python=3.11 fenics==2019.1.0 petsc4py==3.19 matplotlib scipy jupyter
```

2. Clone the `hIPPYlib` [repository](https://github.com/hippylib/hippylib). 
```
git clone https://github.com/hippylib/hippylib.git
```

3. Clone the `SOUPy` [repository](https://github.com/hippylib/soupy/tree/main)
```
git clone https://github.com/hippylib/soupy.git
```

4. Set the path to the `hIPPYlib` and `SOUPy` as environment variables, e.g. 
```
conda activate soupy
conda env config vars set HIPPYLIB_PATH=path/to/hippylib
conda env config vars set SOUPY_PATH=path/to/soupy
```

Examples in the `examples` directory can now be run. 

## Build the SOUPy documentation using Sphinx

Documentation for `SOUPy` can be built using `sphinx`, along with extensions
`myst_nb` and `sphinx_rtd_theme`. These can be installed via `pip`.

To build simply run `make html` from `doc` folder.
