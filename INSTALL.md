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

**For x86 users:** 
```
conda create -n soupy -c conda-forge python=3.11 fenics==2019.1.0 petsc4py==3.19 matplotlib scipy jupyter
conda activate soupy
```
**For ARM users:** `FEniCS` is only available on x86 systems. 
When running on an ARM based mac with the ARM version of conda installed, add `CONDA_SUBDIR=osx-64` before the conda call, 
```
CONDA_SUBDIR=osx-64 conda create -n soupy -c conda-forge python=3.11 fenics==2019.1.0 petsc4py==3.19 matplotlib scipy jupyter
```
Then configure the environment to be an `osx-64` environment
```
conda activate soupy
conda config --env --set subdir osx-64
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
Examples in the `applications` directory can now be run. 
```
cd soupy/examples/poisson
```
Then run in serial
```
python driver_poisson_mean.py 
```
or with MPI (e.g. using 4 processes)
```
mpirun -n 4 python driver_poisson_mean.py 
```

We refer to the full `FEniCS` [installation instructions](https://hippylib.readthedocs.io/en/3.0.0/installation.html) from `hIPPYlib` for more detail. 

### Installation for developers 
1. Create an environment with `FEniCS` with appropriate dependencies (add `CONDA_SUBDIR=osx-64` if using ARM conda, see above)
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
Examples in the `examples` directory can now be run after deactivating and activating the environment again. 

## Installation with Docker

SOUPy (and hIPPYlib) can be used with a Docker image built with FEniCS/dolfin. 
The [FEniCS documentation](https://fenics.readthedocs.io/projects/containers/en/latest/) provides installation instructions for FEniCS's official images (note these are built on older versions of python). Alternatively, a customized docker image built on the FEniCS 2019.1.0 image with hIPPYlib pre-installed is available [here](https://hub.docker.com/r/hippylib/fenics)

More recent builds of the docker images with FEniCS that are compatible with the [Frontera(https://frontera-portal.tacc.utexas.edu/) HPC system] at the Texas Advanced Computing Center can be found under the `hippylib/tacc-containers` [repository](https://github.com/hippylib/tacc-containers). The repository also provides instructions for usage on Frontera using Apptainer. Here, we will summarize the procedure for using the image on local machines with Docker.

1. Pull one the docker images 
``` 
docker pull uvilla/fenics-2019.1-tacc-mvapich2.3-ib:latest 
```

2. Under the desired working directory, clone hIPPYlib and SOUPy
```
git clone https://github.com/hippylib/hippylib.git
git clone https://github.com/hippylib/soupy.git
```

3. The images are built for the MPI implementations on TACC systems. When running locally, the environment variables `MV2_SMP_USE_CMA=0` and `MV2_ENABLE_AFFINITY=0` need to be set.
```
docker run -e MV2_SMP_USE_CMA=0 -e MV2_ENABLE_AFFINITY=0 -ti -v $(pwd):/home1/ uvilla/fenics-2019.1-tacc-mvapich2.3-ib:latest /bin/bash 
```
This will run the container and bind the current working directory to `/home1`.
The running container should now have `/home1/hippylib` and `/home1/soupy`. 

4. With the container running, first set the path to hippylib `export HIPPYLIB_PATH=/home1/hippylib`. The examples in the SOUPy repository can then be executed as 
```
cd /home1/soupy/examples/poisson
python3 driver_poisson_mean.py
```
**Note**: benign MPI warning messages may show up when running on ARM machines. 


## Building the SOUPy documentation using Sphinx

Documentation for `SOUPy` can be built using `sphinx`, along with extensions
`myst_nb` and `sphinx_rtd_theme`. These can be installed via `pip`.

To build simply run `make html` from `doc` folder.