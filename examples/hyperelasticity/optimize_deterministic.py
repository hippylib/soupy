import pickle
import sys, os
import argparse 

import numpy as np
import matplotlib.pyplot as plt 
import dolfin as dl
import scipy.optimize 


sys.path.append( os.environ.get('HIPPYLIB_PATH'))
sys.path.append( os.environ.get('SOUPY_PATH'))

import hippylib as hp 
import soupy

from setupHyperelasticityProblem import hyperelasticity_problem_settings, setup_hyperelasticity_problem


# Optimization options for the form compiler
dl.parameters["form_compiler"]["cpp_optimize"] = True
dl.parameters["form_compiler"]["quadrature_degree"] = 4
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

dl.set_log_active(False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--maxiter', type=int, default=100, help="Maximum number of SD iterations")
    parser.add_argument('-q', '--qoi_type', type=str, default="stiffness", choices=["all", "stiffness", "point"])
    parser.add_argument('-p', '--penalization', type=float, default=1e-1, help="Scaling of penalization")
    args = parser.parse_args()

    # Create mesh and setup model components 
    settings = hyperelasticity_problem_settings()
    settings["qoi_type"] = args.qoi_type
    mesh, Vh, hyperelasticity_varf, control_model, prior = setup_hyperelasticity_problem(settings)
    
    l2_penalty = soupy.L2Penalization(Vh, args.penalization)
    pde_cost = soupy.DeterministicControlCostFunctional(control_model, prior, l2_penalty)

    scipy_cost = soupy.ScipyCostWrapper(pde_cost, verbose=True)
    box_bounds = scipy.optimize.Bounds(lb=0.0, ub=1.0)

    x = control_model.generate_vector()
    control_model.solveFwd(x[soupy.STATE], x)
    disp_fun_init = hp.vector2Function(x[soupy.STATE], Vh[soupy.STATE])

    z0_np = x[soupy.CONTROL].get_local() + 0.5
    options = {'gtol' : 1e-12, 'disp' : True, 'maxiter' : args.maxiter}
    results = scipy.optimize.minimize(scipy_cost.function(), z0_np, 
            method="L-BFGS-B", jac=scipy_cost.jac(), 
            bounds=box_bounds, options=options)
    z_opt = results['x']     

    x[soupy.CONTROL].set_local(z_opt)
    control_model.solveFwd(x[soupy.STATE], x)

    plt.figure()
    ax = dl.plot(disp_fun_init, mode="displacement")
    plt.colorbar(ax)
    plt.title("Displacement with soft material")

    plt.figure()
    ax = dl.plot(hp.vector2Function(x[soupy.CONTROL], Vh[soupy.CONTROL]))
    plt.colorbar(ax)
    plt.title("Optimal design")

    plt.figure()
    ax = dl.plot(hp.vector2Function(x[soupy.STATE], Vh[soupy.STATE]), mode="displacement")
    plt.colorbar(ax)
    plt.title("Optimal displacement")
    plt.show()

