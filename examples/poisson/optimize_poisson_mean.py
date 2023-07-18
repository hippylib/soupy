import sys 
import os 

import dolfin as dl  
import numpy as np 
import matplotlib.pyplot as plt 

sys.path.append(os.environ.get("HIPPYLIB_PATH"))
sys.path.append("../../")
import hippylib as hp 
import soupy 
dl.set_log_active(False)

N_ELEMENTS_X = 32
N_ELEMENTS_Y = 32 
IS_FWD_LINEAR = True

PRIOR_GAMMA = 1.0
PRIOR_DELTA = 20.0
PRIOR_MEAN = -2.0

VARIANCE_WEIGHT = 0.0
SAMPLE_SIZE = 4
PENALTY_WEIGHT = 1e-3

# 1. Setup function spaces 
mesh = dl.UnitSquareMesh(N_ELEMENTS_X, N_ELEMENTS_Y)
Vh_STATE = dl.FunctionSpace(mesh, "CG", 1)
Vh_PARAMETER = dl.FunctionSpace(mesh, "CG", 1)
Vh_CONTROL = dl.FunctionSpace(mesh, "CG", 1)
Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE, Vh_CONTROL] 

# 2. Define PDE Problem 
# This is a linear poisson equation where the control variable 
# is the distributed source term. Dirichlet Boundary conditions
# are on the left and bottom boundaries 

def residual(u,m,p,z):
    return dl.exp(m)*dl.inner(dl.grad(u), dl.grad(p))*dl.dx - z * p *dl.dx 

def boundary(x, on_boundary):
    return on_boundary and (dl.near(x[0], 0) or dl.near(x[1], 0))

boundary_value = dl.Expression("x[1]", degree=1)

bc = dl.DirichletBC(Vh_STATE, boundary_value, boundary)
bc0 = dl.DirichletBC(Vh_STATE, dl.Constant(0.0), boundary)
pde = soupy.PDEVariationalControlProblem(Vh, residual, [bc], [bc0], is_fwd_linear=IS_FWD_LINEAR)

# 3. Define a Gaussian prior for the random parameter
mean_vector = dl.interpolate(dl.Constant(PRIOR_MEAN), Vh_PARAMETER).vector()
prior = hp.BiLaplacianPrior(Vh_PARAMETER, PRIOR_GAMMA, PRIOR_DELTA, mean=mean_vector, robin_bc=True)

# 4. Define the QoI 
u_target = dl.Expression("x[1] + sin(k*x[0]) * sin(k*x[1])", k=1.5*np.pi, degree=2)
u_target_function = dl.interpolate(u_target, Vh_STATE)
u_target_vector = u_target_function.vector()
qoi = soupy.L2MisfitControlQoI(mesh, Vh, u_target_vector)

# 5. Define the ControlModel
control_model = soupy.ControlModel(pde, qoi)

# 6. Choose the risk measure 
risk_settings = soupy.meanVarRiskMeasureSAASettings()
risk_settings["beta"] = VARIANCE_WEIGHT
risk_settings["sample_size"] = SAMPLE_SIZE 
risk_measure = soupy.MeanVarRiskMeasureSAA(control_model, prior, risk_settings)

# 7. Define the penalization term 
penalty = soupy.L2Penalization(Vh, PENALTY_WEIGHT)

# 8. Assemble the cost functional 
cost_functional = soupy.RiskMeasureControlCostFunctional(risk_measure, penalty)

# 9. Define the optimizer 
optimizer = soupy.BFGS(cost_functional)

# 10. Provide initial guess and solve 
print("Starting optimization")
z = cost_functional.generate_vector(soupy.CONTROL)
optimizer.solve(z)
print("Done")

# 11. Post processing
risk_measure.computeComponents(z, order=0)
estimate_risk = risk_measure.cost()

# Saving output to disk
z_fun = hp.vector2Function(z, Vh[soupy.CONTROL])
with dl.HDF5File(mesh.mpi_comm(), "z_opt_no_mpi.h5", "w") as save_file:
    save_file.write(z_fun, "control")

x = cost_functional.generate_vector()
x[soupy.CONTROL].axpy(1.0, z)

# Sample from the prior 
noise = dl.Vector()
prior.init_vector(noise, "noise")
rng = hp.Random(seed=0)
rng.normal(1.0, noise)
prior.sample(noise, x[soupy.PARAMETER])

# solve the forward problem 
control_model.solveFwd(x[soupy.STATE], x)

# save some figures 
os.makedirs("figures_no_mpi", exist_ok=True)

plt.figure()
hp.nb.plot(hp.vector2Function(x[soupy.CONTROL], Vh[soupy.CONTROL]))
plt.title("Optimal control") 
plt.savefig("figures_no_mpi/optimal_control.png")
plt.close()

plt.figure()
hp.nb.plot(hp.vector2Function(x[soupy.PARAMETER], Vh[soupy.PARAMETER]))
plt.title("Sample parameter at optimal") 
plt.savefig("figures_no_mpi/sample_parameter.png")
plt.close()

plt.figure()
hp.nb.plot(hp.vector2Function(x[soupy.STATE], Vh[soupy.STATE]))
plt.title("Sample state at optimal") 
plt.savefig("figures_no_mpi/sample_state.png")
plt.close()

plt.figure()
hp.nb.plot(u_target_function)
plt.title("Target state")
plt.savefig("figures_no_mpi/target_state.png")
plt.close()


