#!/usr/bin/env python
"""Validate Gaussian Mixture Taylor CVaR against SAA-CVaR.

This script compares:
1. Single Taylor Linear CVaR
2. Single Taylor Quadratic CVaR
3. Mixture Taylor Linear CVaR (new)
4. Mixture Taylor Quadratic CVaR (new)
5. SAA-based CVaR (ground truth)
"""

import numpy as np
import dolfin as dl
from mpi4py import MPI
import sys
import os

sys.path.append(os.environ.get('HIPPYLIB_PATH'))
sys.path.append('../../')

import hippylib as hp
import soupy

from soupy.approximations.taylor import (
    TaylorLinearCVaRControlCostFunctional,
    TaylorQuadraticCVaRControlCostFunctional,
    TaylorMixtureLinearCVaRControlCostFunctional,
    TaylorMixtureQuadraticCVaRControlCostFunctional,
)
from semilinearEllipticControlPDE import setup_semilinear_elliptic_pde, semilinear_elliptic_control_settings
from semilinearEllipticOUU import get_target

# Suppress output
import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
logging.getLogger('dijitso').setLevel(logging.WARNING)
dl.set_log_active(False)

print('='*70)
print('Validation: Comparing Mixture Taylor CVaR with SAA-CVaR')
print('='*70)

# Setup problem
comm_mesh = MPI.COMM_SELF
settings = semilinear_elliptic_control_settings()
settings['nx'] = 16
settings['ny'] = 16
settings['n_wells_per_side'] = 7

mesh, pde, Vh, prior = setup_semilinear_elliptic_pde(settings, comm_mesh=comm_mesh)
u_target_expr = get_target('sinusoid', 1.0, comm_mesh)
u_target = dl.interpolate(u_target_expr, Vh[hp.STATE])
qoi = soupy.L2MisfitControlQoI(Vh, u_target.vector())
control_model = soupy.ControlModel(pde, qoi)

# Test at a non-zero control where uncertainty matters
z_test = control_model.generate_vector(soupy.CONTROL)
z_np = np.ones(z_test.get_local().shape) * 2.0
z_test.set_local(z_np)
z_test.apply("")

print('\nTest at z = 2.0 (non-zero control)')
print('-'*70)

# 1. Single Taylor Linear CVaR
print('\n1. Single Taylor Linear CVaR:')
lin_settings = {'beta': 0.95, 'verbose': False}
lin_cvar = TaylorLinearCVaRControlCostFunctional(control_model, prior, None, lin_settings)
lin_cost = lin_cvar.cost(z_test, order=0)
print(f'   Mean (Q0):     {lin_cvar.lin_mean:.6e}')
print(f'   Std:           {lin_cvar.lin_std:.6e}')
print(f'   CVaR (0.95):   {lin_cvar.cvar:.6e}')

# 2. Single Taylor Quadratic CVaR
print('\n2. Single Taylor Quadratic CVaR (N_tr=20):')
quad_settings = {'beta': 0.95, 'N_tr': 20, 'N_mc': 5000, 'verbose': False}
quad_cvar = TaylorQuadraticCVaRControlCostFunctional(control_model, prior, None, quad_settings)
quad_cost = quad_cvar.cost(z_test, order=0)
print(f'   Q0:            {quad_cvar.Q_0:.6e}')
print(f'   CVaR (0.95):   {quad_cvar.cvar:.6e}')

# 3. Mixture Taylor Linear CVaR (N_mix=7)
print('\n3. Mixture Taylor Linear CVaR (N_mix=7, HEP direction):')
mix_lin_settings = {'beta': 0.95, 'N_mix': 7, 'direction': 'hep', 'verbose': False}
mix_lin_cvar = TaylorMixtureLinearCVaRControlCostFunctional(control_model, prior, None, mix_lin_settings)
mix_lin_cost = mix_lin_cvar.cost(z_test, order=0)
print(f'   VaR (0.95):    {mix_lin_cvar.var:.6e}')
print(f'   CVaR (0.95):   {mix_lin_cvar.cvar:.6e}')
print(f'   Component Q0s: {[f"{q:.4e}" for q in mix_lin_cvar.component_means]}')
print(f'   Component stds:{[f"{s:.4e}" for s in mix_lin_cvar.component_stds]}')

# 4. Mixture Taylor Quadratic CVaR (N_mix=7)
print('\n4. Mixture Taylor Quadratic CVaR (N_mix=7, N_tr=10, HEP direction):')
mix_quad_settings = {'beta': 0.95, 'N_mix': 7, 'direction': 'hep', 'N_tr': 10, 'N_mc': 1000, 'verbose': False}
mix_quad_cvar = TaylorMixtureQuadraticCVaRControlCostFunctional(control_model, prior, None, mix_quad_settings)
mix_quad_cost = mix_quad_cvar.cost(z_test, order=0)
print(f'   t_opt:         {mix_quad_cvar.t_opt:.6e}')
print(f'   CVaR (0.95):   {mix_quad_cvar.cvar:.6e}')
print(f'   Component Q0s: {[f"{q:.4e}" for q in mix_quad_cvar.component_Q0]}')

# 5. SAA-based CVaR (N=64 samples)
print('\n5. SAA-based CVaR (N=64 samples):')
saa_settings = soupy.superquantileRiskMeasureSAASettings()
saa_settings['beta'] = 0.95
saa_settings['sample_size'] = 64
saa_settings['seed'] = 1
saa_rm = soupy.SuperquantileRiskMeasureSAA(control_model, prior, settings=saa_settings)

zt = saa_rm.generate_vector(soupy.CONTROL)
zt.get_vector().zero()
zt.get_vector().axpy(1.0, z_test)
zt.set_scalar(0.0)

saa_rm.computeComponents(zt, order=0)
saa_cvar = saa_rm.superquantile()
q_samples = saa_rm.gather_samples()

print(f'   Sample mean:   {np.mean(q_samples):.6e}')
print(f'   Sample std:    {np.std(q_samples):.6e}')
print(f'   CVaR (0.95):   {saa_cvar:.6e}')

# 6. Large-sample SAA reference (N=256)
print('\n6. Large-sample SAA reference (N=256):')
saa_settings2 = soupy.superquantileRiskMeasureSAASettings()
saa_settings2['beta'] = 0.95
saa_settings2['sample_size'] = 256
saa_settings2['seed'] = 42
saa_rm2 = soupy.SuperquantileRiskMeasureSAA(control_model, prior, settings=saa_settings2)
zt2 = saa_rm2.generate_vector(soupy.CONTROL)
zt2.get_vector().zero()
zt2.get_vector().axpy(1.0, z_test)
zt2.set_scalar(0.0)
saa_rm2.computeComponents(zt2, order=0)
saa_cvar2 = saa_rm2.superquantile()
q_samples2 = saa_rm2.gather_samples()

print(f'   Sample mean:   {np.mean(q_samples2):.6e}')
print(f'   Sample std:    {np.std(q_samples2):.6e}')
print(f'   CVaR (0.95):   {saa_cvar2:.6e}')

# Summary comparison
print('\n' + '='*70)
print('Comparison Summary (vs SAA-256)')
print('='*70)
print(f'{"Method":<35} {"CVaR(0.95)":<15} {"Rel. Error":<12}')
print('-'*70)
print(f'{"Single Taylor Linear":<35} {lin_cvar.cvar:<15.4e} {abs(lin_cvar.cvar - saa_cvar2)/saa_cvar2 * 100:>8.1f}%')
print(f'{"Single Taylor Quadratic":<35} {quad_cvar.cvar:<15.4e} {abs(quad_cvar.cvar - saa_cvar2)/saa_cvar2 * 100:>8.1f}%')
print(f'{"Mixture Linear (N=7, HEP)":<35} {mix_lin_cvar.cvar:<15.4e} {abs(mix_lin_cvar.cvar - saa_cvar2)/saa_cvar2 * 100:>8.1f}%')
print(f'{"Mixture Quadratic (N=7, HEP)":<35} {mix_quad_cvar.cvar:<15.4e} {abs(mix_quad_cvar.cvar - saa_cvar2)/saa_cvar2 * 100:>8.1f}%')
print(f'{"SAA (N=64)":<35} {saa_cvar:<15.4e} {abs(saa_cvar - saa_cvar2)/saa_cvar2 * 100:>8.1f}%')
print(f'{"SAA (N=256) [Reference]":<35} {saa_cvar2:<15.4e} {"---":>8}')
print('='*70)

# Test with different mixture sizes
print('\n' + '='*70)
print('Mixture CVaR vs Number of Components')
print('='*70)

for n_mix in [3, 5, 7, 11]:
    # Linear mixture
    mix_lin_settings_n = {'beta': 0.95, 'N_mix': n_mix, 'direction': 'hep', 'verbose': False}
    mix_lin_cvar_n = TaylorMixtureLinearCVaRControlCostFunctional(control_model, prior, None, mix_lin_settings_n)
    mix_lin_cvar_n.cost(z_test, order=0)

    # Quadratic mixture
    mix_quad_settings_n = {'beta': 0.95, 'N_mix': n_mix, 'direction': 'hep', 'N_tr': 10, 'N_mc': 1000, 'verbose': False}
    mix_quad_cvar_n = TaylorMixtureQuadraticCVaRControlCostFunctional(control_model, prior, None, mix_quad_settings_n)
    mix_quad_cvar_n.cost(z_test, order=0)

    lin_err = abs(mix_lin_cvar_n.cvar - saa_cvar2)/saa_cvar2 * 100
    quad_err = abs(mix_quad_cvar_n.cvar - saa_cvar2)/saa_cvar2 * 100
    print(f'N_mix={n_mix:2d}: Linear CVaR={mix_lin_cvar_n.cvar:.4e} ({lin_err:5.1f}%), '
          f'Quad CVaR={mix_quad_cvar_n.cvar:.4e} ({quad_err:5.1f}%)')

# Compare KLE vs HEP direction
print('\n' + '='*70)
print('Direction Comparison: KLE vs HEP (N_mix=7)')
print('='*70)

for direction in ['kle', 'hep']:
    mix_lin_settings_d = {'beta': 0.95, 'N_mix': 7, 'direction': direction, 'verbose': False}
    mix_lin_cvar_d = TaylorMixtureLinearCVaRControlCostFunctional(control_model, prior, None, mix_lin_settings_d)
    mix_lin_cvar_d.cost(z_test, order=0)

    mix_quad_settings_d = {'beta': 0.95, 'N_mix': 7, 'direction': direction, 'N_tr': 10, 'N_mc': 1000, 'verbose': False}
    mix_quad_cvar_d = TaylorMixtureQuadraticCVaRControlCostFunctional(control_model, prior, None, mix_quad_settings_d)
    mix_quad_cvar_d.cost(z_test, order=0)

    lin_err = abs(mix_lin_cvar_d.cvar - saa_cvar2)/saa_cvar2 * 100
    quad_err = abs(mix_quad_cvar_d.cvar - saa_cvar2)/saa_cvar2 * 100
    print(f'{direction.upper():4s}: Linear CVaR={mix_lin_cvar_d.cvar:.4e} ({lin_err:5.1f}%), '
          f'Quad CVaR={mix_quad_cvar_d.cvar:.4e} ({quad_err:5.1f}%)')

print('\n' + '='*70)
print('Done!')
print('='*70)
