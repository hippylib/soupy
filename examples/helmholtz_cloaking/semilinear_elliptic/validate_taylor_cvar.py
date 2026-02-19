#!/usr/bin/env python
"""Validate Taylor-CVaR against SAA-CVaR."""

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
print('Validation: Comparing Taylor-CVaR with SAA-CVaR')
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
# Set some non-zero control values
z_np = np.ones(z_test.get_local().shape) * 2.0  # Set all controls to 2.0
z_test.set_local(z_np)
z_test.apply("")

print('\nTest at z = 2.0 (non-zero control)')
print('-'*70)

# 1. Taylor Linear CVaR
lin_settings = {'beta': 0.95, 'verbose': False}
lin_cvar = TaylorLinearCVaRControlCostFunctional(control_model, prior, None, lin_settings)
lin_cost = lin_cvar.cost(z_test, order=0)
print(f'Taylor Linear CVaR:')
print(f'  Mean (Q0):     {lin_cvar.lin_mean:.6e}')
print(f'  Std:           {lin_cvar.lin_std:.6e}')
print(f'  CVaR (0.95):   {lin_cvar.cvar:.6e}')

# 2. Taylor Quadratic CVaR (with more modes to capture more variance)
quad_settings = {'beta': 0.95, 'N_tr': 50, 'N_mc': 5000, 'verbose': False}
quad_cvar = TaylorQuadraticCVaRControlCostFunctional(control_model, prior, None, quad_settings)
quad_cost = quad_cvar.cost(z_test, order=0)
print(f'\nTaylor Quadratic CVaR:')
print(f'  Q0:            {quad_cvar.Q_0:.6e}')
print(f'  CVaR (0.95):   {quad_cvar.cvar:.6e}')
print(f'  Eigenvalues:   {quad_cvar.d[:5]}')

# 3. SAA-based CVaR (ground truth)
print(f'\nSAA-based CVaR (N=64 samples):')
saa_settings = soupy.superquantileRiskMeasureSAASettings()
saa_settings['beta'] = 0.95
saa_settings['sample_size'] = 64
saa_settings['seed'] = 1
saa_rm = soupy.SuperquantileRiskMeasureSAA(control_model, prior, settings=saa_settings)

# Create augmented vector for CVaR (z, t)
zt = saa_rm.generate_vector(soupy.CONTROL)
zt.get_vector().zero()
zt.get_vector().axpy(1.0, z_test)  # Set z to same non-zero control
zt.set_scalar(0.0)  # t = 0 initially

saa_rm.computeComponents(zt, order=0)
saa_cvar = saa_rm.superquantile()
q_samples = saa_rm.gather_samples()

print(f'  Sample mean:   {np.mean(q_samples):.6e}')
print(f'  Sample std:    {np.std(q_samples):.6e}')
print(f'  CVaR (0.95):   {saa_cvar:.6e}')
print(f'  Sample min:    {np.min(q_samples):.6e}')
print(f'  Sample max:    {np.max(q_samples):.6e}')
print(f'  95th pctile:   {np.percentile(q_samples, 95):.6e}')

# 4. Compare
print('\n' + '='*70)
print('Comparison Summary')
print('='*70)
print(f'                    Mean          Std           CVaR(0.95)')
print(f'Taylor Linear:      {lin_cvar.lin_mean:.4e}    {lin_cvar.lin_std:.4e}    {lin_cvar.cvar:.4e}')
print(f'Taylor Quadratic:   {quad_cvar.Q_0:.4e}    N/A           {quad_cvar.cvar:.4e}')
print(f'SAA (N=64):         {np.mean(q_samples):.4e}    {np.std(q_samples):.4e}    {saa_cvar:.4e}')
print()
print(f'Relative error (Linear vs SAA):    {abs(lin_cvar.cvar - saa_cvar)/saa_cvar * 100:.1f}%')
print(f'Relative error (Quadratic vs SAA): {abs(quad_cvar.cvar - saa_cvar)/saa_cvar * 100:.1f}%')

# 5. More thorough test with larger SAA
print('\n' + '='*70)
print('Large-sample SAA reference (N=256):')
print('='*70)
saa_settings2 = soupy.superquantileRiskMeasureSAASettings()
saa_settings2['beta'] = 0.95
saa_settings2['sample_size'] = 256
saa_settings2['seed'] = 42
saa_rm2 = soupy.SuperquantileRiskMeasureSAA(control_model, prior, settings=saa_settings2)
zt2 = saa_rm2.generate_vector(soupy.CONTROL)
zt2.get_vector().zero()
zt2.get_vector().axpy(1.0, z_test)  # Set z to same non-zero control
zt2.set_scalar(0.0)
saa_rm2.computeComponents(zt2, order=0)
saa_cvar2 = saa_rm2.superquantile()
q_samples2 = saa_rm2.gather_samples()

print(f'SAA (N=256):        {np.mean(q_samples2):.4e}    {np.std(q_samples2):.4e}    {saa_cvar2:.4e}')
print()
print(f'Relative error (Linear vs SAA-256):    {abs(lin_cvar.cvar - saa_cvar2)/saa_cvar2 * 100:.1f}%')
print(f'Relative error (Quadratic vs SAA-256): {abs(quad_cvar.cvar - saa_cvar2)/saa_cvar2 * 100:.1f}%')

# 6. Investigate quadratic approximation
print('\n' + '='*70)
print('Quadratic Taylor Surrogate Analysis')
print('='*70)
print(f'All eigenvalues: {quad_cvar.d}')
print(f'Positive eigenvalues: {quad_cvar.d[quad_cvar.d > 0]}')
print(f'Negative eigenvalues: {quad_cvar.d[quad_cvar.d < 0]}')

# Check surrogate sample distribution
Q_surr = quad_cvar.Q_surrogate
print(f'\nSurrogate samples (N={len(Q_surr)}):')
print(f'  Mean:      {np.mean(Q_surr):.4e}')
print(f'  Std:       {np.std(Q_surr):.4e}')
print(f'  Min:       {np.min(Q_surr):.4e}')
print(f'  Max:       {np.max(Q_surr):.4e}')
print(f'  95th pct:  {np.percentile(Q_surr, 95):.4e}')

# Compare distributions
print(f'\nComparison (SAA-256 vs Taylor Quadratic Surrogate):')
print(f'  Mean diff:     {np.mean(Q_surr) - np.mean(q_samples2):.4e}')
print(f'  Std diff:      {np.std(Q_surr) - np.std(q_samples2):.4e}')
print(f'  95th pct diff: {np.percentile(Q_surr, 95) - np.percentile(q_samples2, 95):.4e}')
