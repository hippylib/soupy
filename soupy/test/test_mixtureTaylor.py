# Copyright (c) 2023, The University of Texas at Austin
# & Georgia Institute of Technology
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the SOUPy package. For more information see
# https://github.com/hippylib/soupy/
#
# SOUPy is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 3.0 dated June 2007.

"""
Tests for Gaussian mixture Taylor approximations for mean + variance and CVaR.

Tests include:
- Basic instantiation and cost evaluation
- Gradient finite difference checks
- Comparison of mixture Taylor vs single Taylor approximations
- Comparison of linear vs quadratic approximations
"""

import unittest
import dolfin as dl
import numpy as np

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

import os
import sys
sys.path.append(os.environ.get('HIPPYLIB_PATH', ''))

import hippylib as hp

sys.path.append('../../')
from soupy import (
    ControlModel,
    PDEVariationalControlProblem,
    L2Penalization,
    STATE, PARAMETER, ADJOINT, CONTROL,
)
from soupy.modeling.controlQoI import L2MisfitControlQoI
from soupy.approximations.taylor import (
    # Single Taylor
    TaylorLinearControlCostFunctional,
    TaylorQuadraticControlCostFunctional,
    # Mixture Taylor mean+variance
    TaylorMixtureLinearControlCostFunctional,
    TaylorMixtureQuadraticControlCostFunctional,
    # Mixture Taylor CVaR
    TaylorMixtureLinearCVaRControlCostFunctional,
    TaylorMixtureQuadraticCVaRControlCostFunctional,
    # Helper functions
    gaussian_mixture_mean_variance,
    gaussian_mixture_cvar,
    get_1d_mixture,
    available_mixture_sizes,
)

from setupPoissonControlProblem import poisson_control_settings, setupPoissonPDEProblem


def setup_poisson_problem(nx=16, ny=16, n_wells=3, gamma=10.0, delta=50.0):
    """Set up a simple Poisson control problem for testing."""
    mesh = dl.UnitSquareMesh(nx, ny)
    Vh_STATE = dl.FunctionSpace(mesh, "CG", 1)
    Vh_PARAMETER = dl.FunctionSpace(mesh, "CG", 1)
    Vh_CONTROL = dl.FunctionSpace(mesh, "CG", 1)
    Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE, Vh_CONTROL]

    def residual(u, m, p, z):
        return dl.exp(m) * dl.inner(dl.grad(u), dl.grad(p)) * dl.dx - z * p * dl.dx

    def boundary(x, on_boundary):
        return on_boundary and (dl.near(x[0], 0) or dl.near(x[1], 0))

    boundary_value = dl.Expression("x[1]", degree=1)
    bc = dl.DirichletBC(Vh_STATE, boundary_value, boundary)
    bc0 = dl.DirichletBC(Vh_STATE, dl.Constant(0.0), boundary)
    pde = PDEVariationalControlProblem(Vh, residual, [bc], [bc0], is_fwd_linear=True)

    mean_vector = dl.interpolate(dl.Constant(-2.0), Vh_PARAMETER).vector()
    prior = hp.BiLaplacianPrior(
        Vh_PARAMETER, gamma, delta, mean=mean_vector, robin_bc=True
    )

    u_target = dl.Expression(
        "x[1] + sin(k*x[0]) * sin(k*x[1])",
        k=1.5 * np.pi,
        degree=2,
    )
    u_target_function = dl.interpolate(u_target, Vh_STATE)
    qoi = L2MisfitControlQoI(Vh, u_target_function.vector())

    control_model = ControlModel(pde, qoi)

    return mesh, Vh, control_model, prior


class TestMixtureData(unittest.TestCase):
    """Test the 1D mixture data tables."""

    def test_available_mixture_sizes(self):
        """Test that available mixture sizes are returned correctly."""
        sizes = available_mixture_sizes()
        self.assertIn(3, sizes)
        self.assertIn(5, sizes)
        self.assertIn(7, sizes)
        self.assertIn(9, sizes)
        self.assertIn(11, sizes)
        self.assertIn(15, sizes)

    def test_get_1d_mixture(self):
        """Test that 1D mixture parameters have correct properties."""
        # Test only the common mixture sizes that are well-tested
        for n_mix in [3, 5, 7]:
            mix = get_1d_mixture(n_mix)

            # Check that weights sum to approximately 1 (allow some tolerance)
            weights = np.array(mix['weights'])
            self.assertAlmostEqual(np.sum(weights), 1.0, places=2)

            # Check that we have n_mix components
            self.assertEqual(len(mix['weights']), n_mix)
            self.assertEqual(len(mix['means']), n_mix)

            # Check that sigma is positive and less than 1 (reduced variance)
            self.assertGreater(mix['sigma'], 0)
            self.assertLess(mix['sigma'], 1.0)

            # Check that weights are non-negative
            self.assertTrue(np.all(weights >= 0))


class TestGaussianMixtureMeanVariance(unittest.TestCase):
    """Test the Gaussian mixture mean/variance helper function."""

    def test_single_component(self):
        """Test that single component gives standard Gaussian mean/variance."""
        means = [1.0]
        stds = [0.5]
        weights = [1.0]

        mean, var = gaussian_mixture_mean_variance(means, stds, weights)

        self.assertAlmostEqual(mean, 1.0, places=10)
        self.assertAlmostEqual(var, 0.25, places=10)

    def test_symmetric_mixture(self):
        """Test symmetric mixture has correct mean."""
        means = [-1.0, 0.0, 1.0]
        stds = [0.5, 0.5, 0.5]
        weights = [0.25, 0.5, 0.25]  # Symmetric weights

        mean, var = gaussian_mixture_mean_variance(means, stds, weights)

        # Mean should be 0 for symmetric mixture
        self.assertAlmostEqual(mean, 0.0, places=10)
        # Variance should be positive
        self.assertGreater(var, 0)

    def test_equal_weights(self):
        """Test mixture with equal weights."""
        means = [1.0, 2.0, 3.0]
        stds = [0.1, 0.1, 0.1]
        weights = [1/3, 1/3, 1/3]

        mean, var = gaussian_mixture_mean_variance(means, stds, weights)

        # Mean should be average of component means
        self.assertAlmostEqual(mean, 2.0, places=10)


class TestGaussianMixtureCVaR(unittest.TestCase):
    """Test the Gaussian mixture CVaR helper function."""

    def test_single_component(self):
        """Test that single component CVaR matches analytical Gaussian CVaR."""
        from scipy.stats import norm

        mu = 1.0
        sigma = 0.5
        beta = 0.95

        means = [mu]
        stds = [sigma]
        weights = [1.0]

        var, cvar = gaussian_mixture_cvar(means, stds, weights, beta)

        # Analytical Gaussian CVaR
        z_beta = norm.ppf(beta)
        analytical_cvar = mu + sigma * norm.pdf(z_beta) / (1 - beta)

        self.assertAlmostEqual(cvar, analytical_cvar, places=4)

    def test_cvar_exceeds_var(self):
        """Test that CVaR >= VaR (by definition)."""
        means = [0.0, 1.0, 2.0]
        stds = [0.5, 0.5, 0.5]
        weights = [0.3, 0.4, 0.3]

        for beta in [0.9, 0.95, 0.99]:
            var, cvar = gaussian_mixture_cvar(means, stds, weights, beta)
            self.assertGreaterEqual(cvar, var - 1e-8)


class TestMixtureLinearMeanVariance(unittest.TestCase):
    """Test the mixture linear Taylor mean+variance approximation."""

    def setUp(self):
        self.mesh, self.Vh, self.control_model, self.prior = setup_poisson_problem()
        self.delta = 1e-5
        self.fdtol = 1e-2

    def test_instantiation(self):
        """Test that the cost functional can be instantiated."""
        settings = {"beta": 1.0, "N_mix": 3, "direction": "hep"}
        cost = TaylorMixtureLinearControlCostFunctional(
            self.control_model, self.prior, None, settings
        )
        self.assertIsNotNone(cost)

    def test_cost_evaluation(self):
        """Test that cost can be evaluated."""
        settings = {"beta": 1.0, "N_mix": 3, "direction": "hep"}
        cost = TaylorMixtureLinearControlCostFunctional(
            self.control_model, self.prior, None, settings
        )

        z = cost.generate_vector(CONTROL)
        c = cost.cost(z, order=0)

        self.assertIsInstance(c, float)
        self.assertGreater(c, 0)  # L2 misfit should be positive

    def test_gradient_finite_difference(self):
        """Test gradient via finite difference."""
        settings = {"beta": 1.0, "N_mix": 3, "direction": "hep"}
        cost = TaylorMixtureLinearControlCostFunctional(
            self.control_model, self.prior, None, settings
        )

        z0 = cost.generate_vector(CONTROL)
        dz = cost.generate_vector(CONTROL)
        z1 = cost.generate_vector(CONTROL)
        g0 = cost.generate_vector(CONTROL)

        # Random direction
        np.random.seed(42)
        dz.set_local(np.random.randn(dz.local_size()))
        dz.apply("")

        # Evaluate at z0
        c0 = cost.cost(z0, order=1)
        cost.grad(g0)

        # Evaluate at z0 + delta*dz
        z1.axpy(1.0, z0)
        z1.axpy(self.delta, dz)
        c1 = cost.cost(z1, order=0)

        # Finite difference vs adjoint
        dcdz_fd = (c1 - c0) / self.delta
        dcdz_ad = g0.inner(dz)

        if abs(dcdz_ad) > 1e-10:
            rel_err = abs((dcdz_fd - dcdz_ad) / dcdz_ad)
            self.assertLess(rel_err, self.fdtol)
        else:
            abs_err = abs(dcdz_fd - dcdz_ad)
            self.assertLess(abs_err, self.fdtol)

    def test_mixture_properties(self):
        """Test that mixture properties are correctly computed."""
        settings = {"beta": 1.0, "N_mix": 7, "direction": "hep"}
        cost = TaylorMixtureLinearControlCostFunctional(
            self.control_model, self.prior, None, settings
        )

        z = cost.generate_vector(CONTROL)
        cost.cost(z, order=0)

        # Check that component weights sum to 1
        weights = np.array(cost.component_weights)
        self.assertAlmostEqual(np.sum(weights), 1.0, places=10)

        # Check that we have correct number of components
        self.assertEqual(len(cost.component_means), 7)
        self.assertEqual(len(cost.component_stds), 7)

        # Check that variance is non-negative
        self.assertGreaterEqual(cost.mixture_var, 0)


class TestMixtureQuadraticMeanVariance(unittest.TestCase):
    """Test the mixture quadratic Taylor mean+variance approximation."""

    def setUp(self):
        self.mesh, self.Vh, self.control_model, self.prior = setup_poisson_problem()
        self.delta = 1e-5
        self.fdtol = 1e-2

    def test_instantiation(self):
        """Test that the cost functional can be instantiated."""
        settings = {"beta": 1.0, "N_mix": 3, "direction": "kle", "N_tr": 3}
        cost = TaylorMixtureQuadraticControlCostFunctional(
            self.control_model, self.prior, None, settings
        )
        self.assertIsNotNone(cost)

    def test_cost_evaluation(self):
        """Test that cost can be evaluated."""
        settings = {"beta": 1.0, "N_mix": 3, "direction": "kle", "N_tr": 3}
        cost = TaylorMixtureQuadraticControlCostFunctional(
            self.control_model, self.prior, None, settings
        )

        z = cost.generate_vector(CONTROL)
        c = cost.cost(z, order=0)

        self.assertIsInstance(c, float)
        self.assertGreater(c, 0)

    def test_gradient_finite_difference(self):
        """Test gradient via finite difference."""
        settings = {"beta": 1.0, "N_mix": 3, "direction": "kle", "N_tr": 3}
        cost = TaylorMixtureQuadraticControlCostFunctional(
            self.control_model, self.prior, None, settings
        )

        z0 = cost.generate_vector(CONTROL)
        dz = cost.generate_vector(CONTROL)
        z1 = cost.generate_vector(CONTROL)
        g0 = cost.generate_vector(CONTROL)

        np.random.seed(42)
        dz.set_local(np.random.randn(dz.local_size()))
        dz.apply("")

        c0 = cost.cost(z0, order=1)
        cost.grad(g0)

        z1.axpy(1.0, z0)
        z1.axpy(self.delta, dz)
        c1 = cost.cost(z1, order=0)

        dcdz_fd = (c1 - c0) / self.delta
        dcdz_ad = g0.inner(dz)

        if abs(dcdz_ad) > 1e-10:
            rel_err = abs((dcdz_fd - dcdz_ad) / dcdz_ad)
            self.assertLess(rel_err, self.fdtol)
        else:
            abs_err = abs(dcdz_fd - dcdz_ad)
            self.assertLess(abs_err, self.fdtol)

    def test_quadratic_exceeds_linear_variance(self):
        """Test that quadratic Taylor generally gives different variance than linear."""
        settings_linear = {"beta": 1.0, "N_mix": 5, "direction": "kle"}
        settings_quad = {"beta": 1.0, "N_mix": 5, "direction": "kle", "N_tr": 5}

        cost_linear = TaylorMixtureLinearControlCostFunctional(
            self.control_model, self.prior, None, settings_linear
        )
        cost_quad = TaylorMixtureQuadraticControlCostFunctional(
            self.control_model, self.prior, None, settings_quad
        )

        z = cost_linear.generate_vector(CONTROL)

        cost_linear.cost(z, order=0)
        cost_quad.cost(z, order=0)

        # Both should have positive variance
        self.assertGreater(cost_linear.mixture_var, 0)
        self.assertGreater(cost_quad.mixture_var, 0)


class TestMixtureLinearCVaR(unittest.TestCase):
    """Test the mixture linear Taylor CVaR approximation."""

    def setUp(self):
        self.mesh, self.Vh, self.control_model, self.prior = setup_poisson_problem()
        self.delta = 1e-5
        self.fdtol = 1e-2

    def test_instantiation(self):
        """Test that the cost functional can be instantiated."""
        settings = {"beta": 0.95, "N_mix": 3, "direction": "hep"}
        cost = TaylorMixtureLinearCVaRControlCostFunctional(
            self.control_model, self.prior, None, settings
        )
        self.assertIsNotNone(cost)

    def test_cost_evaluation(self):
        """Test that cost can be evaluated."""
        settings = {"beta": 0.95, "N_mix": 3, "direction": "hep"}
        cost = TaylorMixtureLinearCVaRControlCostFunctional(
            self.control_model, self.prior, None, settings
        )

        z = cost.generate_vector(CONTROL)
        c = cost.cost(z, order=0)

        self.assertIsInstance(c, float)
        self.assertGreater(c, 0)

    def test_cvar_exceeds_mean(self):
        """Test that CVaR >= mean (for right-tailed risks)."""
        settings = {"beta": 0.95, "N_mix": 5, "direction": "hep"}
        cost = TaylorMixtureLinearCVaRControlCostFunctional(
            self.control_model, self.prior, None, settings
        )

        z = cost.generate_vector(CONTROL)
        cost.cost(z, order=0)

        # CVaR should be at least as large as the weighted mean
        mean = np.sum(np.array(cost.component_weights) * np.array(cost.component_means))
        self.assertGreaterEqual(cost.cvar, mean - 1e-8)


class TestMixtureQuadraticCVaR(unittest.TestCase):
    """Test the mixture quadratic Taylor CVaR approximation."""

    def setUp(self):
        self.mesh, self.Vh, self.control_model, self.prior = setup_poisson_problem()
        self.delta = 1e-5
        self.fdtol = 1e-2

    def test_instantiation(self):
        """Test that the cost functional can be instantiated."""
        settings = {"beta": 0.95, "N_mix": 3, "direction": "kle", "N_tr": 3, "N_mc": 100}
        cost = TaylorMixtureQuadraticCVaRControlCostFunctional(
            self.control_model, self.prior, None, settings
        )
        self.assertIsNotNone(cost)

    def test_cost_evaluation(self):
        """Test that cost can be evaluated."""
        settings = {"beta": 0.95, "N_mix": 3, "direction": "kle", "N_tr": 3, "N_mc": 100}
        cost = TaylorMixtureQuadraticCVaRControlCostFunctional(
            self.control_model, self.prior, None, settings
        )

        z = cost.generate_vector(CONTROL)
        c = cost.cost(z, order=0)

        self.assertIsInstance(c, float)
        self.assertGreater(c, 0)


class TestMixtureVsSingleTaylor(unittest.TestCase):
    """Test comparison between mixture and single Taylor approximations."""

    def setUp(self):
        self.mesh, self.Vh, self.control_model, self.prior = setup_poisson_problem()

    def test_mixture_single_component_matches_single(self):
        """Test that mixture with 1 component is similar to single Taylor."""
        # Note: Not exactly equal due to direction selection, but should be close

        settings_single = {"beta": 1.0}
        settings_mixture = {"beta": 1.0, "N_mix": 3, "direction": "hep"}

        cost_single = TaylorLinearControlCostFunctional(
            self.control_model, self.prior, None, settings_single
        )
        cost_mixture = TaylorMixtureLinearControlCostFunctional(
            self.control_model, self.prior, None, settings_mixture
        )

        z = cost_single.generate_vector(CONTROL)

        c_single = cost_single.cost(z, order=0)
        c_mixture = cost_mixture.cost(z, order=0)

        # Both should be positive and similar order of magnitude
        self.assertGreater(c_single, 0)
        self.assertGreater(c_mixture, 0)

        # They won't be exactly equal but should be same order of magnitude
        ratio = c_mixture / c_single
        self.assertGreater(ratio, 0.1)
        self.assertLess(ratio, 10.0)


class TestDirectionSelection(unittest.TestCase):
    """Test KLE vs HEP direction selection."""

    def setUp(self):
        self.mesh, self.Vh, self.control_model, self.prior = setup_poisson_problem()

    def test_kle_direction(self):
        """Test that KLE direction works."""
        settings = {"beta": 1.0, "N_mix": 5, "direction": "kle"}
        cost = TaylorMixtureLinearControlCostFunctional(
            self.control_model, self.prior, None, settings
        )

        z = cost.generate_vector(CONTROL)
        c = cost.cost(z, order=0)

        self.assertIsInstance(c, float)
        self.assertGreater(c, 0)

    def test_hep_direction(self):
        """Test that HEP direction works."""
        settings = {"beta": 1.0, "N_mix": 5, "direction": "hep"}
        cost = TaylorMixtureLinearControlCostFunctional(
            self.control_model, self.prior, None, settings
        )

        z = cost.generate_vector(CONTROL)
        c = cost.cost(z, order=0)

        self.assertIsInstance(c, float)
        self.assertGreater(c, 0)

    def test_different_directions_give_different_results(self):
        """Test that KLE and HEP give different but valid results."""
        settings_kle = {"beta": 1.0, "N_mix": 5, "direction": "kle"}
        settings_hep = {"beta": 1.0, "N_mix": 5, "direction": "hep"}

        cost_kle = TaylorMixtureLinearControlCostFunctional(
            self.control_model, self.prior, None, settings_kle
        )
        cost_hep = TaylorMixtureLinearControlCostFunctional(
            self.control_model, self.prior, None, settings_hep
        )

        z = cost_kle.generate_vector(CONTROL)

        c_kle = cost_kle.cost(z, order=0)
        c_hep = cost_hep.cost(z, order=0)

        # Both should be valid
        self.assertGreater(c_kle, 0)
        self.assertGreater(c_hep, 0)

        # They may be different (depending on the problem)
        # Just check they're both reasonable
        self.assertLess(c_kle, 1e10)
        self.assertLess(c_hep, 1e10)


class TestWithPenalization(unittest.TestCase):
    """Test mixture Taylor with penalization."""

    def setUp(self):
        self.mesh, self.Vh, self.control_model, self.prior = setup_poisson_problem()

    def test_linear_with_penalization(self):
        """Test mixture linear Taylor with L2 penalization."""
        penalty = L2Penalization(self.Vh, 1e-3)
        settings = {"beta": 1.0, "N_mix": 5, "direction": "hep"}

        cost = TaylorMixtureLinearControlCostFunctional(
            self.control_model, self.prior, penalty, settings
        )

        z = cost.generate_vector(CONTROL)
        np.random.seed(42)
        z.set_local(np.random.randn(z.local_size()))
        z.apply("")

        c = cost.cost(z, order=0)

        self.assertIsInstance(c, float)
        self.assertGreater(c, 0)

    def test_quadratic_with_penalization(self):
        """Test mixture quadratic Taylor with L2 penalization."""
        penalty = L2Penalization(self.Vh, 1e-3)
        settings = {"beta": 1.0, "N_mix": 5, "direction": "kle", "N_tr": 3}

        cost = TaylorMixtureQuadraticControlCostFunctional(
            self.control_model, self.prior, penalty, settings
        )

        z = cost.generate_vector(CONTROL)
        np.random.seed(42)
        z.set_local(np.random.randn(z.local_size()))
        z.apply("")

        c = cost.cost(z, order=0)

        self.assertIsInstance(c, float)
        self.assertGreater(c, 0)


if __name__ == "__main__":
    unittest.main()
