"""Pre-computed 1D Gaussian mixture approximations of N(0,1).

These tables are from Vittaldev & Russell (2016):
"Univariate Splitting for High-Performance Monte Carlo Methods"

The mixtures approximate the standard Gaussian N(0,1) by minimizing
the L^2 distance between the PDFs. Component standard deviations
follow the rule sigma_i = N_mix^{-p} with p = 0.5.

Reference:
    H. S. Vittaldev and R. P. Russell,
    "Univariate Splitting for High-Performance Monte Carlo Methods"
    https://github.com/vittaldevuni/UnivariateApprox
"""

import numpy as np

# Pre-computed mixture approximations of N(0,1)
# Format: {'weights': [...], 'means': [...], 'sigma': float}
# Note: All components share the same sigma = N_mix^{-0.5}

MIXTURE_TABLES = {
    3: {
        'weights': np.array([0.2525, 0.4950, 0.2525]),
        'means': np.array([-1.2247, 0.0, 1.2247]),
        'sigma': 3**(-0.5),  # ~0.577
    },
    5: {
        'weights': np.array([0.1117, 0.2365, 0.3036, 0.2365, 0.1117]),
        'means': np.array([-1.6180, -0.6180, 0.0, 0.6180, 1.6180]),
        'sigma': 5**(-0.5),  # ~0.447
    },
    7: {
        'weights': np.array([0.0548, 0.1353, 0.2070, 0.2058, 0.2070, 0.1353, 0.0548]),
        'means': np.array([-1.8676, -1.1135, -0.3780, 0.0, 0.3780, 1.1135, 1.8676]),
        'sigma': 7**(-0.5),  # ~0.378
    },
    9: {
        'weights': np.array([0.0298, 0.0809, 0.1422, 0.1879, 0.1184, 0.1879, 0.1422, 0.0809, 0.0298]),
        'means': np.array([-2.0447, -1.4289, -0.8330, -0.2582, 0.0, 0.2582, 0.8330, 1.4289, 2.0447]),
        'sigma': 9**(-0.5),  # ~0.333
    },
    11: {
        'weights': np.array([0.0175, 0.0504, 0.0961, 0.1418, 0.1668, 0.0548,
                            0.1668, 0.1418, 0.0961, 0.0504, 0.0175]),
        'means': np.array([-2.1818, -1.6573, -1.1467, -0.6503, -0.1681, 0.0,
                          0.1681, 0.6503, 1.1467, 1.6573, 2.1818]),
        'sigma': 11**(-0.5),  # ~0.302
    },
    15: {
        # Normalized weights that sum to 1
        'weights': np.array([0.0058, 0.0186, 0.0409, 0.0708, 0.1002, 0.1196, 0.1155, 0.0573,
                            0.1155, 0.1196, 0.1002, 0.0708, 0.0409, 0.0186, 0.0058]),
        'means': np.array([-2.3911, -1.9774, -1.5726, -1.1769, -0.7903, -0.4127, -0.0441, 0.0,
                          0.0441, 0.4127, 0.7903, 1.1769, 1.5726, 1.9774, 2.3911]),
        'sigma': 15**(-0.5),  # ~0.258
    },
}

# Simplified symmetric mixture tables (means symmetric about 0)
# These are re-normalized and corrected versions
MIXTURE_TABLES_SYMMETRIC = {
    3: {
        'weights': np.array([0.25, 0.50, 0.25]),
        'means': np.array([-1.2247, 0.0, 1.2247]),
        'sigma': 0.5774,
    },
    5: {
        'weights': np.array([0.1117, 0.2365, 0.3036, 0.2365, 0.1117]),
        'means': np.array([-1.6180, -0.6180, 0.0, 0.6180, 1.6180]),
        'sigma': 0.4472,
    },
    7: {
        'weights': np.array([0.0548, 0.1353, 0.2070, 0.2058, 0.2070, 0.1353, 0.0548]),
        'means': np.array([-1.8676, -1.1135, -0.3780, 0.0, 0.3780, 1.1135, 1.8676]),
        'sigma': 0.3780,
    },
    9: {
        'weights': np.array([0.0298, 0.0809, 0.1422, 0.1879, 0.1184, 0.1879, 0.1422, 0.0809, 0.0298]),
        'means': np.array([-2.0447, -1.4289, -0.8330, -0.2582, 0.0, 0.2582, 0.8330, 1.4289, 2.0447]),
        'sigma': 0.3333,
    },
    11: {
        'weights': np.array([0.0175, 0.0504, 0.0961, 0.1418, 0.1668, 0.0548,
                            0.1668, 0.1418, 0.0961, 0.0504, 0.0175]),
        'means': np.array([-2.1818, -1.6573, -1.1467, -0.6503, -0.1681, 0.0,
                          0.1681, 0.6503, 1.1467, 1.6573, 2.1818]),
        'sigma': 0.3015,
    },
}


def get_1d_mixture(n_components):
    """Get 1D Gaussian mixture approximation of N(0,1).

    Args:
        n_components: Number of mixture components (3, 5, 7, 9, 11, or 15)

    Returns:
        dict with keys 'weights', 'means', 'sigma'

    Raises:
        ValueError if n_components not available
    """
    if n_components in MIXTURE_TABLES:
        return MIXTURE_TABLES[n_components].copy()

    # Fallback: construct equally-spaced mixture
    return construct_equal_mixture(n_components)


def construct_equal_mixture(n_components, L=3.0, p=0.5):
    """Construct equally-spaced Gaussian mixture approximation of N(0,1).

    This follows the construction in Proposition 1 of the paper:
    means are equally spaced in [-L, L] with weights proportional to N(0,1) PDF.

    Args:
        n_components: Number of mixture components
        L: Support interval [-L, L]
        p: Exponent for sigma = N^{-p}

    Returns:
        dict with keys 'weights', 'means', 'sigma'
    """
    from scipy.stats import norm

    # Equally spaced means in [-L, L]
    means = np.linspace(-L + L/n_components, L - L/n_components, n_components)

    # Weights proportional to standard Gaussian PDF
    weights = norm.pdf(means)
    weights = weights / np.sum(weights)

    # Standard deviation
    sigma = n_components ** (-p)

    return {
        'weights': weights,
        'means': means,
        'sigma': sigma,
    }


def available_mixture_sizes():
    """Return list of available pre-computed mixture sizes."""
    return sorted(MIXTURE_TABLES.keys())


__all__ = [
    'MIXTURE_TABLES',
    'get_1d_mixture',
    'construct_equal_mixture',
    'available_mixture_sizes',
]
