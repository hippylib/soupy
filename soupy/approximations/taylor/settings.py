"""Parameter lists shared by Taylor approximation modules."""

from hippylib import ParameterList


def _fill_defaults(data, defaults):
    params = {}
    overrides = data.copy() if data is not None else {}
    for key, value in defaults.items():
        desc = value[1] if isinstance(value, (list, tuple)) and len(value) > 1 else ""
        if key in overrides:
            user_val = overrides[key]
            if isinstance(user_val, (list, tuple)) and len(user_val) == 2:
                params[key] = list(user_val)
            else:
                params[key] = [user_val, desc]
        else:
            params[key] = value
    return ParameterList(params)


def taylor_constant_settings(data=None):
    """Return default settings for the zeroth-order Taylor approximation."""

    defaults = {
        "beta": [0.0, "Weight applied to the variance term in the Taylor approximation"],
        "correction": [False, "Enable Monte Carlo correction of the Taylor surrogate"],
        "N_mc": [0, "Number of Monte Carlo samples used for the correction"],
        "dim": [1, "Dimension of the control variable (used to build mass matrices)"],
        "verbose": [False, "Print detailed output during optimization"],
    }

    return _fill_defaults(data or {}, defaults)


def taylor_linear_settings(data=None):
    """Return default settings for the first-order Taylor approximation."""

    defaults = {
        "beta": [0.0, "Weight applied to the variance term"],
        "correction": [False, "Enable Monte Carlo correction of the Taylor surrogate"],
        "N_mc": [0, "Number of Monte Carlo samples used for the correction"],
        "dim": [1, "Dimension of the control variable"],
        "verbose": [False, "Print detailed output during optimization"],
    }

    return _fill_defaults(data or {}, defaults)


def taylor_quadratic_settings(data=None):
    """Return default settings for the second-order Taylor approximation."""

    defaults = {
        "beta": [0.0, "Weight applied to the variance term"],
        "correction": [False, "Enable Monte Carlo correction of the Taylor surrogate"],
        "N_mc": [0, "Number of Monte Carlo samples used for the correction"],
        "N_tr": [5, "Number of dominant Hessian modes"],
        "dim": [1, "Dimension of the control variable"],
        "verbose": [False, "Print detailed output during optimization"],
    }

    return _fill_defaults(data or {}, defaults)


def taylor_linear_cvar_settings(data=None):
    """Return default settings for the first-order Taylor CVaR approximation.

    Uses analytical Gaussian CVaR formula since linear Taylor gives Q ~ N(μ, σ²).
    CVaR_β[Q] = μ + σ * φ(Φ⁻¹(β)) / (1-β)
    """

    defaults = {
        "beta": [0.95, "CVaR risk level (e.g., 0.95 for 95% CVaR)"],
        "correction": [False, "Enable Monte Carlo correction of the Taylor surrogate"],
        "N_mc": [0, "Number of Monte Carlo samples used for the correction"],
        "epsilon": [1e-4, "Smoothing parameter for CVaR (used in correction)"],
        "dim": [1, "Dimension of the control variable"],
        "verbose": [False, "Print detailed output during optimization"],
    }

    return _fill_defaults(data or {}, defaults)


def taylor_quadratic_cvar_settings(data=None):
    """Return default settings for the second-order Taylor CVaR approximation.

    Uses surrogate MC sampling from the generalized chi-squared distribution.
    """

    defaults = {
        "beta": [0.95, "CVaR risk level (e.g., 0.95 for 95% CVaR)"],
        "N_tr": [5, "Number of dominant Hessian modes"],
        "N_mc": [1000, "Number of surrogate MC samples for CVaR estimation"],
        "epsilon": [1e-4, "Smoothing parameter for CVaR"],
        "correction": [False, "Enable Monte Carlo correction using PDE samples"],
        "N_mc_correction": [0, "Number of PDE MC samples for correction"],
        "dim": [1, "Dimension of the control variable"],
        "verbose": [False, "Print detailed output during optimization"],
    }

    return _fill_defaults(data or {}, defaults)


def taylor_mixture_linear_cvar_settings(data=None):
    """Return default settings for Gaussian mixture linear Taylor CVaR.

    Uses analytical Gaussian CVaR formula for each mixture component.
    The mixture decomposes the prior along a dominant direction (KLE or HEP).

    Reference:
        Chen, Villa, Ghattas (2024)
        "Gaussian mixture Taylor approximations for risk measures of PDEs"
    """

    defaults = {
        "beta": [0.95, "CVaR risk level (e.g., 0.95 for 95% CVaR)"],
        "N_mix": [7, "Number of mixture components (3, 5, 7, 9, 11, or 15)"],
        "direction": ["hep", "Direction selection: 'kle' or 'hep'"],
        "epsilon": [1e-4, "Smoothing parameter for CVaR optimization"],
        "dim": [1, "Dimension of the control variable"],
        "verbose": [False, "Print detailed output during optimization"],
    }

    return _fill_defaults(data or {}, defaults)


def taylor_mixture_quadratic_cvar_settings(data=None):
    """Return default settings for Gaussian mixture quadratic Taylor CVaR.

    Uses surrogate MC sampling from generalized chi-squared for each component.
    The mixture decomposes the prior along a dominant direction (KLE or HEP).

    Reference:
        Chen, Villa, Ghattas (2024)
        "Gaussian mixture Taylor approximations for risk measures of PDEs"
    """

    defaults = {
        "beta": [0.95, "CVaR risk level (e.g., 0.95 for 95% CVaR)"],
        "N_mix": [7, "Number of mixture components (3, 5, 7, 9, 11, or 15)"],
        "direction": ["hep", "Direction selection: 'kle' or 'hep'"],
        "N_tr": [10, "Number of dominant Hessian modes per component"],
        "N_mc": [1000, "Number of surrogate MC samples per component"],
        "epsilon": [1e-4, "Smoothing parameter for CVaR optimization"],
        "dim": [1, "Dimension of the control variable"],
        "verbose": [False, "Print detailed output during optimization"],
    }

    return _fill_defaults(data or {}, defaults)


def taylor_mixture_linear_settings(data=None):
    """Return default settings for Gaussian mixture linear Taylor mean + variance.

    Uses analytical formulas for Gaussian mixture mean and variance.
    The mixture decomposes the prior along a dominant direction (KLE or HEP).

    Risk measure: E[Q] + beta * Var[Q]

    Reference:
        Chen, Villa, Ghattas (2024)
        "Gaussian mixture Taylor approximations for risk measures of PDEs"
    """

    defaults = {
        "beta": [1.0, "Variance weight for risk measure"],
        "N_mix": [7, "Number of mixture components (3, 5, 7, 9, 11, or 15)"],
        "direction": ["hep", "Direction selection: 'kle' or 'hep'"],
        "dim": [1, "Dimension of the control variable"],
        "verbose": [False, "Print detailed output during optimization"],
    }

    return _fill_defaults(data or {}, defaults)


def taylor_mixture_quadratic_settings(data=None):
    """Return default settings for Gaussian mixture quadratic Taylor mean + variance.

    Uses analytical formulas for mean/variance and optional surrogate MC for validation.
    The mixture decomposes the prior along a dominant direction (KLE or HEP).

    Risk measure: E[Q] + beta * Var[Q]

    Reference:
        Chen, Villa, Ghattas (2024)
        "Gaussian mixture Taylor approximations for risk measures of PDEs"
    """

    defaults = {
        "beta": [1.0, "Variance weight for risk measure"],
        "N_mix": [7, "Number of mixture components (3, 5, 7, 9, 11, or 15)"],
        "direction": ["kle", "Direction selection: 'kle' or 'hep' (kle often better for quadratic)"],
        "N_tr": [10, "Number of dominant Hessian modes per component"],
        "N_mc": [0, "Number of surrogate MC samples per component (0 for analytical only)"],
        "dim": [1, "Dimension of the control variable"],
        "verbose": [False, "Print detailed output during optimization"],
    }

    return _fill_defaults(data or {}, defaults)


__all__ = [
    "taylor_constant_settings",
    "taylor_linear_settings",
    "taylor_quadratic_settings",
    "taylor_linear_cvar_settings",
    "taylor_quadratic_cvar_settings",
    "taylor_mixture_linear_cvar_settings",
    "taylor_mixture_quadratic_cvar_settings",
    "taylor_mixture_linear_settings",
    "taylor_mixture_quadratic_settings",
]
