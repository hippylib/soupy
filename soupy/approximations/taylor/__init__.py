"""Taylor-expansion-based approximations for risk-aware control."""

from .constant import TaylorConstantControlCostFunctional  # noqa: F401
from .linear import TaylorLinearControlCostFunctional  # noqa: F401
from .quadratic import TaylorQuadraticControlCostFunctional  # noqa: F401
from .linear_cvar import TaylorLinearCVaRControlCostFunctional, gaussian_cvar  # noqa: F401
from .quadratic_cvar import (
    TaylorQuadraticCVaRControlCostFunctional,
    surrogate_cvar_from_samples,
)  # noqa: F401
from .mixture_linear_cvar import (
    TaylorMixtureLinearCVaRControlCostFunctional,
    gaussian_mixture_cvar,
)  # noqa: F401
from .mixture_quadratic_cvar import (
    TaylorMixtureQuadraticCVaRControlCostFunctional,
)  # noqa: F401
from .alternating_hep_linear_cvar import (
    TaylorAlternatingHEPLinearCVaRControlCostFunctional,
)  # noqa: F401
from .alternating_hep_quadratic_cvar import (
    TaylorAlternatingHEPQuadraticCVaRControlCostFunctional,
)  # noqa: F401
from .mixture_linear import (
    TaylorMixtureLinearControlCostFunctional,
    gaussian_mixture_mean_variance,
)  # noqa: F401
from .mixture_quadratic import (
    TaylorMixtureQuadraticControlCostFunctional,
    quadratic_mixture_mean_variance_analytical,
)  # noqa: F401
from .mixture_data import get_1d_mixture, available_mixture_sizes  # noqa: F401
from .settings import (
    taylor_constant_settings,
    taylor_linear_settings,
    taylor_quadratic_settings,
    taylor_linear_cvar_settings,
    taylor_quadratic_cvar_settings,
    taylor_mixture_linear_cvar_settings,
    taylor_mixture_quadratic_cvar_settings,
    taylor_mixture_linear_settings,
    taylor_mixture_quadratic_settings,
)  # noqa: F401

__all__ = [
    # Single Taylor mean-variance approximations
    "TaylorConstantControlCostFunctional",
    "TaylorLinearControlCostFunctional",
    "TaylorQuadraticControlCostFunctional",
    # Single Taylor CVaR approximations
    "TaylorLinearCVaRControlCostFunctional",
    "TaylorQuadraticCVaRControlCostFunctional",
    # Gaussian mixture Taylor CVaR approximations
    "TaylorMixtureLinearCVaRControlCostFunctional",
    "TaylorMixtureQuadraticCVaRControlCostFunctional",
    "TaylorAlternatingHEPLinearCVaRControlCostFunctional",
    "TaylorAlternatingHEPQuadraticCVaRControlCostFunctional",
    # Gaussian mixture Taylor mean-variance approximations
    "TaylorMixtureLinearControlCostFunctional",
    "TaylorMixtureQuadraticControlCostFunctional",
    # Helper functions
    "gaussian_cvar",
    "gaussian_mixture_cvar",
    "gaussian_mixture_mean_variance",
    "quadratic_mixture_mean_variance_analytical",
    "surrogate_cvar_from_samples",
    "get_1d_mixture",
    "available_mixture_sizes",
    # Settings
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
