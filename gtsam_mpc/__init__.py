"""gtsam-mpc: Linear Model Predictive Control as factor-graph optimization.

This package formulates finite-horizon linear-quadratic MPC as maximum-a-posteriori
inference on a Gaussian factor graph and solves it with GTSAM.
"""

from .bicycle import BicycleMPC, BicycleModel
from .mpc import LinearMPC, MPCResult
from .system import LinearSystem, double_integrator, point_mass_2d

__all__ = [
    "LinearMPC",
    "MPCResult",
    "LinearSystem",
    "double_integrator",
    "point_mass_2d",
    "BicycleModel",
    "BicycleMPC",
]

__version__ = "0.1.0"
