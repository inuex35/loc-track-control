"""gtsam-mpc: estimation, control and tracking as factor-graph optimization.

The package is organized around one idea: estimation, control and tracking are
all maximum-a-posteriori inference on factor graphs built from a shared
vocabulary of factors.

    models       -- dynamics models (linear systems, kinematic bicycle)
    factors      -- reusable GTSAM CustomFactor builders (the vocabulary)
    constraints  -- inequality strategies (barrier / augmented Lagrangian / slack)
    control      -- LinearMPC, BicycleMPC
    estimation   -- MovingHorizonEstimator, ConstantVelocityTracker
    joint        -- JointEstimatorMPC (estimation window + control horizon, one graph)
    paths        -- reference paths and path-tracking geometry
"""

from . import factors, paths
from .constraints import (
    AugmentedLagrangianStrategy,
    BarrierStrategy,
    ConstraintStrategy,
    Inequality,
    SlackStrategy,
    make_strategy,
)
from .control import BicycleMPC, LinearMPC, MPCResult
from .estimation import ConstantVelocityTracker, MovingHorizonEstimator
from .joint import JointEstimatorMPC, JointLocTrackControl
from .models import BicycleModel, LinearSystem, double_integrator, point_mass_2d

__all__ = [
    # models
    "LinearSystem",
    "double_integrator",
    "point_mass_2d",
    "BicycleModel",
    # control
    "LinearMPC",
    "BicycleMPC",
    "MPCResult",
    # constraints
    "ConstraintStrategy",
    "BarrierStrategy",
    "AugmentedLagrangianStrategy",
    "SlackStrategy",
    "Inequality",
    "make_strategy",
    # estimation
    "MovingHorizonEstimator",
    "ConstantVelocityTracker",
    # joint
    "JointEstimatorMPC",
    "JointLocTrackControl",
    # submodules
    "factors",
    "paths",
]

__version__ = "0.2.0"
