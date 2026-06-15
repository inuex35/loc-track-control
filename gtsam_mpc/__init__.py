"""gtsam-mpc: estimation, control and tracking as factor-graph optimization.

The package is organized around one idea: estimation, control and tracking are
all maximum-a-posteriori inference on factor graphs built from a shared
vocabulary of factors.

    models       -- dynamics model (kinematic bicycle)
    factors      -- reusable GTSAM CustomFactor builders (the vocabulary)
    constraints  -- inequality strategies (barrier / augmented Lagrangian / slack)
    control      -- BicycleMPC
    estimation   -- MovingHorizonEstimator
    joint        -- JointEstimatorMPC, JointLocTrackControl (one graph)
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
from .control import BicycleMPC, MPCResult
from .estimation import MovingHorizonEstimator
from .joint import JointEstimatorMPC, JointLocTrackControl
from .models import BicycleModel

__all__ = [
    # models
    "BicycleModel",
    # control
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
    # joint
    "JointEstimatorMPC",
    "JointLocTrackControl",
    # submodules
    "factors",
    "paths",
]

__version__ = "0.3.0"
