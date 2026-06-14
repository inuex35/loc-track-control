"""Inequality-constraint handling strategies for the nonlinear MPC.

An MPC inequality ``g(x) <= 0`` (input box, speed bound, obstacle keep-out) can
be imposed on a GTSAM least-squares graph in three ways, all sharing the same
:class:`Inequality` description (a variable key plus a ``g, dg/dx`` evaluator):

    * :class:`BarrierStrategy` -- a fixed one-sided quadratic penalty. Soft and
      cheap; the bound may be violated by an amount set by the weight.
    * :class:`AugmentedLagrangianStrategy` -- an outer loop that updates a
      multiplier per constraint so the bound is met tightly, without tuning a
      penalty weight (still a sequence of unconstrained GTSAM solves).
    * :class:`SlackStrategy` -- turns ``g <= 0`` into the hard equality
      ``g + s^2 = 0`` with a slack variable, so everything is solved inside a
      single GTSAM ``optimize()``.

Each strategy is given a *factory* that builds the base graph (hard dynamics +
costs), the list of inequalities, an initial guess and the optimizer params,
and returns the optimized ``Values``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import gtsam
import numpy as np
from gtsam.symbol_shorthand import S

# An inequality evaluator maps a variable's value to ``(g, dg/dx)`` with the
# convention ``g <= 0`` feasible.
Evaluator = Callable[[np.ndarray], "tuple[np.ndarray, np.ndarray]"]
GraphFactory = Callable[[], gtsam.NonlinearFactorGraph]


@dataclass
class Inequality:
    """An inequality ``g(.) <= 0`` on a single variable.

    Attributes:
        key: The GTSAM key of the constrained variable.
        dim: Number of scalar inequalities (rows of ``g``).
        evaluate: Maps the variable's value to ``(g, dg/dx)``.
        weight: Optional per-constraint barrier precision (see
            :class:`BarrierStrategy`); ``None`` uses the strategy default.
    """

    key: int
    dim: int
    evaluate: Evaluator
    weight: float | None = None


def _optimize(graph, values, params):
    return gtsam.LevenbergMarquardtOptimizer(graph, values, params).optimize()


def barrier_factor(ineq: "Inequality", weight: float) -> gtsam.CustomFactor:
    """One-sided quadratic penalty factor ``max(0, g)`` with precision ``weight``.

    The standalone builder used both by :class:`BarrierStrategy` and by graphs
    that mix a barrier penalty into a larger custom graph (e.g. the joint
    estimator+controller).
    """
    noise = gtsam.noiseModel.Diagonal.Precisions(np.full(ineq.dim, weight))
    ev = ineq.evaluate

    def error(this, values, H):
        g, J = ev(values.atVector(this.keys()[0]))
        active = g > 0.0
        if H is not None:
            H[0] = J * active[:, None]
        return np.where(active, g, 0.0)

    return gtsam.CustomFactor(noise, [ineq.key], error)


class ConstraintStrategy:
    """Strategy interface: solve the base graph subject to the inequalities."""

    def solve(self, build_base: GraphFactory, inequalities: list[Inequality],
              initial: gtsam.Values, params) -> gtsam.Values:
        raise NotImplementedError


class BarrierStrategy(ConstraintStrategy):
    """Fixed one-sided quadratic penalty: residual ``max(0, g)`` (soft)."""

    def __init__(self, weight: float = 300.0):
        self.weight = float(weight)

    def _factor(self, ineq: Inequality) -> gtsam.CustomFactor:
        weight = self.weight if ineq.weight is None else ineq.weight
        return barrier_factor(ineq, weight)

    def solve(self, build_base, inequalities, initial, params):
        graph = build_base()
        for ineq in inequalities:
            graph.add(self._factor(ineq))
        return _optimize(graph, initial, params)


class AugmentedLagrangianStrategy(ConstraintStrategy):
    """Augmented Lagrangian: outer loop updating per-constraint multipliers.

    The penalty factor for ``g <= 0`` has residual ``max(0, g + lam/rho)`` with
    precision ``rho``; its gradient on the active set is ``(rho*g + lam)*dg/dx``,
    exactly the AL term for an inequality. After each inner solve the multiplier
    is updated ``lam <- max(0, lam + rho*g)`` and ``rho`` is scaled up until the
    worst violation is below ``tol``.
    """

    def __init__(self, iterations: int = 5, rho: float = 100.0,
                 rho_scale: float = 10.0, rho_max: float = 1e6, tol: float = 1e-3):
        self.iterations = int(iterations)
        self.rho0 = float(rho)
        self.rho_scale = float(rho_scale)
        self.rho_max = float(rho_max)
        self.tol = float(tol)

    @staticmethod
    def _factor(ineq: Inequality, lam: np.ndarray, rho: float) -> gtsam.CustomFactor:
        noise = gtsam.noiseModel.Diagonal.Precisions(np.full(ineq.dim, rho))
        ev = ineq.evaluate

        def error(this, values, H):
            g, J = ev(values.atVector(this.keys()[0]))
            shifted = g + lam / rho
            active = shifted > 0.0
            if H is not None:
                H[0] = J * active[:, None]
            return np.where(active, shifted, 0.0)

        return gtsam.CustomFactor(noise, [ineq.key], error)

    def solve(self, build_base, inequalities, initial, params):
        base = build_base()
        lams = [np.zeros(ineq.dim) for ineq in inequalities]
        rho = self.rho0
        values = initial

        for _ in range(self.iterations):
            graph = gtsam.NonlinearFactorGraph(base)
            for ineq, lam in zip(inequalities, lams):
                graph.add(self._factor(ineq, lam, rho))
            values = _optimize(graph, values, params)

            worst = 0.0
            for idx, ineq in enumerate(inequalities):
                g, _ = ineq.evaluate(values.atVector(ineq.key))
                lams[idx] = np.maximum(0.0, lams[idx] + rho * g)
                worst = max(worst, float(g.max()) if g.size else 0.0)
            if worst < self.tol:
                break
            rho = min(rho * self.rho_scale, self.rho_max)

        return values


class SlackStrategy(ConstraintStrategy):
    """Slack reformulation: ``g(x) + s^2 = 0`` hard equality, single solve.

    Each inequality gets a slack variable ``S(i)``; forcing ``g + s^2 = 0`` makes
    ``g`` non-positive. Everything is solved in one GTSAM ``optimize()`` (no
    Python outer loop); GTSAM's elimination produces the multipliers implicitly.
    """

    @staticmethod
    def _factor(ineq: Inequality, skey) -> gtsam.CustomFactor:
        noise = gtsam.noiseModel.Constrained.All(ineq.dim)
        ev = ineq.evaluate

        def error(this, values, H):
            x = values.atVector(this.keys()[0])
            t = values.atVector(this.keys()[1])
            g, J = ev(x)
            if H is not None:
                H[0] = J
                H[1] = np.diag(2.0 * t)
            return g + t * t

        return gtsam.CustomFactor(noise, [ineq.key, skey], error)

    def solve(self, build_base, inequalities, initial, params):
        graph = build_base()
        values = gtsam.Values(initial)
        for i, ineq in enumerate(inequalities):
            skey = S(i)
            if not values.exists(skey):
                # Initialize the slack so g + s^2 ~= 0 at the warm-start point.
                g0, _ = ineq.evaluate(values.atVector(ineq.key))
                values.insert(skey, np.sqrt(np.maximum(0.0, -g0)))
            graph.add(self._factor(ineq, skey))
        return _optimize(graph, values, params)


def make_strategy(mode: str, *, barrier_weight: float = 300.0,
                  al_iterations: int = 5, al_rho: float = 100.0,
                  al_rho_scale: float = 10.0, al_rho_max: float = 1e6,
                  al_tol: float = 1e-3) -> ConstraintStrategy:
    """Build a strategy from a short mode name (``barrier`` / ``al`` / ``slack``)."""
    if mode == "barrier":
        return BarrierStrategy(weight=barrier_weight)
    if mode == "al":
        return AugmentedLagrangianStrategy(
            iterations=al_iterations, rho=al_rho, rho_scale=al_rho_scale,
            rho_max=al_rho_max, tol=al_tol)
    if mode == "slack":
        return SlackStrategy()
    raise ValueError(f"constraint_mode must be 'barrier', 'al' or 'slack', got {mode!r}")
