"""Reusable GTSAM ``CustomFactor`` builders -- the shared factor vocabulary.

Estimation, control and tracking in this package are all maximum-a-posteriori
inference on factor graphs built from the *same* small set of factors. Putting
the factor error functions here (instead of re-deriving them inside each
solver) is the core idea behind the three reference projects this package
echoes: one vocabulary of factors, many graphs.

Every builder returns a ``gtsam.CustomFactor`` with an analytic Jacobian.
Convention for the bicycle stack: state ``x = [x, y, theta, v]`` and control
``u = [a, delta]``.
"""

from __future__ import annotations

import gtsam
import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap an angle to ``(-pi, pi]``."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


# --- noise-model shorthands ----------------------------------------------
def information(matrix: np.ndarray):
    """Gaussian noise from an information (inverse-covariance) matrix."""
    return gtsam.noiseModel.Gaussian.Information(matrix)


def precisions(values: np.ndarray):
    """Diagonal Gaussian noise from per-component precisions (1/variance)."""
    return gtsam.noiseModel.Diagonal.Precisions(np.asarray(values, dtype=float))


def sigmas(values: np.ndarray):
    """Diagonal Gaussian noise from per-component standard deviations."""
    return gtsam.noiseModel.Diagonal.Sigmas(np.asarray(values, dtype=float))


def isotropic(dim: int, sigma: float):
    """Isotropic Gaussian noise of dimension ``dim``."""
    return gtsam.noiseModel.Isotropic.Sigma(dim, sigma)


def hard(dim: int):
    """A hard equality constraint of dimension ``dim``."""
    return gtsam.noiseModel.Constrained.All(dim)


# --- factor builders -----------------------------------------------------
def prior(key, mean: np.ndarray, noise) -> gtsam.CustomFactor:
    """Pull a variable toward ``mean`` (residual ``x - mean``)."""
    mean = np.asarray(mean, dtype=float)
    n = mean.shape[0]

    def error(this, values, H):
        if H is not None:
            H[0] = np.eye(n)
        return values.atVector(this.keys()[0]) - mean

    return gtsam.CustomFactor(noise, [key], error)


def dynamics(model, kx, ku, kx_next, noise) -> gtsam.CustomFactor:
    """Ternary dynamics ``x_{k+1} - f(x_k, u_k) = 0`` (control is a variable)."""
    n = model.n_states

    def error(this, values, H):
        k = this.keys()
        xk = values.atVector(k[0])
        uk = values.atVector(k[1])
        xk1 = values.atVector(k[2])
        if H is not None:
            dfdx, dfdu = model.jacobians(xk, uk)
            H[0] = -dfdx
            H[1] = -dfdu
            H[2] = np.eye(n)
        return xk1 - model.step(xk, uk)

    return gtsam.CustomFactor(noise, [kx, ku, kx_next], error)


def motion(model, kx, kx_next, u: np.ndarray, noise) -> gtsam.CustomFactor:
    """Binary motion ``x_{j} - f(x_i, u) = 0`` for a fixed applied control ``u``."""
    u = np.asarray(u, dtype=float)
    n = model.n_states

    def error(this, values, H):
        xi = values.atVector(this.keys()[0])
        xj = values.atVector(this.keys()[1])
        if H is not None:
            dfdx, _ = model.jacobians(xi, u)
            H[0] = -dfdx
            H[1] = np.eye(n)
        return xj - model.step(xi, u)

    return gtsam.CustomFactor(noise, [kx, kx_next], error)


def state_cost(key, target: np.ndarray, noise, wrap_index: int | None = 2):
    """Quadratic cost on the tracking error ``x - target``.

    If ``wrap_index`` is given, that component (a heading) is compared modulo
    ``2*pi`` so the cost sees the shortest angular error.
    """
    target = np.asarray(target, dtype=float)
    n = target.shape[0]

    def error(this, values, H):
        e = values.atVector(this.keys()[0]) - target
        if wrap_index is not None:
            e[wrap_index] = wrap_angle(e[wrap_index])
        if H is not None:
            H[0] = np.eye(n)
        return e

    return gtsam.CustomFactor(noise, [key], error)


def zero_cost(key, dim: int, noise):
    """Quadratic regularization of a variable toward zero (e.g. control effort)."""
    def error(this, values, H):
        if H is not None:
            H[0] = np.eye(dim)
        return values.atVector(this.keys()[0])

    return gtsam.CustomFactor(noise, [key], error)


def position_measurement(key, z: np.ndarray, noise, dim: int = 4):
    """Measure the position ``x[:2]`` (residual ``x[:2] - z``); e.g. GPS."""
    z = np.asarray(z, dtype=float)

    def error(this, values, H):
        x = values.atVector(this.keys()[0])
        if H is not None:
            J = np.zeros((2, dim))
            J[0, 0] = 1.0
            J[1, 1] = 1.0
            H[0] = J
        return x[:2] - z

    return gtsam.CustomFactor(noise, [key], error)


def scalar_measurement(key, value: float, noise, index: int = 3, dim: int = 4):
    """Measure one state component ``x[index]`` (e.g. wheel speed at index 3)."""
    def error(this, values, H):
        x = values.atVector(this.keys()[0])
        if H is not None:
            J = np.zeros((1, dim))
            J[0, index] = 1.0
            H[0] = J
        return np.array([x[index] - value])

    return gtsam.CustomFactor(noise, [key], error)
