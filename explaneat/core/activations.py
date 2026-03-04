"""Shared activation function registry for numpy and sympy.

Provides a unified mapping from activation name strings (matching neat-python's
ActivationFunctionSet) to numpy and sympy implementations.

Numpy functions: np.ndarray -> np.ndarray
Sympy functions: sympy.Expr -> sympy.Expr (lazy-imported to avoid cost when not needed)

Usage:
    from explaneat.core.activations import get_numpy_activation, get_sympy_activation

    sigmoid_np = get_numpy_activation("sigmoid")
    result = sigmoid_np(np.array([0.0, 1.0]))

    sigmoid_sp = get_sympy_activation("sigmoid")
    import sympy
    x = sympy.Symbol("x")
    expr = sigmoid_sp(x)
"""

from typing import Callable, Dict

import numpy as np


# =============================================================================
# Numpy activation functions
# =============================================================================


def _np_sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def _np_tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _np_sin(x: np.ndarray) -> np.ndarray:
    return np.sin(x)


def _np_gauss(x: np.ndarray) -> np.ndarray:
    return np.exp(-(x ** 2))


def _np_relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _np_softplus(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return np.log1p(np.exp(x))


def _np_identity(x: np.ndarray) -> np.ndarray:
    return x


def _np_clamped(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -1.0, 1.0)


def _np_inv(x: np.ndarray) -> np.ndarray:
    result = np.where(np.abs(x) < 1e-7, 0.0, 1.0 / np.where(np.abs(x) < 1e-7, 1.0, x))
    return result


def _np_log(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, np.log(np.where(x > 0, x, 1.0)), 0.0)


def _np_exp(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return np.exp(x)


def _np_abs(x: np.ndarray) -> np.ndarray:
    return np.abs(x)


def _np_hat(x: np.ndarray) -> np.ndarray:
    """Triangular function peaking at x=1, zero outside [0, 2]."""
    return np.where(
        (x >= 0) & (x <= 1),
        x,
        np.where(
            (x > 1) & (x <= 2),
            2.0 - x,
            0.0,
        ),
    )


def _np_square(x: np.ndarray) -> np.ndarray:
    return x ** 2


def _np_cube(x: np.ndarray) -> np.ndarray:
    return x ** 3


# =============================================================================
# Registry
# =============================================================================

_NUMPY_REGISTRY: Dict[str, Callable] = {
    "sigmoid": _np_sigmoid,
    "tanh": _np_tanh,
    "sin": _np_sin,
    "gauss": _np_gauss,
    "relu": _np_relu,
    "softplus": _np_softplus,
    "identity": _np_identity,
    "clamped": _np_clamped,
    "inv": _np_inv,
    "log": _np_log,
    "exp": _np_exp,
    "abs": _np_abs,
    "hat": _np_hat,
    "square": _np_square,
    "cube": _np_cube,
}

ACTIVATIONS: frozenset = frozenset(_NUMPY_REGISTRY.keys())
"""Set of all registered activation names, for introspection."""


def get_numpy_activation(name: str) -> Callable:
    """Return the numpy activation function for *name*.

    Raises KeyError if the activation name is not registered.
    """
    try:
        return _NUMPY_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown activation {name!r}. "
            f"Available: {sorted(ACTIVATIONS)}"
        )


# =============================================================================
# Sympy activation functions (lazy-loaded)
# =============================================================================

_SYMPY_REGISTRY: Dict[str, Callable] = {}
_sympy_loaded = False


def _load_sympy_registry() -> None:
    """Populate the sympy registry. Called once on first access."""
    global _sympy_loaded
    if _sympy_loaded:
        return

    import sympy

    _SYMPY_REGISTRY["sigmoid"] = lambda x: 1 / (1 + sympy.exp(-x))
    _SYMPY_REGISTRY["tanh"] = lambda x: sympy.tanh(x)
    _SYMPY_REGISTRY["sin"] = lambda x: sympy.sin(x)
    _SYMPY_REGISTRY["gauss"] = lambda x: sympy.exp(-(x ** 2))
    _SYMPY_REGISTRY["relu"] = lambda x: sympy.Max(0, x)
    _SYMPY_REGISTRY["softplus"] = lambda x: sympy.log(1 + sympy.exp(x))
    _SYMPY_REGISTRY["identity"] = lambda x: x
    _SYMPY_REGISTRY["clamped"] = lambda x: sympy.Max(-1, sympy.Min(1, x))
    _SYMPY_REGISTRY["inv"] = lambda x: sympy.Piecewise(
        (0, sympy.Abs(x) < 1e-7), (1 / x, True)
    )
    _SYMPY_REGISTRY["log"] = lambda x: sympy.Piecewise(
        (0, x <= 0), (sympy.log(x), True)
    )
    _SYMPY_REGISTRY["exp"] = lambda x: sympy.exp(x)
    _SYMPY_REGISTRY["abs"] = lambda x: sympy.Abs(x)
    _SYMPY_REGISTRY["hat"] = lambda x: sympy.Piecewise(
        (x, (x >= 0) & (x <= 1)),
        (2 - x, (x > 1) & (x <= 2)),
        (0, True),
    )
    _SYMPY_REGISTRY["square"] = lambda x: x ** 2
    _SYMPY_REGISTRY["cube"] = lambda x: x ** 3

    _sympy_loaded = True


def get_sympy_activation(name: str) -> Callable:
    """Return the sympy activation function for *name*.

    Sympy is lazy-imported on first call to avoid import overhead when only
    numpy activations are needed.

    Raises KeyError if the activation name is not registered.
    """
    if name not in ACTIVATIONS:
        raise KeyError(
            f"Unknown activation {name!r}. "
            f"Available: {sorted(ACTIVATIONS)}"
        )
    _load_sympy_registry()
    return _SYMPY_REGISTRY[name]
