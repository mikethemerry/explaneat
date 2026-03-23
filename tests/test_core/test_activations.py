"""Tests for the shared activation function registry."""

import numpy as np
import pytest


# =============================================================================
# Expected activation names (matching neat-python's ActivationFunctionSet)
# =============================================================================

EXPECTED_ACTIVATIONS = {
    "sigmoid",
    "tanh",
    "sin",
    "gauss",
    "relu",
    "softplus",
    "identity",
    "clamped",
    "inv",
    "log",
    "exp",
    "abs",
    "hat",
    "square",
    "cube",
}


class TestRegistryCompleteness:
    """All 15 neat-python activations must be registered."""

    def test_all_activations_registered(self):
        from explaneat.core.activations import ACTIVATIONS

        assert ACTIVATIONS == EXPECTED_ACTIVATIONS

    def test_activations_is_frozenset(self):
        from explaneat.core.activations import ACTIVATIONS

        # Should be immutable so callers cannot accidentally modify it
        assert isinstance(ACTIVATIONS, frozenset)

    def test_numpy_available_for_all(self):
        from explaneat.core.activations import get_numpy_activation

        for name in EXPECTED_ACTIVATIONS:
            fn = get_numpy_activation(name)
            assert callable(fn), f"get_numpy_activation({name!r}) not callable"

    def test_sympy_available_for_all(self):
        from explaneat.core.activations import get_sympy_activation

        for name in EXPECTED_ACTIVATIONS:
            fn = get_sympy_activation(name)
            assert callable(fn), f"get_sympy_activation({name!r}) not callable"


class TestUnknownActivation:
    """KeyError for unknown activation names."""

    def test_numpy_unknown_raises_keyerror(self):
        from explaneat.core.activations import get_numpy_activation

        with pytest.raises(KeyError, match="banana"):
            get_numpy_activation("banana")

    def test_sympy_unknown_raises_keyerror(self):
        from explaneat.core.activations import get_sympy_activation

        with pytest.raises(KeyError, match="banana"):
            get_sympy_activation("banana")


class TestNumpySigmoid:
    """Verify sigmoid numpy implementation."""

    def test_sigmoid_zero(self):
        from explaneat.core.activations import get_numpy_activation

        fn = get_numpy_activation("sigmoid")
        result = fn(np.array([0.0]))
        np.testing.assert_allclose(result, [0.5], atol=1e-7)

    def test_sigmoid_large_positive(self):
        from explaneat.core.activations import get_numpy_activation

        fn = get_numpy_activation("sigmoid")
        result = fn(np.array([100.0]))
        np.testing.assert_allclose(result, [1.0], atol=1e-7)

    def test_sigmoid_large_negative(self):
        from explaneat.core.activations import get_numpy_activation

        fn = get_numpy_activation("sigmoid")
        result = fn(np.array([-100.0]))
        np.testing.assert_allclose(result, [0.0], atol=1e-7)

    def test_sigmoid_vectorized(self):
        from explaneat.core.activations import get_numpy_activation

        fn = get_numpy_activation("sigmoid")
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = fn(x)
        expected = 1.0 / (1.0 + np.exp(-x))
        np.testing.assert_allclose(result, expected, atol=1e-7)


class TestNumpyRelu:
    """Verify relu numpy implementation."""

    def test_relu_positive(self):
        from explaneat.core.activations import get_numpy_activation

        fn = get_numpy_activation("relu")
        result = fn(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_relu_negative(self):
        from explaneat.core.activations import get_numpy_activation

        fn = get_numpy_activation("relu")
        result = fn(np.array([-1.0, -2.0, -3.0]))
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_relu_mixed(self):
        from explaneat.core.activations import get_numpy_activation

        fn = get_numpy_activation("relu")
        result = fn(np.array([-1.0, 0.0, 1.0]))
        np.testing.assert_array_equal(result, [0.0, 0.0, 1.0])


class TestNumpyIdentity:
    """Verify identity numpy implementation."""

    def test_identity_passthrough(self):
        from explaneat.core.activations import get_numpy_activation

        fn = get_numpy_activation("identity")
        x = np.array([-2.0, 0.0, 3.14])
        result = fn(x)
        np.testing.assert_array_equal(result, x)


class TestNumpyTanh:
    """Verify tanh numpy implementation."""

    def test_tanh_zero(self):
        from explaneat.core.activations import get_numpy_activation

        fn = get_numpy_activation("tanh")
        result = fn(np.array([0.0]))
        np.testing.assert_allclose(result, [0.0], atol=1e-7)

    def test_tanh_large_positive(self):
        from explaneat.core.activations import get_numpy_activation

        fn = get_numpy_activation("tanh")
        result = fn(np.array([100.0]))
        np.testing.assert_allclose(result, [1.0], atol=1e-7)

    def test_tanh_vectorized(self):
        from explaneat.core.activations import get_numpy_activation

        fn = get_numpy_activation("tanh")
        x = np.array([-1.0, 0.0, 1.0])
        result = fn(x)
        expected = np.tanh(x)
        np.testing.assert_allclose(result, expected, atol=1e-7)


class TestNumpyOtherActivations:
    """Verify remaining numpy activation functions."""

    def test_sin(self):
        from explaneat.core.activations import get_numpy_activation

        fn = get_numpy_activation("sin")
        x = np.array([0.0, np.pi / 2, np.pi])
        result = fn(x)
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-7)

    def test_gauss(self):
        from explaneat.core.activations import get_numpy_activation

        fn = get_numpy_activation("gauss")
        result = fn(np.array([0.0]))
        np.testing.assert_allclose(result, [1.0], atol=1e-7)
        # Gauss drops off away from zero
        result_far = fn(np.array([5.0]))
        assert result_far[0] < 0.01

    def test_softplus(self):
        from explaneat.core.activations import get_numpy_activation

        fn = get_numpy_activation("softplus")
        result = fn(np.array([0.0]))
        np.testing.assert_allclose(result, [np.log(2.0)], atol=1e-7)
        # softplus is always positive
        result_neg = fn(np.array([-100.0]))
        assert result_neg[0] >= 0.0

    def test_clamped(self):
        from explaneat.core.activations import get_numpy_activation

        fn = get_numpy_activation("clamped")
        x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        result = fn(x)
        np.testing.assert_array_equal(result, [-1.0, -0.5, 0.0, 0.5, 1.0])

    def test_inv_nonzero(self):
        from explaneat.core.activations import get_numpy_activation

        fn = get_numpy_activation("inv")
        result = fn(np.array([2.0, 4.0]))
        np.testing.assert_allclose(result, [0.5, 0.25], atol=1e-7)

    def test_inv_near_zero(self):
        from explaneat.core.activations import get_numpy_activation

        fn = get_numpy_activation("inv")
        result = fn(np.array([0.0, 1e-10]))
        # Should return 0 for values near zero
        assert result[0] == 0.0

    def test_log_positive(self):
        from explaneat.core.activations import get_numpy_activation

        fn = get_numpy_activation("log")
        result = fn(np.array([1.0, np.e]))
        np.testing.assert_allclose(result, [0.0, 1.0], atol=1e-7)

    def test_log_nonpositive(self):
        from explaneat.core.activations import get_numpy_activation

        fn = get_numpy_activation("log")
        result = fn(np.array([0.0, -1.0]))
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_exp(self):
        from explaneat.core.activations import get_numpy_activation

        fn = get_numpy_activation("exp")
        result = fn(np.array([0.0, 1.0]))
        np.testing.assert_allclose(result, [1.0, np.e], atol=1e-7)

    def test_exp_large_clipped(self):
        from explaneat.core.activations import get_numpy_activation

        fn = get_numpy_activation("exp")
        # Very large input should not overflow
        result = fn(np.array([1000.0]))
        assert np.isfinite(result[0])

    def test_abs(self):
        from explaneat.core.activations import get_numpy_activation

        fn = get_numpy_activation("abs")
        result = fn(np.array([-3.0, 0.0, 3.0]))
        np.testing.assert_array_equal(result, [3.0, 0.0, 3.0])

    def test_hat(self):
        from explaneat.core.activations import get_numpy_activation

        fn = get_numpy_activation("hat")
        x = np.array([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        result = fn(x)
        # Peak at x=1, zero outside [0, 2]
        expected = [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0]
        np.testing.assert_allclose(result, expected, atol=1e-7)

    def test_square(self):
        from explaneat.core.activations import get_numpy_activation

        fn = get_numpy_activation("square")
        result = fn(np.array([-2.0, 0.0, 3.0]))
        np.testing.assert_array_equal(result, [4.0, 0.0, 9.0])

    def test_cube(self):
        from explaneat.core.activations import get_numpy_activation

        fn = get_numpy_activation("cube")
        result = fn(np.array([-2.0, 0.0, 3.0]))
        np.testing.assert_array_equal(result, [-8.0, 0.0, 27.0])


class TestSympyActivations:
    """Verify sympy activation functions produce correct symbolic expressions."""

    def test_sympy_sigmoid(self):
        import sympy

        from explaneat.core.activations import get_sympy_activation

        fn = get_sympy_activation("sigmoid")
        x = sympy.Symbol("x")
        result = fn(x)
        # Should simplify to 1/(1+exp(-x))
        val_at_zero = float(result.subs(x, 0))
        assert abs(val_at_zero - 0.5) < 1e-7

    def test_sympy_relu(self):
        import sympy

        from explaneat.core.activations import get_sympy_activation

        fn = get_sympy_activation("relu")
        x = sympy.Symbol("x")
        result = fn(x)
        # relu is a named Function for readable formulas
        assert "relu" in str(result)
        assert result.args == (x,)

    def test_sympy_identity(self):
        import sympy

        from explaneat.core.activations import get_sympy_activation

        fn = get_sympy_activation("identity")
        x = sympy.Symbol("x")
        result = fn(x)
        assert result == x

    def test_sympy_tanh(self):
        import sympy

        from explaneat.core.activations import get_sympy_activation

        fn = get_sympy_activation("tanh")
        x = sympy.Symbol("x")
        result = fn(x)
        val_at_zero = float(result.subs(x, 0))
        assert abs(val_at_zero) < 1e-7

    def test_sympy_square(self):
        import sympy

        from explaneat.core.activations import get_sympy_activation

        fn = get_sympy_activation("square")
        x = sympy.Symbol("x")
        result = fn(x)
        assert float(result.subs(x, 3)) == 9.0

    def test_sympy_cube(self):
        import sympy

        from explaneat.core.activations import get_sympy_activation

        fn = get_sympy_activation("cube")
        x = sympy.Symbol("x")
        result = fn(x)
        assert float(result.subs(x, -2)) == -8.0

    def test_sympy_exp(self):
        import sympy

        from explaneat.core.activations import get_sympy_activation

        fn = get_sympy_activation("exp")
        x = sympy.Symbol("x")
        result = fn(x)
        val_at_one = float(result.subs(x, 1))
        assert abs(val_at_one - np.e) < 1e-7

    def test_sympy_sin(self):
        import sympy

        from explaneat.core.activations import get_sympy_activation

        fn = get_sympy_activation("sin")
        x = sympy.Symbol("x")
        result = fn(x)
        val = float(result.subs(x, sympy.pi / 2))
        assert abs(val - 1.0) < 1e-7

    def test_sympy_abs(self):
        import sympy

        from explaneat.core.activations import get_sympy_activation

        fn = get_sympy_activation("abs")
        x = sympy.Symbol("x")
        result = fn(x)
        assert float(result.subs(x, -3)) == 3.0

    def test_sympy_gauss(self):
        import sympy

        from explaneat.core.activations import get_sympy_activation

        fn = get_sympy_activation("gauss")
        x = sympy.Symbol("x")
        result = fn(x)
        val_at_zero = float(result.subs(x, 0))
        assert abs(val_at_zero - 1.0) < 1e-7


class TestNumpySympyAgreement:
    """Numpy and sympy implementations should agree for common activations."""

    ACTIVATIONS_TO_CHECK = [
        "sigmoid",
        "tanh",
        "sin",
        # relu excluded: intentionally a named Function (not numerically evaluable)
        "identity",
        "abs",
        "square",
        "cube",
        "exp",
        "softplus",
        "gauss",
    ]

    @pytest.mark.parametrize("name", ACTIVATIONS_TO_CHECK)
    def test_agreement(self, name):
        import sympy

        from explaneat.core.activations import get_numpy_activation, get_sympy_activation

        np_fn = get_numpy_activation(name)
        sp_fn = get_sympy_activation(name)

        x_sym = sympy.Symbol("x")
        expr = sp_fn(x_sym)

        test_values = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
        np_results = np_fn(test_values)

        for val, np_result in zip(test_values, np_results):
            sp_result = float(expr.subs(x_sym, val))
            assert abs(np_result - sp_result) < 1e-6, (
                f"{name}: numpy={np_result}, sympy={sp_result} at x={val}"
            )
