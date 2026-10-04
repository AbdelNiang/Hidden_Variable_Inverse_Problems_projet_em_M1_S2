import importlib

import numpy as np
import pytest

from core.base import InverseProblemWithLatent
from core.em import em
from core.em_reg import em_regularized
from models.rotation2d import RotationProjection2D
from models.translation1d import Translation1D
from utils.data import generate_data
from utils.optim import ArmijoOptimizer


def make_rotation_model(size=7):
    axis = np.linspace(-1, 1, size)
    grid_x, grid_y = np.meshgrid(axis, axis)
    return RotationProjection2D(
        sigma=0.2,
        angles=np.array([0.0, 0.37]),
        X_grid=grid_x,
        Y_grid=grid_y,
        x_1d=axis,
    )


def test_translation_adjoint_rolls_each_batch_row_independently():
    model = Translation1D(0.2, np.array([1]), p=3)
    signals = np.array([[1, 2, 3], [10, 20, 30]])

    np.testing.assert_array_equal(
        model.apply_adjoint(signals, 1),
        np.roll(signals, -1, axis=-1),
    )


@pytest.mark.parametrize(
    "make_model",
    [lambda: Translation1D(0.4, np.array([-1, 0, 1]), p=4), make_rotation_model],
    ids=["translation", "rotation"],
)
def test_fixed_weight_gradient_matches_finite_differences(make_model):
    model = make_model()
    parameter_shape = (model.p,) if isinstance(model, Translation1D) else model.shape_2d
    rng = np.random.default_rng(4)
    theta = rng.normal(size=parameter_shape)
    observations = rng.normal(size=(5, model.p))
    weights = model.compute_weights(observations, theta)
    direction = rng.normal(size=parameter_shape)
    epsilon = 1e-6

    finite_difference = (
        model.Q(observations, theta + epsilon * direction, weights)
        - model.Q(observations, theta - epsilon * direction, weights)
    ) / (2 * epsilon)
    directional_gradient = np.sum(
        model.gradient_Q(observations, theta, weights) * direction
    )

    assert finite_difference == pytest.approx(directional_gradient, rel=1e-5)


def test_rotation_at_zero_preserves_asymmetric_image():
    model = make_rotation_model()
    image = np.arange(49, dtype=float).reshape(7, 7)

    np.testing.assert_allclose(model._rotate_array(image, 0.0), image)


def test_rotation_projection_adjoint_matches_inner_product():
    model = make_rotation_model()
    rng = np.random.default_rng(5)
    image = rng.normal(size=model.shape_2d)
    projection = rng.normal(size=model.p)
    angle = 0.37

    left = np.vdot(model.apply_operator(image, angle), projection)
    right = np.vdot(image, model.apply_adjoint(projection, angle))

    assert left == pytest.approx(right, rel=1e-10, abs=1e-10)


def test_explicit_m_step_supports_image_parameters():
    model = make_rotation_model(size=3)
    observations = np.ones((2, model.p))
    weights = np.full((2, model.n_latent), 1 / model.n_latent)

    estimate = model.solve_m_step(observations, weights, lambda_reg=0.1)

    assert estimate.shape == model.shape_2d
    assert np.all(np.isfinite(estimate))


def test_isometric_m_step_uses_regularization():
    model = Translation1D(0.5, np.array([0]), p=3)
    observations = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    weights = np.ones((2, 1))

    unregularized = model.solve_m_step(observations, weights, lambda_reg=0)
    regularized = model.solve_m_step(observations, weights, lambda_reg=2)

    assert np.linalg.norm(regularized) < np.linalg.norm(unregularized)


def test_em_accepts_image_shaped_initialization_from_model():
    model = make_rotation_model(size=3)
    observations = np.ones((2, model.p))
    initial = model.initialize_theta(observations)

    assert initial.shape == model.shape_2d
    estimate, history, _ = em(
        model,
        observations,
        theta_init=initial,
        n_iter=1,
        optimizer=ArmijoOptimizer(max_iter=10),
        verbose=False,
    )
    assert estimate.shape == model.shape_2d
    assert len(history) == 2
    assert np.all(estimate[~model.mask] == 0)


def test_data_generation_uses_local_rng_and_latent_probabilities():
    model = Translation1D(
        1e-9, np.array([0, 1]), p=3, quad_weights=np.array([0.1, 0.9])
    )
    np.random.seed(17)
    expected_global_value = np.random.random()
    np.random.seed(17)

    _, latent = generate_data(model, np.array([1.0, 0.0, 0.0]), 1000, seed=8)
    actual_global_value = np.random.random()

    assert actual_global_value == expected_global_value
    assert np.mean(latent == 1) == pytest.approx(0.9, abs=0.04)


def test_armijo_reports_line_search_failure():
    optimizer = ArmijoOptimizer(max_iter=3)
    theta = np.zeros(2)

    with pytest.raises(RuntimeError, match="Armijo"):
        optimizer.step(theta, lambda value: 0.0, lambda value: np.ones_like(value))


def test_regularized_armijo_objective_matches_gradient():
    class GradientCheckingOptimizer:
        def step(self, theta, objective, gradient):
            direction = np.arange(1, theta.size + 1, dtype=float).reshape(theta.shape)
            epsilon = 1e-6
            finite_difference = (
                objective(theta + epsilon * direction)
                - objective(theta - epsilon * direction)
            ) / (2 * epsilon)
            directional_gradient = np.sum(gradient(theta) * direction)
            assert finite_difference == pytest.approx(directional_gradient, rel=1e-5)
            return theta.copy()

    model = Translation1D(0.4, np.array([-1, 0, 1]), p=4)
    observations = np.array([[0.0, 1.0, 0.0, -1.0], [1.0, 0.0, -1.0, 0.0]])
    estimate, _, _ = em_regularized(
        model,
        observations,
        lambda_reg=0.7,
        theta_init=np.array([0.1, 0.2, 0.3, 0.4]),
        n_iter=1,
        optimizer=GradientCheckingOptimizer(),
        verbose=False,
    )

    assert estimate.shape == (4,)


def test_e_step_respects_nonuniform_latent_prior():
    model = Translation1D(
        0.2, np.array([0, 1]), p=3, quad_weights=np.array([1.0, 3.0])
    )
    observations = np.zeros((2, 3))

    weights = model.compute_weights(observations, np.zeros(3))

    np.testing.assert_allclose(weights, np.tile([0.25, 0.75], (2, 1)))


def test_core_and_demo_modules_import():
    for module_name in (
        "demo",
        "core.em_reg",
        "utils.optim",
        "utils.metrics",
        "utils.viz",
    ):
        importlib.import_module(module_name)


def test_base_rejects_invalid_noise_scale_and_latent_weights():
    class MinimalModel(InverseProblemWithLatent):
        def apply_operator(self, theta, z):
            return theta

        def apply_adjoint(self, y, z):
            return y

    with pytest.raises(ValueError, match="sigma"):
        MinimalModel(0, np.array([0]))
    with pytest.raises(ValueError, match="quad_weights"):
        MinimalModel(1, np.array([0, 1]), quad_weights=np.array([1.0, 0.0]))