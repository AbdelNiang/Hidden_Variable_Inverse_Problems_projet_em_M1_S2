from abc import ABC, abstractmethod
import numpy as np


class InverseProblemWithLatent(ABC):

    def __init__(self, sigma, latent_grid, quad_weights=None):
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be finite and positive")

        self.sigma = float(sigma)
        self.latent_grid = np.asarray(latent_grid)
        if self.latent_grid.ndim != 1 or self.latent_grid.size == 0:
            raise ValueError("latent_grid must be a non-empty one-dimensional array")
        self.n_latent = len(self.latent_grid)

        if quad_weights is None:
            quad_weights = np.ones(self.n_latent)
        self.quad_weights = np.asarray(quad_weights, dtype=float)
        if (
            self.quad_weights.shape != (self.n_latent,)
            or not np.all(np.isfinite(self.quad_weights))
            or np.any(self.quad_weights <= 0)
        ):
            raise ValueError("quad_weights must be finite, positive, and match latent_grid")
        self.quad_weights /= np.sum(self.quad_weights)

    # ===============================
    # Opérateurs
    # ===============================

    @abstractmethod
    def apply_operator(self, theta, z):
        pass

    @abstractmethod
    def apply_adjoint(self, y, z):
        pass

    # ===============================
    # E-step
    # ===============================
    def initialize_theta(self, Y):
        Y = self._validate_observations(Y)
        return np.mean(Y, axis=0)

    @staticmethod
    def _validate_observations(Y):
        Y = np.asarray(Y, dtype=float)
        if Y.ndim != 2 or Y.shape[0] == 0:
            raise ValueError("Y must be a non-empty two-dimensional array")
        if not np.all(np.isfinite(Y)):
            raise ValueError("Y must contain only finite values")
        return Y

    def _predictions(self, theta, observation_shape):
        predictions = np.stack([
            self.apply_operator(theta, z) for z in self.latent_grid
        ])
        if predictions.ndim != 2 or predictions.shape[1:] != observation_shape:
            raise ValueError("operator output shape must match observation shape")
        return predictions

    def _validate_weights(self, weights, n_observations):
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (n_observations, self.n_latent):
            raise ValueError("weights must have shape (n_observations, n_latent)")
        if (
            not np.all(np.isfinite(weights))
            or np.any(weights < 0)
            or not np.allclose(np.sum(weights, axis=1), 1.0)
        ):
            raise ValueError("each row of weights must be a probability distribution")
        return weights

    def compute_weights(self, Y, theta):
        Y = self._validate_observations(Y)
        predictions = self._predictions(theta, Y.shape[1:])
        residuals = Y[:, None, :] - predictions[None, :, :]
        losses = np.sum(residuals**2, axis=2)
        log_weights = (
            -losses / (2 * self.sigma**2)
            + np.log(self.quad_weights)[None, :]
        )
        log_weights -= np.max(log_weights, axis=1, keepdims=True)
        weights = np.exp(log_weights)
        return weights / np.sum(weights, axis=1, keepdims=True)

    def Q(self, Y, theta, weights):
        Y = self._validate_observations(Y)
        weights = self._validate_weights(weights, Y.shape[0])
        predictions = self._predictions(theta, Y.shape[1:])
        residuals = Y[:, None, :] - predictions[None, :, :]
        losses = np.sum(residuals**2, axis=2)
        return -np.sum(weights * losses) / (2 * self.sigma**2)

    def gradient_Q(self, Y, theta, weights):
        Y = self._validate_observations(Y)
        weights = self._validate_weights(weights, Y.shape[0])
        theta = np.asarray(theta)
        predictions = self._predictions(theta, Y.shape[1:])
        residuals = Y[:, None, :] - predictions[None, :, :]
        gradient = np.zeros_like(theta, dtype=float)

        for latent_index, latent_value in enumerate(self.latent_grid):
            adjoint_residuals = self.apply_adjoint(
                residuals[:, latent_index, :], latent_value
            )
            expected_shape = (Y.shape[0],) + theta.shape
            if adjoint_residuals.shape != expected_shape:
                raise ValueError("batched adjoint output must match theta shape")
            weight_shape = (Y.shape[0],) + (1,) * theta.ndim
            gradient += np.sum(
                weights[:, latent_index].reshape(weight_shape) * adjoint_residuals,
                axis=0,
            )

        return gradient / self.sigma**2

    def solve_m_step(self, Y, weights, lambda_reg=0.0):
        Y = self._validate_observations(Y)
        weights = self._validate_weights(weights, Y.shape[0])
        if not np.isfinite(lambda_reg) or lambda_reg < 0:
            raise ValueError("lambda_reg must be finite and non-negative")

        n_observations = Y.shape[0]
        theta_shape = self.apply_adjoint(Y[:1], self.latent_grid[0]).shape[1:]
        if not theta_shape:
            raise ValueError("adjoint must return a batched parameter array")
        parameter_size = int(np.prod(theta_shape))
        right_hand_side = np.zeros(theta_shape, dtype=float)

        for latent_index, latent_value in enumerate(self.latent_grid):
            adjoint_observations = self.apply_adjoint(Y, latent_value)
            weight_shape = (n_observations,) + (1,) * len(theta_shape)
            right_hand_side += np.sum(
                weights[:, latent_index].reshape(weight_shape) * adjoint_observations,
                axis=0,
            )

        regularization = self.sigma**2 * lambda_reg
        if getattr(self, "is_isometry", False):
            return right_hand_side / (n_observations + regularization)

        normal_matrix = np.zeros((parameter_size, parameter_size), dtype=float)
        for latent_index, latent_value in enumerate(self.latent_grid):
            latent_mass = np.sum(weights[:, latent_index])
            for parameter_index in range(parameter_size):
                basis = np.zeros(theta_shape, dtype=float)
                basis.flat[parameter_index] = 1.0
                forward_basis = self.apply_operator(basis, latent_value)
                normal_column = self.apply_adjoint(forward_basis, latent_value)
                normal_matrix[:, parameter_index] += latent_mass * normal_column.ravel()

        normal_matrix += regularization * np.eye(parameter_size)
        try:
            solution = np.linalg.solve(normal_matrix, right_hand_side.ravel())
        except np.linalg.LinAlgError:
            solution = np.linalg.pinv(normal_matrix) @ right_hand_side.ravel()
        return solution.reshape(theta_shape)