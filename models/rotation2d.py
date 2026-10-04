import numpy as np
from core.base import InverseProblemWithLatent


class RotationProjection2D(InverseProblemWithLatent):

    def __init__(self, sigma, angles, X_grid, Y_grid, x_1d, quad_weights=None):
        super().__init__(sigma, angles, quad_weights=quad_weights)

        self.X = np.asarray(X_grid, dtype=float)
        self.Y = np.asarray(Y_grid, dtype=float)
        self.x_1d = np.asarray(x_1d, dtype=float)
        self.p = self.x_1d.size
        if (
            self.x_1d.ndim != 1
            or self.p < 2
            or not np.all(np.isfinite(self.x_1d))
            or np.any(np.diff(self.x_1d) <= 0)
        ):
            raise ValueError("x_1d must be a finite, strictly increasing grid")
        if self.X.shape != (self.p, self.p) or self.Y.shape != self.X.shape:
            raise ValueError("X_grid and Y_grid must match the square x_1d grid")
        expected_x, expected_y = np.meshgrid(self.x_1d, self.x_1d)
        if not np.allclose(self.X, expected_x) or not np.allclose(self.Y, expected_y):
            raise ValueError("X_grid and Y_grid must use meshgrid(x_1d, x_1d)")

        self.shape_2d = self.X.shape
        self.mask = self.X**2 + self.Y**2 <= 1
        self.x_flat = self.X.flatten()
        self.y_flat = self.Y.flatten()
        self.projection_weights = np.empty(self.p)
        self.projection_weights[0] = (self.x_1d[1] - self.x_1d[0]) / 2
        self.projection_weights[-1] = (self.x_1d[-1] - self.x_1d[-2]) / 2
        if self.p > 2:
            self.projection_weights[1:-1] = (
                self.x_1d[2:] - self.x_1d[:-2]
            ) / 2

    # ===============================
    # ROTATION
    # ===============================
    def _rotation_stencil(self, angle):
        cosine = np.cos(angle)
        sine = np.sin(angle)
        x_source = cosine * self.x_flat + sine * self.y_flat
        y_source = -sine * self.x_flat + cosine * self.y_flat
        inside = (
            (x_source >= self.x_1d[0])
            & (x_source <= self.x_1d[-1])
            & (y_source >= self.x_1d[0])
            & (y_source <= self.x_1d[-1])
        )

        row_upper = np.clip(np.searchsorted(self.x_1d, y_source), 1, self.p - 1)
        col_upper = np.clip(np.searchsorted(self.x_1d, x_source), 1, self.p - 1)
        row_lower = row_upper - 1
        col_lower = col_upper - 1
        row_fraction = (
            (y_source - self.x_1d[row_lower])
            / (self.x_1d[row_upper] - self.x_1d[row_lower])
        )
        col_fraction = (
            (x_source - self.x_1d[col_lower])
            / (self.x_1d[col_upper] - self.x_1d[col_lower])
        )
        row_fraction = np.clip(row_fraction, 0, 1)
        col_fraction = np.clip(col_fraction, 0, 1)
        valid = inside.astype(float)
        weights = (
            (1 - row_fraction) * (1 - col_fraction) * valid,
            (1 - row_fraction) * col_fraction * valid,
            row_fraction * (1 - col_fraction) * valid,
            row_fraction * col_fraction * valid,
        )
        return row_lower, row_upper, col_lower, col_upper, weights

    def _rotate_array(self, array_2d, angle):
        array_2d = np.asarray(array_2d, dtype=float)
        if array_2d.shape != self.shape_2d:
            raise ValueError("array_2d must match the model image shape")

        row_lower, row_upper, col_lower, col_upper, weights = (
            self._rotation_stencil(angle)
        )
        w00, w01, w10, w11 = weights
        rotated = (
            w00 * array_2d[row_lower, col_lower]
            + w01 * array_2d[row_lower, col_upper]
            + w10 * array_2d[row_upper, col_lower]
            + w11 * array_2d[row_upper, col_upper]
        )
        return rotated.reshape(self.shape_2d)

    def _rotate_adjoint_batch(self, arrays_2d, angle):
        arrays_2d = np.asarray(arrays_2d, dtype=float)
        if arrays_2d.ndim != 3 or arrays_2d.shape[1:] != self.shape_2d:
            raise ValueError("arrays_2d must have shape (n, p, p)")
        row_lower, row_upper, col_lower, col_upper, weights = (
            self._rotation_stencil(angle)
        )
        w00, w01, w10, w11 = weights
        result = np.zeros((arrays_2d.shape[0], self.p * self.p))
        values = arrays_2d.reshape(arrays_2d.shape[0], -1)
        batch_indices = np.arange(arrays_2d.shape[0])[:, None]
        np.add.at(
            result,
            (batch_indices, (row_lower * self.p + col_lower)[None, :]),
            values * w00[None, :],
        )
        np.add.at(
            result,
            (batch_indices, (row_lower * self.p + col_upper)[None, :]),
            values * w01[None, :],
        )
        np.add.at(
            result,
            (batch_indices, (row_upper * self.p + col_lower)[None, :]),
            values * w10[None, :],
        )
        np.add.at(
            result,
            (batch_indices, (row_upper * self.p + col_upper)[None, :]),
            values * w11[None, :],
        )
        return result.reshape((arrays_2d.shape[0],) + self.shape_2d)

    def _rotate_adjoint(self, array_2d, angle):
        array_2d = np.asarray(array_2d, dtype=float)
        if array_2d.shape != self.shape_2d:
            raise ValueError("array_2d must match the model image shape")
        return self._rotate_adjoint_batch(array_2d[None, :, :], angle)[0]

    # ===============================
    # PROJECTION
    # ===============================
    def _project(self, array_2d):
        """
        Projection selon y :
        ∫ θ(x,y) dy
        """
        return self.projection_weights @ array_2d

    # ===============================
    # OPÉRATEUR DIRECT
    # ===============================
    def apply_operator(self, theta_2d, angle):
        theta_2d = np.asarray(theta_2d, dtype=float)
        if theta_2d.shape != self.shape_2d:
            raise ValueError("theta_2d must match the model image shape")
        theta_2d = np.where(self.mask, theta_2d, 0)
        rotated = self._rotate_array(theta_2d, angle)
        return self._project(rotated)

    # ===============================
    # ADJOINT
    # ===============================
    def apply_adjoint(self, y_1d, angle):
        """
        y_1d peut être :
        - (p,)     =>  1 signal
        - (n, p)    => batch de signaux
        """

        y_1d = np.asarray(y_1d, dtype=float)
        if y_1d.ndim == 2:
            if y_1d.shape[1] != self.p:
                raise ValueError("batched projections must have shape (n, p)")
            backprojected = (
                self.projection_weights[None, :, None] * y_1d[:, None, :]
            )
            adjoint = self._rotate_adjoint_batch(backprojected, angle)
            return np.where(self.mask[None, :, :], adjoint, 0)
        if y_1d.shape != (self.p,):
            raise ValueError("projection must have shape (p,)")

        backprojected = self.projection_weights[:, None] * y_1d[None, :]
        adjoint = self._rotate_adjoint(backprojected, angle)
        return np.where(self.mask, adjoint, 0)

    def initialize_theta(self, Y):
        Y = self._validate_observations(Y)
        if Y.shape[1] != self.p:
            raise ValueError("Y's observation dimension must match the projection size")
        estimate = np.zeros(self.shape_2d)
        for weight, angle in zip(self.quad_weights, self.latent_grid):
            estimate += weight * np.mean(self.apply_adjoint(Y, angle), axis=0)
        return estimate
    # ===============================
    # CONTRAINTE
    # ===============================
    def enforce_constraints(self, theta_2d):
        theta_2d = theta_2d.copy()
        theta_2d[~self.mask] = 0
        return theta_2d
