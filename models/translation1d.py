import numpy as np
from core.base import InverseProblemWithLatent

class Translation1D(InverseProblemWithLatent):
    """
    Cas 1 : (A_z θ)_j = θ(x_j - z) avec translation circulaire
    """

    def __init__(self, sigma, shifts, p, quad_weights=None):
        super().__init__(sigma, shifts, quad_weights=quad_weights)
        if p <= 0:
            raise ValueError("p must be positive")
        self.p = int(p)
        self.is_isometry = True

    def apply_operator(self, theta, z):
        if np.shape(theta)[-1] != self.p:
            raise ValueError("theta's last dimension must equal p")
        shift_int = int(np.round(z))
        return np.roll(theta, shift_int, axis=-1)

    def apply_adjoint(self, y, z):
        if np.shape(y)[-1] != self.p:
            raise ValueError("y's last dimension must equal p")
        shift_int = int(np.round(z))
        return np.roll(y, -shift_int, axis=-1)
