import numpy as np


class GradientAscent:
    def __init__(self, lr=0.1):
        self.lr = lr

    def step(self, theta, Q_fn, grad_fn):
        grad = grad_fn(theta)
        return theta + self.lr * grad


class ArmijoOptimizer:
    def __init__(self, eta0=1.0, beta=0.5, c=1e-4, max_iter=50):
        if eta0 <= 0 or not 0 < beta < 1 or not 0 < c < 1 or max_iter <= 0:
            raise ValueError("invalid Armijo line-search parameters")
        self.eta0 = eta0
        self.beta = beta
        self.c = c
        self.max_iter = max_iter

    def step(self, theta, Q_fn, grad_fn):
        grad = grad_fn(theta)
        if not np.all(np.isfinite(grad)):
            raise ValueError("Armijo gradient must be finite")
        grad_norm_squared = np.sum(grad**2)
        if grad_norm_squared <= np.finfo(float).eps:
            return theta.copy()

        eta = self.eta0
        Q_current = Q_fn(theta)
        if not np.isfinite(Q_current):
            raise ValueError("Armijo objective must be finite")

        for _ in range(self.max_iter):
            theta_new = theta + eta * grad
            Q_new = Q_fn(theta_new)

            if Q_new >= Q_current + self.c * eta * grad_norm_squared:
                return theta_new

            eta *= self.beta

        raise RuntimeError("Armijo line search failed to find an improving step")