import numpy as np


def em(model, Y, theta_init=None, n_iter=50, optimizer=None, verbose=True):
    if n_iter < 0:
        raise ValueError("n_iter must be non-negative")
    theta = (
        model.initialize_theta(Y)
        if theta_init is None
        else np.asarray(theta_init).copy()
    )

    history = [theta.copy()]
    Q_history = []

    for iteration in range(n_iter):
        weights = model.compute_weights(Y, theta)
        Q_val = model.Q(Y, theta, weights)
        Q_history.append(Q_val)

        if optimizer is None:
            theta = model.solve_m_step(Y, weights)
        else:
            Q_fn = lambda candidate: model.Q(Y, candidate, weights)
            grad_fn = lambda candidate: model.gradient_Q(Y, candidate, weights)
            theta = optimizer.step(theta, Q_fn, grad_fn)

        history.append(theta.copy())

        if verbose and iteration % 10 == 0:
            print(f"Iteration {iteration} | Q = {Q_val:.4f}")

    return theta, history, Q_history