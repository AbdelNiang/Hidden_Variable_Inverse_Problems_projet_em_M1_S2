import numpy as np


def em_regularized(
    model,
    Y,
    lambda_reg=1e-2,
    theta_init=None,
    n_iter=50,
    optimizer=None,
    verbose=True,
):
    if lambda_reg < 0 or not np.isfinite(lambda_reg):
        raise ValueError("lambda_reg must be finite and non-negative")
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
        Q_fn = lambda candidate: (
            model.Q(Y, candidate, weights)
            - 0.5 * lambda_reg * np.sum(candidate**2)
        )
        Q_val = Q_fn(theta)
        Q_history.append(Q_val)

        if optimizer is None:
            theta = model.solve_m_step(Y, weights, lambda_reg=lambda_reg)
        else:
            grad_fn = lambda candidate: (
                model.gradient_Q(Y, candidate, weights) - lambda_reg * candidate
            )
            theta = optimizer.step(theta, Q_fn, grad_fn)

        history.append(theta.copy())
        if verbose and iteration % 10 == 0:
            print(f"Iteration {iteration} | Q = {Q_val:.4f}")

    return theta, history, Q_history