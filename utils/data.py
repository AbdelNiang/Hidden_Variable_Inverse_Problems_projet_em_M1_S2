import numpy as np

def generate_data(model, theta_true, n_samples, seed=0):
    """
    Génère Y_i = A_{Z_i} θ + bruit

    Returns:
        Y: observations with shape (n_samples, observation_size)
        Z: sampled latent values with shape (n_samples,)
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(seed)

    Y_list = []
    Z_list = []

    for _ in range(n_samples):
        z = rng.choice(model.latent_grid, p=model.quad_weights)
        y = model.apply_operator(theta_true, z)
        noise = model.sigma * rng.standard_normal(size=y.shape)
        Y_list.append(y + noise)
        Z_list.append(z)

    return np.array(Y_list), np.array(Z_list)