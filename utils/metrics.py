import numpy as np

def translation_invariant_error(theta_est, theta_true):
    """
    Calcule :
        min_a || T_a(theta_est) - theta_true ||

    où T_a est une translation circulaire
    """

    p = len(theta_true)
    errors = []

    for shift in range(p):
        shifted = np.roll(theta_est, shift)
        err = np.linalg.norm(shifted - theta_true)
        errors.append(err)

    return min(errors)
