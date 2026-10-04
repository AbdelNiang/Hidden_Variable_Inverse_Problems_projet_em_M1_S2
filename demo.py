import numpy as np
import matplotlib.pyplot as plt

# ===== Models =====
from models.translation1d import Translation1D
from models.rotation2d import RotationProjection2D

# ===== EM =====
from core.em import em

# ===== Optim =====
from utils.optim import GradientAscent, ArmijoOptimizer

# ===== Utils =====
from utils.data import generate_data
from utils.metrics import translation_invariant_error
from utils.viz import (
    plot_signals,
    plot_samples_1d,
    plot_images,
    plot_convergence
)


# ============================================================
# CAS 1 : Translation 1D
# ============================================================

def demo_translation():

    print("\n" + "=" * 50)
    print("CAS 1 : Translation 1D")
    print("=" * 50)

    p = 50
    sigma = 0.3
    rng = np.random.default_rng(0)
    shifts = np.arange(-5, 6)

    model = Translation1D(
        sigma=sigma,
        shifts=shifts,
        p=p
    )

    x = np.linspace(0, 1, p)

    # signal gaussien 1D
    theta_true = np.exp(-50 * (x - 0.5) ** 2)

    # génération des données
    Y, Z = generate_data(
        model,
        theta_true,
        n_samples=100
    )

    # affichage des données
    plot_samples_1d(Y)

    # initialisation EM
    theta_init = rng.standard_normal(p)

    # EM
    theta_est, history, Q_history = em(
        model,
        Y,
        theta_init,
        n_iter=10
    )

    # erreur
    err = translation_invariant_error(
        theta_est,
        theta_true
    )

    print(f"Erreur (invariante translation) : {err:.4f}")

    # affichage reconstruction
    plot_signals(theta_true, theta_est)

    # convergence
    plot_convergence(
        Q_history,
        "Convergence de Q (Translation)"
    )


# ============================================================
# Construction mélange gaussiennes 2D asymétriques
# ============================================================

def build_asymmetric_gaussian_mixture(X, Y):

    theta = np.zeros_like(X)

    components = [

        {
            "mu": (-0.35, -0.2),
            "sigma_x": 0.15,
            "sigma_y": 0.08,
            "theta": np.pi / 6,
            "weight": 1.0,
            "skew": 12
        },

        {
            "mu": (0.35, 0.25),
            "sigma_x": 0.25,
            "sigma_y": 0.12,
            "theta": -np.pi / 4,
            "weight": 0.8,
            "skew": -10
        }
    ]

    for c in components:

        mu_x, mu_y = c["mu"]

        sx = c["sigma_x"]
        sy = c["sigma_y"]

        angle = c["theta"]

        w = c["weight"]

        skew_strength = c["skew"]

        # rotation
        ct = np.cos(angle)
        st = np.sin(angle)

        # coordonnées tournées
        Xr = ct * (X - mu_x) + st * (Y - mu_y)
        Yr = -st * (X - mu_x) + ct * (Y - mu_y)

        # gaussienne anisotrope
        G = w * np.exp(
            -(
                Xr**2 / (2 * sx**2)
                + Yr**2 / (2 * sy**2)
            )
        )

        # asymétrie
        skew = 1 / (1 + np.exp(-skew_strength * Xr))

        G *= skew

        theta += G

    # normalisation
    theta /= np.max(theta)

    return theta


# ============================================================
# CAS 2 : Rotation + Projection 2D
# ============================================================

def demo_rotation():

    print("\n" + "=" * 50)
    print("CAS 2 : Rotation + Projection 2D")
    print("=" * 50)

    p = 40

    sigma = 0.04
    rng = np.random.default_rng(1)

    angles = np.linspace(
        0,
        2 * np.pi,
        12,
        endpoint=False
    )

    # grille 2D
    x = np.linspace(-1, 1, p)

    X, Y_grid = np.meshgrid(x, x)

    # modèle
    model = RotationProjection2D(
        sigma=sigma,
        angles=angles,
        X_grid=X,
        Y_grid=Y_grid,
        x_1d=x
    )

    # ========================================================
    # Construction du vrai signal
    # ========================================================

    theta_true = build_asymmetric_gaussian_mixture(
        X,
        Y_grid
    )

    theta_true = model.enforce_constraints(theta_true)

    # ========================================================
    # Génération des données
    # ========================================================

    Y, Z = generate_data(
        model,
        theta_true,
        n_samples=50
    )

    # IMPORTANT :
    # Y contient des projections 1D
    plot_samples_1d(Y)

    # ========================================================
    # Initialisation
    # ========================================================

    theta_init = rng.standard_normal((p, p))

    theta_init = model.enforce_constraints(theta_init)

    # ========================================================
    # Optimiseur
    # ========================================================

    optimizer = ArmijoOptimizer()

    print(
        "Optimizer utilisé :",
        optimizer.__class__.__name__
    )

    # ========================================================
    # EM
    # ========================================================

    theta_est, history, Q_history = em(
        model,
        Y,
        theta_init,
        n_iter=30,
        optimizer=optimizer
    )

    theta_est = model.enforce_constraints(theta_est)

    # ========================================================
    # Erreur
    # ========================================================

    err = (
        np.linalg.norm(theta_est - theta_true)
        / np.linalg.norm(theta_true)
    )

    print(f"Erreur relative : {err:.4f}")

    # ========================================================
    # Affichage reconstruction
    # ========================================================

    plot_images(
        theta_true,
        theta_est
    )

    # ========================================================
    # Convergence
    # ========================================================

    plot_convergence(
        Q_history,
        "Convergence de Q (Rotation)"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    demo_translation()

    demo_rotation()