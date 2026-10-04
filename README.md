# Hidden-Variable Inverse Problems

An EM implementation for inverse problems with discrete latent transformations,
developed by Niokhobaye Abdel and Arthur Conche at Universite Paris Cite.

## Model

For observations $Y_i = A_{Z_i}\theta + \varepsilon_i$, the code estimates
the shared signal $\theta$ while treating each transformation $Z_i$ as latent.
The E-step computes posterior weights; the M-step either solves weighted normal
equations or takes an Armijo-checked gradient step.

Implemented examples:

- Circular translations of one-dimensional signals.
- Two-dimensional rotations followed by numerical projection.
- Optional quadratic regularization.

The 2D model uses the transpose of its discrete bilinear interpolation and
projection operators, so its adjoint can be checked by an inner-product test.

## Setup

Python 3.11 or 3.12 is supported.

```bash
python -m pip install -e ".[test]"
MPLBACKEND=Agg python -m pytest -q
```

Run `python demo.py` for the interactive examples. The data generator and demo
initializations use local seeded random generators and do not reset NumPy's
global random state.

## Report

The mathematical report and slides are in [`rapport/`](rapport/), including
[`doc-2-1.pdf`](rapport/doc-2-1.pdf).

## Acknowledgements

This project was carried out jointly with Arthur Conche. The software was
developed in collaboration with him.
