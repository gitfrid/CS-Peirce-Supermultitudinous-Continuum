#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FULL RMT + UNIVERSALITY PIPELINE (VS CODE READY)
------------------------------------------------
This script:
1. Builds operator (self-adjoint)
2. Computes eigenvalues
3. Extracts BULK spectrum
4. Computes spacing distribution
5. Compares against Poisson and GOE
6. Produces plot + numerical fit scores

Run:
    python rmt_full_pipeline.py
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt

# ---------------- GRID ----------------
def build_grid(N=600, r_min=1e-6, r_max=10.0):
    return np.geomspace(r_min, r_max, N)

# ---------------- DERIVATIVE ----------------
def derivative_matrix(r):
    N = len(r)
    data, rows, cols = [], [], []

    for i in range(1, N - 1):
        h1 = r[i] - r[i - 1]
        h2 = r[i + 1] - r[i]

        rows += [i, i, i]
        cols += [i - 1, i, i + 1]
        data += [-1 / h1, (1 / h1 - 1 / h2), 1 / h2]

    return sp.csr_matrix((data, (rows, cols)), shape=(N, N))

# ---------------- OPERATOR ----------------
def build_operator(r, r_eps=1e-5):
    D = derivative_matrix(r)
    L = -D.T @ D

    V = -1.0 / (r + r_eps)
    V = np.clip(V, -1e3, 1e3)

    return L + sp.diags(V, 0)

# ---------------- EIGENVALUES ----------------
def compute_eigs(H, k=80):
    vals = sla.eigsh(H, k=k, which='SA', return_eigenvectors=False)
    return np.sort(np.real(vals))

# ---------------- BULK EXTRACTION ----------------
def extract_bulk(vals):
    n = len(vals)
    return vals[n//4 : 3*n//4]

# ---------------- SPACING ----------------
def spacing_distribution(vals):
    vals = np.sort(vals)
    spacings = np.diff(vals)
    s = spacings / np.mean(spacings)
    return s

# ---------------- PDFs ----------------
def poisson_pdf(s):
    return np.exp(-s)


def goe_pdf(s):
    return (np.pi/2) * s * np.exp(-np.pi * s**2 / 4)

# ---------------- FIT QUALITY ----------------
def fit_error(empirical_hist, x, model_pdf):
    model_vals = model_pdf(x)
    return np.mean((empirical_hist - model_vals)**2)

# ---------------- MAIN ----------------
def main():
    print("Running full RMT analysis...\n")

    # 1. Build system
    r = build_grid()
    H = build_operator(r)

    # 2. Eigenvalues
    vals = compute_eigs(H)

    # 3. Bulk only (critical!)
    vals_bulk = extract_bulk(vals)

    # Shift to positive (for log stability if needed)
    vals_bulk = vals_bulk - np.min(vals_bulk)

    # 4. Spacing
    s = spacing_distribution(vals_bulk)

    # 5. Histogram
    hist, bins = np.histogram(s, bins=40, density=True)
    centers = 0.5 * (bins[1:] + bins[:-1])

    # 6. Compare
    poisson_err = fit_error(hist, centers, poisson_pdf)
    goe_err = fit_error(hist, centers, goe_pdf)

    # 7. Plot
    x = np.linspace(0, max(s), 300)

    plt.figure()
    plt.bar(centers, hist, width=centers[1] - centers[0], alpha=0.6, label="Empirical")
    plt.plot(x, poisson_pdf(x), label="Poisson")
    plt.plot(x, goe_pdf(x), label="GOE")

    plt.xlabel("s (normalized spacing)")
    plt.ylabel("P(s)")
    plt.title("RMT Level Spacing (BULK)")
    plt.legend()

    plt.savefig("C:/math/rmt_result.png")

    print("\n=== RESULTS ===")
    print(f"Poisson fit error: {poisson_err:.6f}")
    print(f"GOE fit error:     {goe_err:.6f}")

    if poisson_err < goe_err:
        print("\n→ System is LIKELY INTEGRABLE (Poisson)")
    else:
        print("\n→ System shows CHAOTIC signature (GOE)")

    print("\nPlot saved to: C:/math/rmt_result.png")


if __name__ == "__main__":
    main()
