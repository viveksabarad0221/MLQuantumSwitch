"""povm_compatibility.py

Reproduces Fig. 3 from:
Guo & Luo, Phys. Rev. A 110, 062206 (2024)

Includes an additional quantifier:
    F(M,N) = \\sum_{i,j} ||{M_i, N_j}||_F

This version uses QuTiP (Qobj) for operator algebra and norms.
"""

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt


# ============================================================
# Basic operators (QuTiP)
# ============================================================

sigma_x = qt.sigmax()
sigma_z = qt.sigmaz()
I2 = qt.qeye(2)


# ============================================================
# Helper norms
# ============================================================

def schatten2_norm(A):
    """Frobenius / Schatten-2 norm."""
    # QuTiP's 'fro' matches the Frobenius (Schatten-2) norm.
    return float(A.norm("fro"))


def trace_norm(A):
    """Schatten-1 (trace) norm."""
    return float(A.norm("tr"))


# ============================================================
# Commutator / Anticommutator
# ============================================================

def comm(A, B):
    return A * B - B * A


def anticomm(A, B):
    return A * B + B * A


# ============================================================
# POVMs from the paper
# ============================================================

def povms(theta):
    """
    Returns M(theta), N(theta) each with 2 outcomes.
    """
    ct = float(np.cos(theta))
    st = float(np.sin(theta))

    M1 = 0.5 * (I2 + (sigma_z * ct + sigma_x * st))
    M2 = 0.5 * (I2 - (sigma_z * ct + sigma_x * st))

    N1 = 0.5 * (I2 + (sigma_z * ct - sigma_x * st))
    N2 = 0.5 * (I2 - (sigma_z * ct - sigma_x * st))

    M = [M1, M2]
    N = [N1, N2]
    return M, N


# ============================================================
# Quantifiers from the paper
# ============================================================

def I_jordan(theta):
    """
    I(M,N) = sum ||{Mi,Nj}||_1 - d
    """
    M, N = povms(theta)
    d = 2

    total = 0.0
    for Mi in M:
        for Nj in N:
            C = 0.5 * anticomm(Mi, Nj)
            total += trace_norm(C)

    return total - d


def gamma_p(theta, p=2):
    """
    Υ_p from the paper.
    """
    M, N = povms(theta)

    total = 0.0
    for Mi in M:
        for Nj in N:
            A = comm(Mi, Nj)
            if p == 2:
                total += schatten2_norm(A)
            else:
                s = np.linalg.svd(A.full(), compute_uv=False)
                total += (np.sum(s**p))**(1/p)
    return total


def I2_quant(theta):
    """
    I2(M,N) = sum tr|[Mi,Nj]|^2
    """
    M, N = povms(theta)

    total = 0.0
    for Mi in M:
        for Nj in N:
            A = comm(Mi, Nj)
            total += float((A.dag() * A).tr())
    return float(total)


def Ir_quant(theta):
    """Closed-form from paper."""
    return float(1 - 1 / (np.cos(theta) + np.sin(theta)))


def IR_quant(theta):
    """Closed-form from paper."""
    return float(1 - np.sqrt(2 + 1) / np.sqrt(2 + np.cos(theta) + np.sin(theta)))


# ============================================================
# ⭐ NEW QUANTIFIER (what you asked)
# ============================================================

def frob_jordan(theta):
    """
    NEW:
    F(M,N) = sum ||{Mi,Nj}||_F
    (Frobenius norm of anticommutators)
    """
    M, N = povms(theta)
    d = 2
    total = 0.0
    for Mi in M:
        for Nj in N:
            C = 0.5 * anticomm(Mi, Nj)
            total += schatten2_norm(C)
    return total - d


# ============================================================
# Plot (Figure 3 style)
# ============================================================

def make_plot():
    thetas = np.linspace(0, np.pi/4, 300)

    I_vals = np.array([I_jordan(t) for t in thetas], dtype=float)
    gamma_vals = np.array([gamma_p(t, 2) for t in thetas], dtype=float)
    I2_vals = np.array([I2_quant(t) for t in thetas], dtype=float)
    Ir_vals = np.array([Ir_quant(t) for t in thetas], dtype=float)
    IR_vals = np.array([IR_quant(t) for t in thetas], dtype=float)
    F_vals = np.array([frob_jordan(t) for t in thetas], dtype=float)  # ⭐ new curve

    plt.figure(figsize=(6, 4))

    plt.plot(thetas, I_vals, label="I (Jordan)", lw=2)
    plt.plot(thetas, gamma_vals, label=r"$\Upsilon_2$", lw=2)
    plt.plot(thetas, I2_vals, label=r"$I_2$", lw=2)
    # plt.plot(thetas, Ir_vals, label=r"$I_r$", lw=2)
    # plt.plot(thetas, IR_vals, label=r"$I_R$", lw=2)

    # ⭐ your new quantifier
    plt.plot(thetas, F_vals, "--", label="Frobenius Jordan (NEW)", lw=2)

    plt.xlabel(r"$\theta$")
    plt.ylabel("Incompatibility quantifier")
    plt.title("Figure 3 reproduction + Frobenius Jordan")

    plt.legend(frameon=False)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    make_plot()
