"""
qchannel_kernel.py
==================

A small, self-contained module to simulate *quantum channel compatibility* learning
with Choi-matrix kernels.

Current scope:
- Fast path: *partially depolarizing* channels on dimension d:
            Φ_t(ρ) = t ρ + (1-t) Tr(ρ) I/d,  with t in [0,1].
    Ground-truth compatibility for a *pair* (Φ_s, Φ_t) uses a known analytic condition
    for depolarizing channels (Eq. (5) in Zhang & Nechita, arXiv:2204.09963).

- Extension: *general CPTP qubit channels* represented by Kraus operators.
    For generic channels, compatibility labeling requires solving an SDP; this module
    provides an optional CVXPY-based oracle for qubits.

Why this scope?
- General channel compatibility is an SDP; this module avoids external SDP solvers and
  still lets you benchmark kernel methods end-to-end with a nontrivial compatibility boundary.

You can extend:
- Replace `label_fn` in `generate_dataset(...)` with your own compatibility oracle.
- Replace channel family generation with arbitrary CPTP channels once you provide labels
  (e.g., from cvxpy).

Main ideas implemented:
- Feature object for a channel is its Choi matrix J(Φ).
- Linear kernel is Hilbert–Schmidt overlap: k(Φ,Ψ)=Tr(J(Φ) J(Ψ)).
- A *pair* of channels is embedded as a direct sum J(Φ) ⊕ J(Ψ), so
      k((Φ1,Φ2),(Ψ1,Ψ2)) = Tr(J(Φ1)J(Ψ1)) + Tr(J(Φ2)J(Ψ2)).

This matches "direct measurement" via a SWAP test on the full Choi register (in principle),
but in simulation we compute overlaps numerically.

Dependencies:
- numpy
Optional:
- scikit-learn (for SVM evaluation helpers)
- cvxpy (for the SDP compatibility oracle on general qubit channels)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Sequence, Tuple

import numpy as np
import numpy.random as npr
import qutip as qt
from qutip import Qobj

# scikit-learn is optional at import-time; only required for `run_benchmark(...)`.
try:  # pragma: no cover
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
except Exception:  # pragma: no cover
    train_test_split = None
    SVC = None
    accuracy_score = balanced_accuracy_score = roc_auc_score = confusion_matrix = None


Array = np.ndarray


class QuantumChannel(Protocol):
    """Minimal protocol for channels used by this module."""

    @property
    def d_in(self) -> int: ...

    @property
    def d_out(self) -> int: ...

    def choi(self) -> Qobj: ...


def _as_choi_qobj(J: Qobj | Array, d_out: int, d_in: int) -> Qobj:
    if isinstance(J, Qobj):
        if J.dims == [[d_out, d_in], [d_out, d_in]]:
            return J
        return Qobj(J.full(), dims=[[d_out, d_in], [d_out, d_in]])
    arr = np.asarray(J, dtype=np.complex128)
    return Qobj(arr, dims=[[d_out, d_in], [d_out, d_in]])


def _is_psd_hermitian(op: Qobj, eps: float = 1e-10) -> bool:
    if not op.isherm:
        return False
    w = np.array(op.eigenenergies(), dtype=np.float64)
    return bool(np.min(w) >= -eps)


def is_cptp_choi(J: Qobj | Array, d_out: int, d_in: int, eps: float = 1e-8) -> bool:
    """Check (approximately) whether a Choi matrix corresponds to a CPTP map.

    Convention: J = (Φ ⊗ id)(|Ω⟩⟨Ω|) with |Ω⟩ normalized.
    Then TP is: Tr_out(J) = I_in / d_in.
    """
    if np.shape(J) != (d_out * d_in, d_out * d_in):
        return False
    Jq = _as_choi_qobj(J, d_out=d_out, d_in=d_in)
    if not _is_psd_hermitian(Jq, eps=eps):
        return False
    tr_out = Jq.ptrace(1)
    target = qt.qeye(d_in) / d_in
    return bool((tr_out - target).norm() <= eps)


# -----------------------------
# 1) Channel family: depolarizing
# -----------------------------

def maximally_entangled_choi_state(d: int) -> Qobj:
    """
    Return |Ω><Ω| where |Ω> = (1/sqrt(d)) Σ_i |i,i> on C^d ⊗ C^d.
    Returned as a QuTiP Qobj on C^d ⊗ C^d with dims [[d,d],[d,d]].
    """
    # Build |Ω⟩ = (1/√d) Σ_i |i,i⟩ as a NumPy vector (much faster than repeated qutip.tensor).
    omega = np.zeros((d * d,), dtype=np.complex128)
    for i in range(d):
        omega[i * d + i] = 1.0 / np.sqrt(d)
    J = np.outer(omega, omega.conj())
    return Qobj(J, dims=[[d, d], [d, d]])


@lru_cache(maxsize=None)
def _choi_id_normalized(d: int) -> Qobj:
    return maximally_entangled_choi_state(d)


@lru_cache(maxsize=None)
def _choi_depol_base(d: int) -> Qobj:
    # J(Δ) = I/(d^2) under the normalized-|Ω⟩ convention.
    return Qobj((np.eye(d * d, dtype=np.complex128) / (d * d)), dims=[[d, d], [d, d]])


def choi_depolarizing(t: float, d: int) -> Qobj:
    """
    Choi matrix of the partially depolarizing channel Φ_t(ρ)= t ρ + (1-t) Tr(ρ) I/d.
    Using J(id)=|Ω><Ω| and J(Δ)=I/d^2.
    """
    if not (0.0 <= t <= 1.0):
        raise ValueError(f"t must be in [0,1], got {t}")
    J_id = _choi_id_normalized(d)
    J_depol = _choi_depol_base(d)
    J = t * J_id + (1.0 - t) * J_depol
    return Qobj(J.full(), dims=[[d, d], [d, d]])


@dataclass(frozen=True)
class DepolarizingChannel:
    """Φ_t on dimension d."""
    t: float
    d: int = 2

    @property
    def d_in(self) -> int:
        return self.d

    @property
    def d_out(self) -> int:
        return self.d

    def choi(self) -> Qobj:
        return choi_depolarizing(self.t, self.d)


def choi_from_kraus(kraus_ops: Sequence[Qobj | Array], d_in: int, d_out: int) -> Qobj:
    """Build a (normalized) Choi matrix from Kraus operators.

    Args:
        kraus_ops: sequence of K_i with shape (d_out, d_in)
        d_in: input dimension
        d_out: output dimension

    Returns:
        J of shape (d_out*d_in, d_out*d_in), with |Ω⟩ normalized.
    """
    omega = maximally_entangled_choi_state(d_in)
    omega_arr = np.asarray(omega.full(), dtype=np.complex128)
    J_arr = np.zeros((d_out * d_in, d_out * d_in), dtype=np.complex128)
    I_in = qt.qeye(d_in)
    for K in kraus_ops:
        Kq = K if isinstance(K, Qobj) else Qobj(np.asarray(K, dtype=np.complex128), dims=[[d_out], [d_in]])
        if Kq.shape != (d_out, d_in):
            raise ValueError(f"Kraus operator has shape {Kq.shape}, expected {(d_out, d_in)}")
        A = qt.tensor(Kq, I_in)
        A_arr = np.asarray(A.full(), dtype=np.complex128)
        J_arr = J_arr + A_arr @ omega_arr @ A_arr.conj().T
    return Qobj(J_arr, dims=[[d_out, d_in], [d_out, d_in]])


@dataclass(frozen=True)
class KrausChannel:
    """A CPTP channel specified by Kraus operators."""

    kraus_ops: Tuple[Qobj, ...]
    _d_in: int
    _d_out: int
    validate: bool = True
    eps: float = 1e-8

    @property
    def d_in(self) -> int:
        return self._d_in

    @property
    def d_out(self) -> int:
        return self._d_out

    def choi(self) -> Qobj:
        J = choi_from_kraus(self.kraus_ops, d_in=self._d_in, d_out=self._d_out)
        if self.validate and not is_cptp_choi(J, d_out=self._d_out, d_in=self._d_in, eps=self.eps):
            raise ValueError("Kraus operators do not define a CPTP map (within eps).")
        return J


def random_qubit_channel(
    rng: npr.Generator,
    kraus_rank: int = 4,
) -> KrausChannel:
    """Sample a random qubit CPTP channel using a random Stinespring isometry.

    Constructs an isometry V: C^2 -> C^2 ⊗ C^k and reads off k Kraus operators.
    """
    if kraus_rank < 1:
        raise ValueError("kraus_rank must be >= 1")
    d_in = d_out = 2
    k = int(kraus_rank)
    # Random complex Ginibre matrix then QR to get an isometry.
    G = (rng.normal(size=(d_out * k, d_in)) + 1j * rng.normal(size=(d_out * k, d_in))) / np.sqrt(2.0)
    Q, _ = np.linalg.qr(G)
    # Ensure shape (2k,2)
    Q = Q[:, :d_in]
    # Reshape to (k, d_out, d_in) and take Kraus operators K_e = <e|V.
    blocks = Q.reshape(k, d_out, d_in)
    kraus = tuple(Qobj(blocks[e], dims=[[d_out], [d_in]]) for e in range(k))
    return KrausChannel(kraus_ops=kraus, _d_in=d_in, _d_out=d_out, validate=True)


@dataclass(frozen=True)
class ChannelPair:
    """A data point: (Φ, Ψ)."""
    ch1: QuantumChannel
    ch2: QuantumChannel

    @property
    def d_in(self) -> int:
        if self.ch1.d_in != self.ch2.d_in:
            raise ValueError("Input dimensions mismatch.")
        return self.ch1.d_in

    @property
    def d_out(self) -> int:
        if self.ch1.d_out != self.ch2.d_out:
            raise ValueError("Output dimensions mismatch.")
        return self.ch1.d_out

    @property
    def d(self) -> int:
        if self.d_in != self.d_out:
            raise ValueError("This pair does not have square channels (d_in != d_out).")
        return self.d_in


# -----------------------------------------
# 2) Compatibility oracle for depolarizing
# -----------------------------------------

def compatible_two_depolarizing(s: float, t: float, d: int, eps: float = 1e-12) -> bool:
    """
    Necessary and sufficient condition for compatibility of two depolarizing channels Φ_s and Φ_t:

        t + s - (2/d) * sqrt((1-t)(1-s)) <= 1

    (Eq. (5) in Zhang & Nechita, arXiv:2204.09963, citing Has17/Haa19/NPR21.)

    Returns True if compatible, else False.
    """
    if not (0.0 <= s <= 1.0 and 0.0 <= t <= 1.0):
        raise ValueError(f"s,t must be in [0,1], got s={s}, t={t}")
    lhs = t + s - (2.0 / d) * np.sqrt(max(0.0, (1.0 - t) * (1.0 - s)))
    return lhs <= 1.0 + eps


def label_pair_depolarizing(pair: ChannelPair, eps: float = 1e-12) -> int:
    """Return 1 if compatible, else 0, for a ChannelPair of depolarizing channels."""
    if not isinstance(pair.ch1, DepolarizingChannel) or not isinstance(pair.ch2, DepolarizingChannel):
        raise TypeError("label_pair_depolarizing expects DepolarizingChannel instances.")
    return int(compatible_two_depolarizing(pair.ch1.t, pair.ch2.t, pair.d, eps=eps))


def compatible_two_channels_sdp_qubit(
    ch1: QuantumChannel,
    ch2: QuantumChannel,
    *,
    solver: str = "SCS",
    eps: float = 1e-6,
    verbose: bool = False,
) -> bool:
    """Compatibility oracle for two *qubit* channels using an SDP (CVXPY).

    ch1, ch2 must both be 2->2 channels.

    The SDP searches for a joint CPTP channel Λ: C^2 -> C^2 ⊗ C^2 such that
    its marginals are ch1 and ch2.
    """
    if ch1.d_in != 2 or ch1.d_out != 2 or ch2.d_in != 2 or ch2.d_out != 2:
        raise ValueError("This oracle only supports qubit channels (2->2).")
    if ch1.d_in != ch2.d_in or ch1.d_out != ch2.d_out:
        raise ValueError("Dimension mismatch between channels.")

    try:
        import cvxpy as cp  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("cvxpy is required for compatible_two_channels_sdp_qubit(...)") from e

    J1 = np.asarray(ch1.choi().full(), dtype=np.complex128)
    J2 = np.asarray(ch2.choi().full(), dtype=np.complex128)

    d_in = 2
    dA = dB = 2
    D = dA * dB * d_in  # dimension of the Choi register (A,B,in)
    J = cp.Variable((D, D), complex=True, hermitian=True)

    def idx(a: int, b: int, i: int) -> int:
        return (a * dB + b) * d_in + i

    constraints: list[Any] = [J >> 0]

    # Trace-preserving: Tr_{AB} J = I_in / d_in
    target_tp = np.eye(d_in, dtype=np.complex128) / d_in
    for i in range(d_in):
        for ip in range(d_in):
            expr = 0
            for a in range(dA):
                for b in range(dB):
                    expr += J[idx(a, b, i), idx(a, b, ip)]
            constraints.append(expr == target_tp[i, ip])

    # Marginal to A: Tr_B J = J1
    for a in range(dA):
        for i in range(d_in):
            row = a * d_in + i
            for ap in range(dA):
                for ip in range(d_in):
                    col = ap * d_in + ip
                    expr = 0
                    for b in range(dB):
                        expr += J[idx(a, b, i), idx(ap, b, ip)]
                    constraints.append(expr == J1[row, col])

    # Marginal to B: Tr_A J = J2
    for b in range(dB):
        for i in range(d_in):
            row = b * d_in + i
            for bp in range(dB):
                for ip in range(d_in):
                    col = bp * d_in + ip
                    expr = 0
                    for a in range(dA):
                        expr += J[idx(a, b, i), idx(a, bp, ip)]
                    constraints.append(expr == J2[row, col])

    prob = cp.Problem(cp.Minimize(0), constraints)
    try:
        prob.solve(solver=solver, verbose=verbose, eps=eps)
    except TypeError:
        # Some solvers don't accept `eps`; fall back.
        prob.solve(solver=solver, verbose=verbose)
    return prob.status in ("optimal", "optimal_inaccurate")


def label_pair_qubit_sdp(pair: ChannelPair, solver: str = "SCS", eps: float = 1e-6) -> int:
    """Return 1 if compatible (via SDP), else 0. Only supports 2->2 channels."""
    return int(
        compatible_two_channels_sdp_qubit(pair.ch1, pair.ch2, solver=solver, eps=eps, verbose=False)
    )


# --------------------------
# 3) Kernels on Choi matrices
# --------------------------

def hs_overlap(A: Qobj | Array, B: Qobj | Array) -> float:
    """Hilbert–Schmidt overlap Tr(A^† B). For Hermitian A,B this is Tr(AB)."""
    A_arr = A.full() if isinstance(A, Qobj) else np.asarray(A, dtype=np.complex128)
    B_arr = B.full() if isinstance(B, Qobj) else np.asarray(B, dtype=np.complex128)
    return float(np.real(np.trace(A_arr.conj().T @ B_arr)))


def kernel_linear_choi_pair(p: ChannelPair, q: ChannelPair, normalize: bool = True) -> float:
    """
    Linear kernel on channel pairs using Choi overlaps:

        k(p,q) = Tr(J1 J1') + Tr(J2 J2')

    If normalize=True, divide by d^2 so that values are O(1).
    """
    d_in, d_out = p.d_in, p.d_out
    if q.d_in != d_in or q.d_out != d_out:
        raise ValueError("Dimension mismatch between pairs.")
    k = hs_overlap(p.ch1.choi(), q.ch1.choi()) + hs_overlap(p.ch2.choi(), q.ch2.choi())
    scale = (d_in * d_out)
    return k / scale if normalize else k


def kernel_poly_from_linear(k0: float, degree: int = 2, c: float = 1.0) -> float:
    """Polynomial kernel built from a base scalar kernel value: (k0 + c)^degree."""
    if degree < 1:
        raise ValueError("degree must be >= 1")
    return float((k0 + c) ** degree)


def kernel_rbf_from_distance_sq(dist2: float, gamma: float) -> float:
    """RBF kernel exp(-gamma * dist^2)."""
    return float(np.exp(-gamma * dist2))


def pair_feature_blockdiag(p: ChannelPair) -> Array:
    """
    Block-diagonal embedding: J(Φ) ⊕ J(Ψ).
    Shape: (2 (d_out*d_in), 2 (d_out*d_in))
    """
    J1 = np.asarray(p.ch1.choi().full(), dtype=np.complex128)
    J2 = np.asarray(p.ch2.choi().full(), dtype=np.complex128)
    top = np.concatenate([J1, np.zeros_like(J1)], axis=1)
    bot = np.concatenate([np.zeros_like(J2), J2], axis=1)
    return np.concatenate([top, bot], axis=0)


def hs_distance_sq_pair(p: ChannelPair, q: ChannelPair, normalize: bool = True) -> float:
    """
    Squared Hilbert–Schmidt distance between blockdiag embeddings:
        ||(J1⊕J2) - (J1'⊕J2')||_2^2
      = ||J1-J1'||_2^2 + ||J2-J2'||_2^2.
    """
    d_in, d_out = p.d_in, p.d_out
    if q.d_in != d_in or q.d_out != d_out:
        raise ValueError("Dimension mismatch between pairs.")
    Jp1, Jp2 = p.ch1.choi(), p.ch2.choi()
    Jq1, Jq2 = q.ch1.choi(), q.ch2.choi()
    d2 = hs_overlap(Jp1 - Jq1, Jp1 - Jq1) + hs_overlap(Jp2 - Jq2, Jp2 - Jq2)
    scale = (d_in * d_out)
    return d2 / scale if normalize else d2


# --------------------------
# 4) Dataset + Gram matrices
# --------------------------

def generate_dataset(
    n: int,
    d: int = 2,
    seed: int = 0,
    t_range: Tuple[float, float] = (0.0, 1.0),
    distribution: Literal["uniform", "beta"] = "uniform",
    beta_ab: Tuple[float, float] = (0.7, 0.7),
    label_fn: Optional[Callable[[ChannelPair], int]] = None,
    channel_family: Literal["depolarizing", "random_qubit"] = "depolarizing",
    kraus_rank: int = 4,
) -> Tuple[List[ChannelPair], np.ndarray]:
    """
        Generate n random ChannelPairs.

        - If channel_family='depolarizing': sample parameters s,t and return pairs (Φ_s, Φ_t).
        - If channel_family='random_qubit': sample two independent random 2->2 CPTP channels.

    Returns:
      pairs: list[ChannelPair]
      y:     array shape (n,) with labels 0/1
    """
    rng = npr.default_rng(seed)

    if label_fn is None:
        if channel_family == "depolarizing":
            label_fn = label_pair_depolarizing
        else:
            label_fn = label_pair_qubit_sdp

    pairs: List[ChannelPair] = []
    y = np.zeros((n,), dtype=np.int64)

    if channel_family == "depolarizing":
        lo, hi = t_range
        if lo < 0 or hi > 1 or lo >= hi:
            raise ValueError("t_range must satisfy 0 <= lo < hi <= 1.")
        for i in range(n):
            if distribution == "uniform":
                s = float(rng.uniform(lo, hi))
                t = float(rng.uniform(lo, hi))
            elif distribution == "beta":
                a, b = beta_ab
                s = float(lo + (hi - lo) * rng.beta(a, b))
                t = float(lo + (hi - lo) * rng.beta(a, b))
            else:
                raise ValueError("distribution must be 'uniform' or 'beta'.")
            p = ChannelPair(DepolarizingChannel(s, d=d), DepolarizingChannel(t, d=d))
            pairs.append(p)
            y[i] = int(label_fn(p))

    elif channel_family == "random_qubit":
        if d != 2:
            raise ValueError("channel_family='random_qubit' only supports d=2.")
        for i in range(n):
            c1 = random_qubit_channel(rng, kraus_rank=kraus_rank)
            c2 = random_qubit_channel(rng, kraus_rank=kraus_rank)
            p = ChannelPair(c1, c2)
            pairs.append(p)
            y[i] = int(label_fn(p))
    else:
        raise ValueError("channel_family must be 'depolarizing' or 'random_qubit'.")

    return pairs, y


def gram_matrix(
    pairs: List[ChannelPair],
    kernel: Callable[[ChannelPair, ChannelPair], float],
    symmetric: bool = True,
    dtype=np.float64,
) -> np.ndarray:
    """
    Compute Gram matrix K_ij = kernel(p_i, p_j).
    """
    n = len(pairs)
    K = np.zeros((n, n), dtype=dtype)
    if symmetric:
        for i in range(n):
            K[i, i] = kernel(pairs[i], pairs[i])
            for j in range(i + 1, n):
                v = kernel(pairs[i], pairs[j])
                K[i, j] = v
                K[j, i] = v
    else:
        for i in range(n):
            for j in range(n):
                K[i, j] = kernel(pairs[i], pairs[j])
    return K


# --------------------------
# 5) Benchmarking (SVM, precomputed kernel)
# --------------------------

@dataclass
class BenchmarkResult:
    acc: float
    bacc: float
    auc: float
    cm: np.ndarray
    details: Dict[str, float | str]


def run_benchmark(
    n: int = 800,
    test_size: float = 0.3,
    d: int = 2,
    seed: int = 0,
    kernel_kind: Literal["linear", "poly", "rbf"] = "poly",
    poly_degree: int = 2,
    poly_c: float = 1.0,
    rbf_gamma: float = 2.0,
    C: float = 5.0,
    distribution: Literal["uniform", "beta"] = "uniform",
    beta_ab: Tuple[float, float] = (0.7, 0.7),
    t_range: Tuple[float, float] = (0.0, 1.0),
    channel_family: Literal["depolarizing", "random_qubit"] = "depolarizing",
    kraus_rank: int = 4,
) -> BenchmarkResult:
    """
    End-to-end benchmark:
    - generate random channel pairs (depolarizing or random qubit CPTP)
    - label by an oracle (analytic for depolarizing; SDP requires cvxpy for general qubits)
      - build Gram matrix from a Choi-based kernel
      - train kernel SVM (precomputed kernel) and evaluate

    Returns accuracy, balanced accuracy, AUC, confusion matrix.

    Note:
      For 'rbf', we build it from HS distance on blockdiag Choi embedding:
          k(p,q)=exp(-gamma * ||Jp-Jq||_2^2).
    """
    if (
        train_test_split is None
        or SVC is None
        or accuracy_score is None
        or balanced_accuracy_score is None
        or roc_auc_score is None
        or confusion_matrix is None
    ):
        raise ImportError("scikit-learn is required for run_benchmark(...)")

    # Help static type-checkers after the runtime guard above.
    assert train_test_split is not None
    assert SVC is not None
    assert accuracy_score is not None
    assert balanced_accuracy_score is not None
    assert roc_auc_score is not None
    assert confusion_matrix is not None

    pairs, y = generate_dataset(
        n=n,
        d=d,
        seed=seed,
        distribution=distribution,
        beta_ab=beta_ab,
        t_range=t_range,
        channel_family=channel_family,
        kraus_rank=kraus_rank,
    )

    idx = np.arange(n)
    idx_tr, idx_te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)

    pairs_tr = [pairs[i] for i in idx_tr]
    pairs_te = [pairs[i] for i in idx_te]
    y_tr = y[idx_tr]
    y_te = y[idx_te]

    # define kernels
    if kernel_kind == "linear":
        def k(p, q):
            return kernel_linear_choi_pair(p, q, normalize=True)
    elif kernel_kind == "poly":
        def k(p, q):
            k0 = kernel_linear_choi_pair(p, q, normalize=True)
            return kernel_poly_from_linear(k0, degree=poly_degree, c=poly_c)
    elif kernel_kind == "rbf":
        def k(p, q):
            dist2 = hs_distance_sq_pair(p, q, normalize=True)
            return kernel_rbf_from_distance_sq(dist2, gamma=rbf_gamma)
    else:
        raise ValueError("kernel_kind must be 'linear', 'poly', or 'rbf'.")

    # Gram matrices
    K_tr = gram_matrix(pairs_tr, k, symmetric=True)
    # test Gram: K_te[i,j] = k(test_i, train_j)
    K_te = np.zeros((len(pairs_te), len(pairs_tr)), dtype=np.float64)
    for i, p in enumerate(pairs_te):
        for j, q in enumerate(pairs_tr):
            K_te[i, j] = k(p, q)

    clf = SVC(kernel="precomputed", C=C, probability=True, random_state=seed)
    clf.fit(K_tr, y_tr)

    y_hat = clf.predict(K_te)
    y_prob = clf.predict_proba(K_te)[:, 1]

    acc = float(accuracy_score(y_te, y_hat))
    bacc = float(balanced_accuracy_score(y_te, y_hat))
    auc = float(roc_auc_score(y_te, y_prob))
    cm = confusion_matrix(y_te, y_hat)

    return BenchmarkResult(
        acc=acc,
        bacc=bacc,
        auc=auc,
        cm=cm,
        details=dict(
            n=n,
            test_size=test_size,
            d=d,
            seed=seed,
            kernel_kind=kernel_kind,
            poly_degree=poly_degree,
            poly_c=poly_c,
            rbf_gamma=rbf_gamma,
            C=C,
            compatible_frac=float(np.mean(y)),
        ),
    )


__all__ = [
    "DepolarizingChannel",
    "KrausChannel",
    "QuantumChannel",
    "ChannelPair",
    "choi_depolarizing",
    "choi_from_kraus",
    "random_qubit_channel",
    "is_cptp_choi",
    "compatible_two_depolarizing",
    "compatible_two_channels_sdp_qubit",
    "label_pair_depolarizing",
    "label_pair_qubit_sdp",
    "kernel_linear_choi_pair",
    "kernel_poly_from_linear",
    "kernel_rbf_from_distance_sq",
    "hs_distance_sq_pair",
    "generate_dataset",
    "gram_matrix",
    "BenchmarkResult",
    "run_benchmark",
]

def run_cross_family_benchmark(
    n_train: int = 800,
    n_test: int = 400,
    d: int = 2,
    seed: int = 0,
    kernel_kind: Literal["linear", "poly", "rbf"] = "poly",
    poly_degree: int = 2,
    poly_c: float = 1.0,
    rbf_gamma: float = 2.0,
    C: float = 5.0,
    # train (depolarizing) sampling
    distribution: Literal["uniform", "beta"] = "uniform",
    beta_ab: Tuple[float, float] = (0.7, 0.7),
    t_range: Tuple[float, float] = (0.0, 1.0),
    # test (random qubit) sampling
    kraus_rank: int = 4,
) -> BenchmarkResult:
    """
    Train on depolarizing channel pairs, test on random qubit channel pairs.
    Labels:
      - train: analytic depolarizing compatibility
      - test: SDP oracle for qubit-channel compatibility (requires cvxpy)
    """

    if (
        SVC is None
        or accuracy_score is None
        or balanced_accuracy_score is None
        or roc_auc_score is None
        or confusion_matrix is None
    ):
        raise ImportError("scikit-learn is required for run_cross_family_benchmark(...)")

    # 1) generate TRAIN depolarizing
    pairs_tr, y_tr = generate_dataset(
        n=n_train,
        d=d,
        seed=seed,
        distribution=distribution,
        beta_ab=beta_ab,
        t_range=t_range,
        channel_family="depolarizing",
        label_fn=label_pair_depolarizing,
    )

    # 2) generate TEST random qubit (d must be 2 in current implementation)
    if d != 2:
        raise ValueError("random_qubit test currently only supports d=2.")
    pairs_te, y_te = generate_dataset(
        n=n_test,
        d=2,
        seed=seed + 1,
        channel_family="random_qubit",
        kraus_rank=kraus_rank,
        label_fn=label_pair_qubit_sdp,  # <- needs cvxpy
    )

    # 3) kernel definition (same as run_benchmark)
    if kernel_kind == "linear":
        def k(p, q):
            return kernel_linear_choi_pair(p, q, normalize=True)
    elif kernel_kind == "poly":
        def k(p, q):
            k0 = kernel_linear_choi_pair(p, q, normalize=True)
            return kernel_poly_from_linear(k0, degree=poly_degree, c=poly_c)
    elif kernel_kind == "rbf":
        def k(p, q):
            dist2 = hs_distance_sq_pair(p, q, normalize=True)
            return kernel_rbf_from_distance_sq(dist2, gamma=rbf_gamma)
    else:
        raise ValueError("kernel_kind must be 'linear', 'poly', or 'rbf'.")

    # 4) Gram matrices:
    #    train Gram is square; test Gram is (n_test x n_train)
    K_tr = gram_matrix(pairs_tr, k, symmetric=True)
    K_te = np.zeros((len(pairs_te), len(pairs_tr)), dtype=np.float64)
    for i, p in enumerate(pairs_te):
        for j, q in enumerate(pairs_tr):
            K_te[i, j] = k(p, q)

    # 5) fit + evaluate
    clf = SVC(kernel="precomputed", C=C, probability=True, random_state=seed)
    clf.fit(K_tr, y_tr)

    y_hat = clf.predict(K_te)
    y_prob = clf.predict_proba(K_te)[:, 1]

    acc = float(accuracy_score(y_te, y_hat))
    bacc = float(balanced_accuracy_score(y_te, y_hat))
    auc = float(roc_auc_score(y_te, y_prob))
    cm = confusion_matrix(y_te, y_hat)

    return BenchmarkResult(
        acc=acc,
        bacc=bacc,
        auc=auc,
        cm=cm,
        details=dict(
            n_train=n_train,
            n_test=n_test,
            d=d,
            seed=seed,
            kernel_kind=kernel_kind,
            poly_degree=poly_degree,
            poly_c=poly_c,
            rbf_gamma=rbf_gamma,
            C=C,
            train_family="depolarizing",
            test_family="random_qubit",
            train_compat_frac=float(np.mean(y_tr)),
            test_compat_frac=float(np.mean(y_te)),
            kraus_rank=kraus_rank,
        ),
    )
