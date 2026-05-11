"""
Quantum measurement incompatibility and clustering utilities.

This module provides:

- operator_double_ket: Convert a square Qobj operator into its column vector form |A⟩⟩.
- IncompatibilityToolkit: Stateless methods to
    * generate qubit Pauli operators and binary projective POVMs from Bloch directions,
    * validate POVMs (positivity and completeness),
    * compute mutual eigenspace disturbance between two POVMs by analytical, numerical, or commutator-based formulas,
    * sample unit vectors on a spherical cap and build POVMs from Bloch vectors,
    * recover the Bloch direction from a binary projective qubit POVM.
- ClusteringToolkit: Stateless methods to
    * cluster items given a pairwise distance matrix via PAM-style k-medoids (with 'kmeans++', 'linear++' or 'random' init) or k-means (classical MDS followed by Lloyd’s algorithm),
    * helper routines for medoid initialization, MDS embedding, and distance computations.

Dependencies:
- qutip
- numpy
- Optional: scikit-learn and sklearn-extra for alternative clustering implementations

Example:
    from incompatibility_tools import IncompatibilityToolkit, ClusteringToolkit
    import numpy as np

    # Construct binary projective POVMs along X, Y, Z axes
    dirs = np.eye(3)
    povms = [IncompatibilityToolkit.projective_qubit_povm_from_axis(d) for d in dirs]

    # Compute analytical disturbance between X and Z POVMs
    d_xz = IncompatibilityToolkit.mutual_eigenspace_disturbance(povms[0], povms[2], method='analytical')

    # Cluster a distance matrix of observables
    labels = ClusteringToolkit.cluster_from_distance([[0, d_xz, ...], ...], n_clusters=2)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Optional

import numpy as np
from numpy.linalg import norm as _norm

from qutip import Qobj, Bloch, qeye, sigmax, sigmay, sigmaz, tensor, commutator, expect, basis, ket2dm, operator_to_vector
import warnings
import os
import pickle
import matplotlib.pyplot as plt

_SIGMAS = [sigmax(), sigmay(), sigmaz()]

def X_operator(d: int) -> Qobj:
    """Generalized Pauli X operator in dimension d."""
    mat = np.roll(np.eye(d), 1, axis=1)
    return Qobj(mat)

def Z_operator(d: int) -> Qobj:
    """Generalized Pauli Z operator in dimension d."""
    omega = np.exp(2j*np.pi/d)
    return Qobj(np.diag([omega**j for j in range(d)]))

def generate_mubs(d: int, mubs_array: Sequence[int]) -> List[List[Qobj]]:
    """
    Generate projectors (MUBs) in dimension d for the specified indices in mubs_array.
    Works fully if d is prime (or prime power with this simple construction).

    Parameters
    ----------
    d : int
        Dimension (prime or prime power).
    mubs_array : sequence of int
        Sequence of MUB indices to generate, length must be d+1 and values 1 or 0.
        First index corresponds to computational basis, second to X basis, others to XZ^m.

    Returns:
    -------
        A list of lists of projectors [ [P_0,...,P_{d-1}], ..., MUBs ].
    """
    if len(mubs_array) != d+1:
        raise ValueError(f"Expected {d+1} MUB indices for dimension {d}, got {len(mubs_array)}")
    X = X_operator(d)
    Z = Z_operator(d)
    mub_projectors = []

    comp_basis = [basis(d, j) for j in range(d)]

    # 3) Eigenbases of XZ^m, m=1,...,d-1
    for idx, flag in enumerate(mubs_array):
        if not flag:
            continue
        if idx == 0:
            mub_projectors.append([ket2dm(ket) for ket in comp_basis])
        elif idx == 1:
            # X basis = eigenbasis of X Z^0
            _, evecs = X.eigenstates(output_type='kets')
            mub_projectors.append([ket2dm(v) for v in evecs])
        else:
            m = idx - 1           # <-- crucial fix
            U = X * (Z ** m)      # eigenbasis of XZ^m for m=1..d-1 # type: ignore
            _, evecs = U.eigenstates(output_type='kets')
            mub_projectors.append([ket2dm(v) for v in evecs])
    return mub_projectors

# --------- Higher dimensional (qudit) utilities ---------
def generalized_gell_mann(d: int) -> list[Qobj]:
    """Return an orthonormal (HS) set of traceless Hermitian generators for SU(d).

    Constructs the standard generalized Gell-Mann matrices:
    - Symmetric off-diagonal: |i><j| + |j><i|
    - Anti-symmetric off-diagonal: -i|i><j| + i|j><i|
    - Diagonal (d-1) traceless combinations.

    Parameters
    ----------
    d : int
        Dimension (>1).

    Returns
    -------
    list[Qobj]
        List of (d^2 - 1) traceless Hermitian Qobj with Tr(lambda_a lambda_b)=2 delta_{ab}.
    """
    if d <= 1:
        raise ValueError("d must be > 1")

    mats: list[Qobj] = []
    zero = Qobj(np.zeros((d, d), dtype=complex))
    # Off-diagonal: symmetric and anti-symmetric combinations of |i><j|
    for i in range(d):
        for j in range(i + 1, d):
            eij = basis(d, i) * basis(d, j).dag()  # type: ignore
            eji = basis(d, j) * basis(d, i).dag() # type: ignore
            mats.append(eij + eji)                 # symmetric: |i><j| + |j><i|
            mats.append(-1j * eij + 1j * eji)     # anti-symmetric: -i|i><j| + i|j><i|

    # Diagonal traceless combinations: ∑_{r=0}^{k-1} |r><r| - k |k><k| for k=1..d-1
    proj = lambda r: basis(d, r) * basis(d, r).dag() # type: ignore
    for k in range(1, d):
        diag = sum((proj(r) for r in range(k)), zero) - k * proj(k)
        mats.append(diag)

    # Normalize to Hilbert-Schmidt orthonormality Tr(lambda_a lambda_b)=2 δ_ab
    normed: list[Qobj] = []
    for M in mats:
        trMM = (M * M).tr().real # type: ignore
        if trMM <= 0:
            raise ValueError("Encountered non-positive self inner product in generator construction.")
        normed.append(M * np.sqrt(2.0 / trMM))

    return normed

def projector_to_qudit_bloch(P: Qobj, generators: Sequence[Qobj]) -> np.ndarray:
    """Map a rank-1 projector P in C^d to its (d^2-1)-dimensional Bloch vector.

    Uses the expansion  P = I/d + 1/2 * sum_a r_a lambda_a,  with Tr(lambda_a lambda_b)=2 delta_ab.
    Returns r = [r_a].
    """
    d = P.shape[0]
    I = qeye(d)
    coeffs = []
    for G in generators:
        coeffs.append(((P - I/d) @ G).tr().real)
    return np.array(coeffs, dtype=float)

def bloch_vector_to_projector(r: np.ndarray, generators: Sequence[Qobj]) -> Qobj:
    """Reconstruct a (candidate) rank-1 projector from Bloch vector components r.

    Returns the Hermitian matrix  P = I/d + 1/2 * sum_a r_a lambda_a.
    Caller should verify positivity / idempotence if needed.
    """
    r = np.asarray(r, dtype=float)
    dimension = generators[0].dims[0]
    d = 0
    for dim in dimension:
        d += dim  # type: ignore
    if len(generators) != d*d - 1:
        raise ValueError("Expected d^2-1 generators for d dimensional generators.")
    if r.shape != (len(generators),):
        raise ValueError("r must have length equal to number of generators or d^2-1 where d is the generator dimension.")
    I = qeye(dimension)
    P = I / d
    for ra, G in zip(r, generators):
        P = P + 0.5 * ra * G
    return P

def _is_hermitian_qobj(A: Qobj, atol: float = 1e-9) -> bool:
    return (A - A.dag()).norm() <= atol


def _is_psd_qobj(A: Qobj, atol: float = 1e-9) -> bool:
    if not _is_hermitian_qobj(A, atol=atol):
        return False
    # Use eigenvalue check for PSD
    vals = np.linalg.eigvalsh(A.full())
    return np.all(vals >= -atol) # type: ignore


def _dimension_from_povm(povm: Sequence[Qobj]) -> int:
    if len(povm) == 0:
        raise ValueError("Empty POVM.")
    d0 = povm[0].shape[0]
    for E in povm:
        if E.shape[0] != d0 or E.shape[1] != d0:
            raise ValueError("POVM elements must share the same square dimension.")
    return d0


def _normalize_direction(n: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    n = np.asarray(n, dtype=float).reshape(-1)
    if n.shape[0] != 3:
        raise ValueError("Direction must be a 3-vector.")
    l = _norm(n)
    if l <= atol:
        raise ValueError("Direction vector has near-zero length.")
    return n / l

def operator_double_ket(A: Qobj) -> Qobj:
    """Convert a square operator to its double-ket column-vector form |A⟩⟩.

    Parameters
    ----------
    A : Qobj
        Square operator to vectorize.

    Returns
    -------
    Qobj
        Column vector Qobj containing the operator entries in column-major
        order (shape (d*d, 1)).
    """
    d1, d2 = A.shape
    if d1 != d2:
        raise ValueError("Operator must be square.")
    d = d1
    vec = np.reshape(A.full(), (d*d, 1), order='C')  # column-stacking
    return Qobj(vec, dims=[[d, d], [1, 1]])

def complete_kraus_square(kraus_list: list[Qobj], clip_tol: float = 1e-12) -> list[Qobj]:
    """
    Given a list of square Kraus operators (Qobj, each d×d), return a TP-completed set by
    appending ONE extra Kraus. If Σ K†K ⪯ I: add K_extra = sqrt(I - Σ K†K).
    If Σ K†K ≻ I: globally rescale by 1/sqrt(λ_max(Σ K†K)) first, then add the completion.

    Args:
        kraus_list : list of Qobj, each with shape (d, d) and consistent dims.
        clip_tol   : tolerance for clipping small negative eigenvalues (numerical).

    Returns:
        list[Qobj] : possibly rescaled originals + [K_extra], now trace-preserving (up to clip_tol).
    """
    if not kraus_list:
        raise ValueError("kraus_list must be non-empty.")
    d_out, d_in = kraus_list[0].shape
    if d_out != d_in:
        raise ValueError("This function only supports SQUARE Kraus operators (d_out == d_in).")
    d = d_in

    # Shape/dims sanity
    for K in kraus_list:
        if K.shape != (d, d):
            raise ValueError("All Kraus must have identical square shape (d, d).")

    # 1) M = Σ K†K  (acts on input space)
    M = Qobj(np.zeros((d, d), dtype=complex), dims=[[d], [d]])
    for K in kraus_list:
        M += K.dag() * K # type: ignore
    # Hermitize (kill numerical skew)
    M = 0.5 * (M + M.dag())

    # 2) If super-normalized, rescale by s = λ_max(M)
    I = qeye(d)
    if float((M - I).eigenenergies().real.max()) > clip_tol:
        lam_max = float(M.eigenenergies().real.max())
        scale = 1.0 / np.sqrt(lam_max)
        kraus_list = [scale * K for K in kraus_list]
        M = (1.0 / lam_max) * M
        M = 0.5 * (M + M.dag())

    # 3) Δ = I - M  (should be PSD up to numerics). Build sqrt(Δ).
    Delta = I - M
    Delta = 0.5 * (Delta + Delta.dag())  # ensure Hermitian

    evals, evecs = Delta.eigenstates()
    # clip eigenvalues; throw if significantly negative
    ev = np.array([float(w.real) for w in evals])
    if ev.min() < -clip_tol:
        raise ValueError("Completion failed: Δ has significantly negative eigenvalues; check inputs.")
    ev_clipped = np.clip(ev, 0.0, None)

    # sqrt(Δ) in eigenbasis
    sqrtDelta = Qobj(np.zeros((d, d), dtype=complex), dims=[[d], [d]])
    for i in range(len(evals)):
        sqrtDelta += np.sqrt(ev_clipped[i]) * (evecs[i] * evecs[i].dag())
    sqrtDelta = Qobj(sqrtDelta.full(), dims=[[d], [d]])  # ensure clean dims

    # 4) Single extra Kraus
    K_extra = sqrtDelta

    return kraus_list + [K_extra]


class CompatibilityMeasure:
    """Utilities for measuring incompatibility between quantum measurements.

    Parameters
    ----------
    atol : float
        Absolute numerical tolerance used for PSD/Hermitian checks.
    method : str
        Default disturbance computation method; one of {'analytical',
        'numerical', 'experimental'}.
    rng : seed | np.random.Generator | None
        RNG seed or generator used for stochastic behavior.

    Main methods
    ------------
    - pauli_operators(): return Pauli X, Y, Z as Qobj
    - is_povm(povm): quick validity checks for POVM elements
    - mutual_eigenspace_disturbance(povm1, povm2): compute disturbance scalar
    - incompatibility_distance_matrix(povms): pairwise distance matrix
    - noisy_povm_with_kraus(P, lam, ...): build noisy POVM and Kraus ops
    - bloch_direction_from_projective(povm): recover Bloch axis of a binary
      projective qubit POVM.
    """

    def __init__(self, *, atol: float = 1e-7, method: str = "analytical", rng=None):
        self.atol = float(atol)
        self.method = method
        self.rng = np.random.default_rng(rng) 

    @staticmethod
    def pauli_operators() -> Tuple[Qobj, Qobj, Qobj]:
        """Return the three single-qubit Pauli operators as Qobj.

        Returns
        -------
        (Qobj, Qobj, Qobj)
            Pauli X, Y, Z (in that order).
        """
        return sigmax(), sigmay(), sigmaz()

    # NOTE: projective_qubit_povm_from_axis moved to ClusteringToolkit per refactor

    def is_povm(self, povm: Sequence[Qobj]) -> bool:
        """Quickly test whether `povm` is a valid POVM.

        Checks that elements are square Qobj of the same dimension, positive
        semidefinite (within `self.atol`), and that they sum to the identity
        (within `self.atol`). Returns True when all checks pass.
        """
        if len(povm) == 0:
            return False
        d = _dimension_from_povm(povm)
        if any(not _is_psd_qobj(E, atol=self.atol) for E in povm):
            return False
        S = sum(povm[1:], povm[0] * 0)
        for E in povm:
            S = S + E
        return (S - qeye(d)).norm() <= self.atol

    def mutual_eigenspace_disturbance(self, povm1: Sequence[Qobj], povm2: Sequence[Qobj], *, state: Qobj | None = None, debug: bool = False) -> float:
        """Compute a disturbance/incompatibility scalar between two POVMs.

        Three algorithms are supported via `self.method`:
        - 'analytical': closed form formula for binary qubit projective POVMs,
          implemented via nested traces.
        - 'numerical': constructs superoperators and evaluates overlaps
          numerically (suitable for higher-dimensional POVMs).
        - 'experimental': a commutator-based heuristic used for experiments.

        Parameters
        ----------
        povm1, povm2 : sequence of Qobj
            The POVMs to compare (must have same element shape/dimension).
        state : Qobj, optional
            If provided, the state used in the analytical/numerical expressions;
            otherwise the maximally mixed state is used.

        Returns
        -------
        float
            A nonnegative scalar quantifying mutual eigenspace disturbance.
        """
        method = self.method
        if povm1[0].shape != povm2[0].shape:
            raise ValueError("POVMs must have the same dimension.")
        if state is None:
            dimensions = povm1[0].dims
            state = qeye(dimensions[0]) / povm1[0].shape[0]
        if state.isket:
            state = ket2dm(state)
        # force trace-1, keep Hermiticity check
        state = 0.5*(state + state.dag())
        if not _is_hermitian_qobj(state, atol=self.atol):
            raise ValueError("State must be Hermitian.")
        tr = state.tr().real
        if abs(tr) <= self.atol:
            raise ValueError("State has ~zero trace.")
        state = state / tr

        if method == "analytical":
            summation = 0.0
            for E in povm1:
                for F in povm2:
                    summation += (state * E * F * E * F).tr() # type: ignore
            return float(np.sqrt(1 - (summation.real))) 

        elif method == "numerical":
            # Build superoperator C_tilda = sum_i E_i ⊗ E_i^*
            C_tilda = tensor(povm1[0], povm1[0].conj())
            for E in povm1[1:]:
                C_tilda = C_tilda + tensor(E, E.conj())
            dimensions = C_tilda.dims  # type: ignore
            D_list = [operator_double_ket(F) for F in povm2]
            # D = ∑ |F⟩⟩⟨⟨F|
            first = D_list[0]
            D = first * first.dag()  # type: ignore
            for dvec in D_list[1:]:
                D = D + dvec * dvec.dag()  # type: ignore
            D.dims = dimensions  # type: ignore
            # print(type(D))
            # robust trace evaluation with clipping to avoid small negative rounding errors inside sqrt
            d_dim = int(povm1[0].shape[0])
            M = tensor(qeye(d_dim), state.trans())
            M.dims = dimensions  # type: ignore
            # print(type(M))

            trval = (D * C_tilda * M).tr().real  # type: ignore
            if debug:
                # Validate POVM properties
                d = d_dim
                I = qeye(d)
                S1 = 0; S2 = 0
                for E in povm1: S1 = S1 + E
                for F in povm2: S2 = S2 + F
                res1 = (S1 - I).norm(); res2 = (S2 - I).norm()
                min_eig_E = min((np.min(np.linalg.eigvalsh(E.full())) for E in povm1))
                min_eig_F = min((np.min(np.linalg.eigvalsh(F.full())) for F in povm2))
                warnings.warn(f"[NCOM-debug] sum(povm1)-I norm={res1:.3e}, min eig(E)={min_eig_E:.3e}; sum(povm2)-I norm={res2:.3e}, min eig(F)={min_eig_F:.3e}")

                # Inspect superoperator PSD-ness
                C_psd = np.all(np.linalg.eigvalsh(C_tilda.full()) >= -1e-10)
                D_psd = np.all(np.linalg.eigvalsh(D.full()) >= -1e-10)
                warnings.warn(f"[NCOM-debug] PSD checks: C_tilda={C_psd}, D={D_psd}")

                # Decompose trval into contributions per F
                contribs = []
                for dvec in D_list:
                    val = (dvec.dag() * C_tilda * M * dvec).real  # type: ignore
                    contribs.append(float(val))
                contribs = np.array(contribs, dtype=float)
                warnings.warn(f"[NCOM-debug] trval={trval:.6g}, per-F min={contribs.min():.6g}, max={contribs.max():.6g}, sum={contribs.sum():.6g}")

            # Clamp overlap to [0,1] within numerical tolerance
            overlap = float(trval)
            if overlap < -self.atol or overlap > 1 + self.atol:
                warnings.warn(f"[NCOM] Overlap out of [0,1]: {overlap:.6g}. Clipping to [0,1].")
            overlap = float(np.clip(overlap, 0.0, 1.0))
            arg = 1.0 - overlap
            if arg < 0.0 and arg > -self.atol:
                arg = 0.0
            if arg < 0.0:
                raise ValueError(f"Negative value under sqrt beyond tolerance: {arg}")
            NCOM = np.sqrt(arg)
            return float(NCOM)
        elif method == "experimental":
            summation = 0.0
            for E in povm1:
                for F in povm2:
                    summation += (state * commutator(E, F) * commutator(E, F).dag()).tr()  # type: ignore
            return float(np.sqrt(summation / 2))
        else:
            raise ValueError(f"Unknown method: {method}")

    def incompatibility_distance_matrix(self, povms: Sequence[Sequence[Qobj]]) -> np.ndarray:
        """Build the symmetric pairwise disturbance/distance matrix for `povms`.

        Returns an (n x n) numpy array with zeros on the diagonal and the
        disturbance metric at (i, j) for i != j.
        """
        n = len(povms)
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                d = self.mutual_eigenspace_disturbance(povms[i], povms[j], debug=False)
                D[i, j] = d
                D[j, i] = d
        return D

    # ---------- Convenience wrappers ----------
    @staticmethod
    def bloch_direction_from_projective(povm: Sequence[Qobj]) -> np.ndarray:
        """Recover the Bloch vector n from a binary projective qubit POVM.

        Expects `povm = [E_plus, E_minus]` where E_plus = (I + n·σ)/2. The
        returned 3-vector `n` is normalized and gives the measurement axis.
        """
        if len(povm) != 2:
            raise ValueError("Expected binary POVM [E_plus, E_minus].")
        d = _dimension_from_povm(povm)
        if d != 2:
            raise ValueError("Only qubit POVMs supported for this method.")
        I = qeye(2)
        n_sigma = 2 * povm[0] - I  # equals n · σ
        comps = np.array([(n_sigma * _SIGMAS[i]).tr().real for i in range(3)]) / 2.0 # type: ignore


        return comps

class ClusteringToolkit:
    """Sampling and clustering utilities for observables.

    The toolkit provides instance-based RNG and convenient defaults for
    sampling Bloch vectors within a cone, converting Bloch vectors into
    binary qubit projective POVMs, generating noisy datasets, computing
    pairwise incompatibility distances using a linked :class:`CompatibilityMeasure`,
    and clustering those distances via k-medoids, k-means, or HDBSCAN.

    Parameters
    ----------
    rng : seed | np.random.Generator | None
        RNG or seed for reproducible sampling.
    default_cluster_method : str
        Default clustering algorithm to use ('k-medoids', 'k-means', 'hdbscan').
    init : str
        Initialization strategy for k-medoids/k-means ('kmeans++', 'linear++', 'random').
    n_points : int
        Default number of data points sampled per cluster/axis.
    spread_angle : float
        Default cone opening angle in degrees for sampling Bloch directions.
    cm : CompatibilityMeasure | None
        CompatibilityMeasure instance used for noisy POVM generation and distance
        computations; if None a default instance will be created.
    mubs_array : sequence of int | None
        Sequence of MUB indices to generate, length must be d+1 and values 1 or 0.
        First index corresponds to computational basis, second to X basis, others to XZ^m.
    dimensions : int
        Dimension of the objects to cluster (used if `axes` is None).
    splits : int
        Number of Kraus operators to generate for noisy POVMs.


    Main methods
    ------------
    - sample_unit_vectors_cone(n, spread_angle, axis): sample Bloch directions in a cone
    - projective_qubit_povm_from_axis(n): build a binary projective POVM from a Bloch axis
    - generate_noisy_dataset(etas, n_clusters, ...): sample vectors, add noise, compute distances
    - cluster_from_distance(D, n_clusters): cluster objects given a distance matrix

    """

    def __init__(self, *, rng=None, cluster_method: str = "k-medoids", init: str = "kmeans++",
                 n_points: int = 100, spread_angle: float = 11.25, cm: CompatibilityMeasure | None = None, n_clusters: int = 2,
                 mubs_array: Sequence[int] | None = None, dimensions: int = 2, splits: int = 10):
        """Instance-based clustering toolkit.

        Parameters added by request:
        - n_points: number of data points for each cluster (per axis)
        - spread_angle: cone opening angle in degrees
        - cm: CompatibilityMeasure instance to generate POVMs
        - axes: iterable of 3-vector axes for clusters
        - splits: Splits for generation of Kraus operators for noisy POVMs
        """
        self.rng = np.random.default_rng(rng)
        self.cluster_method = cluster_method
        self.init = init
        self.splits = splits
        self.n_points = int(n_points)
        self.spread_angle = float(spread_angle)
        self.n_clusters = int(n_clusters)
        if mubs_array is None:
            self.mubs_array = np.asarray([1] * n_clusters + [0] * (dimensions + 1 - n_clusters), dtype=int)
        else:
            self.mubs_array = np.asarray(mubs_array, dtype=int)
        self.dimensions = dimensions
        if dimensions != len(self.mubs_array) - 1:
            raise ValueError("Length of mubs_array must be dimensions + 1.")
        if cm is None:
            self.cm = CompatibilityMeasure()
        else:
            self.cm = cm

    def sample_qudit_povms_cone(self) -> list[list[Qobj]]:
        """Approximate a "cone" around a basis by random small-unitary perturbations.

        Implementation: For every set of basis projectors apply a random unitary
        U = exp(i * ε H) with H a random traceless Hermitian (Gaussian) and ε sampled
        uniformly in [0, θ], where θ = spread_angle (radians).
        """

        theta = np.deg2rad(self.spread_angle)
        list_mubs = generate_mubs(d = self.dimensions, mubs_array = list(self.mubs_array))
        gens = generalized_gell_mann(self.dimensions)
        pvms = []
        for pvm in list_mubs:
            for _ in range(self.n_points):
                eps = self.rng.uniform(0.0, theta)
                projs = []
                for P in pvm:
                    # random traceless Hermitian H = sum_a c_a G_a with c_a ~ N(0,1)
                    coeffs = self.rng.normal(size=len(gens))
                    H = 0
                    for c, G in zip(coeffs, gens):
                        H = H + c * G
                    # scale H to have Fro norm 1 then apply perturbation
                    fro = np.sqrt((H*H).tr().real)
                    if fro > 1e-14:
                        H = (1.0/fro) * H
                    U = (1j * eps * H).expm()
                    Pp = U * P * U.dag()
                    # Re-project via leading eigenvector to keep it rank-1
                    eigenvals, eigenvecs = Pp.eigenstates('kets')
                    v = eigenvecs[np.argmax(eigenvals)]
                    P_clean = v * v.dag() 
                    projs.append(P_clean)
                pvms.append(projs)
        return pvms
    
    # ----- Dataset preparation (instance-based) -----

    def noisy_povm_with_kraus(self, P, lam, p=None, *, random_split=False, rng=None):
        """
        Build the noisy measurement {E_i} and a Kraus realization {N_{i,j}} from a
        projective POVM {P_i} using the isotropic-noise model:
            E_i = (1 - λ) P_i + λ p_i I,   0 ≤ λ ≤ 1,   p ∈ Δ^{k-1}.
        Kraus operators per outcome i are constructed as
            N_{i,j} = √(a_{ij} P_i  + b_{ij} I)        for j = 1..splits
        with nonnegative weights obeying
            ∑_j a_{ij} = 1 - λ,     ∑_j b_{ij} = λ p_i.
        Weights are split either uniformly or by a Dirichlet draw (random_split=True).

        Parameters
        ----------
        P : list[Qobj]
            Projective POVM {P_i} on C^d with ∑_i P_i = I (each Qobj is d×d).
        lam : float
            Noise strength λ ∈ [0,1].
        p : array_like | None
            Outcome probabilities (length k). If None, p_i = 1/k.
        splits : int
            Number of Kraus terms carrying P_i and I, respectively (≥1).
        random_split : bool
            If True, weights are Dirichlet-distributed; else equal split.
        rng : None | int | np.random.Generator
            Seed/Generator for reproducible splits.

        Returns
        -------
        E : list[Qobj]
            Noisy POVM elements {E_i}. Enforces ∑_i E_i = I by a final
            Hermitian correction on the last element.
        N : list[list[Qobj]]
            Kraus operators per outcome, satisfying  E_i = ∑_j N_{i,j}† N_{i,j}.
        """
        if not P:
            raise ValueError("Empty POVM.")
        d = P[0].shape[0]
        if any(Q.shape != (d, d) for Q in P):
            raise ValueError("All POVM elements must be square Qobj of the same dimension.")
        # use a proper identity of dimension d
        I = qeye(d)

        lam = float(lam)
        if lam < -1e-12 or lam > 1 + 1e-12:
            raise ValueError("λ ∈ [0,1].")

        k = len(P)

        rng = np.random.default_rng(rng)

        if p is None:
            p = rng.random(k)
            p = p / p.sum()
        else:
            p = np.asarray(p, dtype=float)
            if p.shape != (k,):
                raise ValueError("p must have length k.")
            s = p.sum()
            if not np.isfinite(s) or s <= 0:
                raise ValueError("Invalid probability vector p.")
            p = p / s

        if self.splits < 1:
            raise ValueError("splits ≥ 1.")

        def split_mass(total, m):
            if m == 1:
                return np.array([total])
            w = rng.dirichlet(np.ones(m)) if random_split else np.full(m, 1.0 / m)
            return total * w

        E, N = [], []
        for i in range(k):
            a_parts = split_mass(1.0 - lam, self.splits)
            b_parts = split_mass(lam * p[i], self.splits)

            Ei = (1 - lam) * P[i] + (lam * p[i]) * I
            E.append(Ei)

            Ni = []
            for aj, bj in zip(a_parts, b_parts):
                # sqrt(a P_i + b I) = sqrt(a+b) P_i + sqrt(b) (I - P_i)
                K = np.sqrt(aj + bj) * P[i] + np.sqrt(bj) * (I - P[i])
                Ni.append(K)
            N.append(Ni)

        # Ensure completeness sum_i E_i = I (apply Hermitian correction to last element if needed)
        zero_q = 0 * P[0]
        S = zero_q
        for Ei in E:
            S = S + Ei
        corr = I - S
        # make Hermitian and apply small correction to last element
        corr = 0.5 * (corr + corr.dag())
        if corr.norm() > 1e-12:
            E[-1] = E[-1] + corr

        # Build list of all Kraus and ensure ∑ K†K = I (append sqrt of missing positive part)
        all_kraus = [K for Ni in N for K in Ni]
        all_kraus = complete_kraus_square(all_kraus, clip_tol=1e-12)
        S = zero_q
        for Ki in all_kraus:
            S = S + (Ki.dag() * Ki) # type: ignore
        if (S - I).norm() > 1e-6:
            raise ValueError("Numerical error: Kraus operators do not sum to identity.")
        return E, all_kraus    

    def prepare_povm_dataset(self, *, etas: list[float] | None = None, spread_angle: float | None = None, cone_seeds: tuple[int, ...] | None = None, base_seed: int = 12345, noisy: bool = True) -> dict:
        """Prepare POVM datasets (projective or isotropically noisy) around the instance's `axes`.

        Parameters
        ----------
        etas : list[float] | None
            Noise scaling factors; if None and `noisy` is False, uses [0.0].
        spread_angle : float | None
            Cone opening angle in degrees; if None uses ``self.spread_angle``.
        cone_seeds : tuple[int, ...]
            Seeds for per-axis cone samplers.
        base_seed : int
            Seed for base noise magnitudes.
        noisy : bool
            If True generate noisy POVMs; else projective POVMs.

        Returns
        -------
        dict
            Dataset with keys 'meta', 'pvms', 'per_eta'.
        """
        if noisy and (not etas):
            raise ValueError("At least one eta must be provided for noisy dataset generation.")
        etas = [0.0] if (etas is None) else list(etas)
        spread_angle = float(self.spread_angle) if spread_angle is None else float(spread_angle)
        if spread_angle < 0.0 or spread_angle > 180.0:
            raise ValueError("spread_angle must be in [0, 180].")
        if cone_seeds is None:
            cone_seeds = tuple(self.rng.integers(0, 2**31 - 1, size=self.n_clusters))
        if len(cone_seeds) != self.n_clusters:
            raise ValueError("Number of cone_seeds must match number of clusters/axes.")
        pvms = self.sample_qudit_povms_cone()  # list of list of Qobj; length N
        
        rng = np.random.default_rng(base_seed)
        base_Rl = rng.uniform(0.0, 1.0, size=len(pvms))  # one base noise level per POVM element    

        per_eta: dict[float, dict] = {}
        for eta in etas:
            lam_vec = (eta * base_Rl) if noisy else np.zeros_like(base_Rl)
            noisy_kraus, noisy_E, noisy_arrays = [], [], []
            if noisy:
                for i, proj in enumerate(pvms):
                    E_list, all_kraus = self.noisy_povm_with_kraus(proj, lam=float(lam_vec[i]))
                    noisy_kraus.append(all_kraus) 
                    noisy_E.append(E_list)
                    noisy_arrays.append([Ei.full() for Ei in E_list])
            else:
                for proj in pvms:
                    noisy_kraus.append(proj)                      # trivial Kraus
                    noisy_E.append(proj)
                    noisy_arrays.append([Ei.full() for Ei in proj])
            per_eta[float(eta)] = {
                "lam_vec": lam_vec,
                "noisy_kraus": noisy_kraus,                      # list[list[Qobj]]
                "noisy_E": noisy_E,                              # list[list[Qobj]]
                "noisy_E_array": np.asarray(noisy_arrays, dtype=complex)
            }

        return {
            "meta": {
                "spread_angle": spread_angle, "mubs_array": self.mubs_array.tolist(), 
                "n_points": int(self.n_points),
                "etas": [float(e) for e in etas], "cone_seeds": tuple(int(s) for s in cone_seeds),
                "base_seed": int(base_seed), "noisy": bool(noisy)
            },
            "pvms": pvms,
            "per_eta": per_eta
        }
    
    # ----- Clustering -----
    def cluster_from_distance(self, D, n_clusters: int, *, method: str | None = None, n_init: int = 10, max_iter: int = 300, tol: float = 1e-4, random_state=None):
        """Cluster items given a precomputed (symmetric) distance matrix D.

        Parameters
        ----------
        D : array-like, shape (n, n)
            Symmetric precomputed distance matrix.
        n_clusters : int
            Number of clusters to produce.
        method : str, optional
            Clustering method override; if None the instance default is used.
        n_init, max_iter, tol, random_state : see implementation

        Returns
        -------
        numpy.ndarray
            Integer label array of length n assigning each item to a cluster.
        """
        method = self.cluster_method if method is None else method
        D = np.asarray(D, dtype=float)
        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError("D must be square (n x n).")
        n = D.shape[0]
        if not (1 <= n_clusters <= n):
            raise ValueError("n_clusters out of range")
        if np.any(D < -1e-12):
            raise ValueError("Distances must be non-negative.")
        if not np.allclose(D, D.T, atol=1e-10, rtol=0):
            D = 0.5 * (D + D.T)
        np.fill_diagonal(D, 0.0)
        rng = np.random.default_rng(random_state) if random_state is not None else self.rng
        if method.lower() in {"k-medoids", "kmedoids", "pam"}:
            best_labels, best_cost = None, np.inf
            for _ in range(max(1, n_init)):
                labels, medoids, cost = self._pam_kmedoids(D, n_clusters, rng, max_iter, init=self.init)
                if cost < best_cost - 1e-12:
                    best_labels, best_cost = labels, cost
            return best_labels
        elif method.lower() in {"k-means", "kmeans"}:
            X = self._classical_mds(D)
            best_labels, best_inertia = None, np.inf
            for _ in range(max(1, n_init)):
                labels, inertia = self._kmeans_lloyd(X, n_clusters, rng, max_iter, tol)
                if inertia < best_inertia - 1e-12:
                    best_labels, best_inertia = labels, inertia
            return best_labels
        elif method.lower() == "hdbscan":
            try:
                import hdbscan
            except ImportError as e:
                raise ImportError("hdbscan package is required for method='hdbscan'.") from e
            clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='precomputed', cluster_selection_method='eom')
            return clusterer.fit_predict(D)
        else:
            raise ValueError("Unsupported clustering method")

    # helper methods (adapted)
    def _pam_kmedoids(self, D, k, rng, max_iter, init="kmeans++"):
        n = D.shape[0]
        medoids = self._init_medoids(D, k, rng, mode=init)
        prev_cost = np.inf
        for _ in range(max_iter):
            distances_to_medoids = D[:, medoids]
            labels = np.argmin(distances_to_medoids, axis=1)
            cost = distances_to_medoids[np.arange(n), labels].sum()
            new_medoids = medoids.copy()
            for j in range(k):
                idx = np.where(labels == j)[0]
                if idx.size == 0:
                    far_idx = np.argmax(np.min(D[:, medoids], axis=1))
                    new_medoids[j] = far_idx
                    continue
                subD = D[np.ix_(idx, idx)]
                within_sums = subD.sum(axis=1)
                new_medoids[j] = idx[np.argmin(within_sums)]
            medoids = np.unique(new_medoids)
            while medoids.size < k:
                dmin = np.min(D[:, medoids], axis=1) if medoids.size > 0 else D.mean(axis=1)
                cand = int(np.argmax(dmin))
                if cand in medoids:
                    remaining = np.setdiff1d(np.arange(n), medoids)
                    cand = int(rng.choice(remaining))
                medoids = np.append(medoids, cand)
            if prev_cost - cost <= 1e-12:
                break
            prev_cost = cost
        distances_to_medoids = D[:, medoids]
        labels = np.argmin(distances_to_medoids, axis=1)
        cost = distances_to_medoids[np.arange(n), labels].sum()
        return labels.astype(int), medoids.astype(int), float(cost)

    @staticmethod
    def _init_medoids(D, k, rng, mode="kmeans++"):
        n = D.shape[0]
        mode = mode.lower()
        if mode == "random":
            return np.array(rng.choice(n, size=k, replace=False), dtype=int)
        medoids = [int(rng.integers(0, n))]
        for _ in range(1, k):
            dmin = np.min(D[:, medoids], axis=1)
            dmin[medoids] = 0.0
            if mode == "kmeans++":
                weights = dmin * dmin
            elif mode in {"linear++", "kmedoids++"}:
                weights = dmin
            else:
                raise ValueError("init must be one of {'kmeans++','linear++','random'}.")
            total = weights.sum()
            if not np.isfinite(total) or total <= 1e-20:
                choices = np.setdiff1d(np.arange(n), np.array(medoids))
                medoids.append(int(rng.choice(choices)))
            else:
                probs = weights / total
                idx = int(rng.choice(n, p=probs))
                if idx in medoids:
                    choices = np.setdiff1d(np.arange(n), np.array(medoids))
                    idx = int(rng.choice(choices))
                medoids.append(idx)
        return np.array(medoids, dtype=int)

    @staticmethod
    def _classical_mds(D):
        n = D.shape[0]
        D2 = D ** 2
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ D2 @ J
        B = 0.5 * (B + B.T)
        w, V = np.linalg.eigh(B)
        idx = np.argsort(w)[::-1]
        w = w[idx]; V = V[:, idx]
        pos = w > (1e-12 * w[0] if w[0] > 0 else 1e-12)
        if not np.any(pos):
            return np.zeros((n, 1), dtype=float)
        Lhalf = np.sqrt(w[pos])
        return V[:, pos] * Lhalf

    def _kmeans_lloyd(self, X, k, rng, max_iter, tol):
        if not np.isfinite(X).all():
            raise ValueError("Non-finite embedding; use k-medoids.")
        n, d = X.shape
        centers = np.empty((k, d), dtype=X.dtype)
        centers[0] = X[rng.integers(0, n)]
        closest_sq = self._row_min_sqdist(X, centers[0:1])
        closest_sq = np.clip(closest_sq, 0.0, None)
        for i in range(1, k):
            weights = np.clip(closest_sq, 0.0, None)
            total = weights.sum()
            if not np.isfinite(weights).all() or total <= 1e-20:
                idx = int(rng.integers(0, n))
            else:
                probs = weights / total
                idx = int(rng.choice(n, p=probs))
            centers[i] = X[idx]
            new_d2 = self._row_min_sqdist(X, centers[i:i+1])
            closest_sq = np.minimum(closest_sq, new_d2)
            closest_sq = np.clip(closest_sq, 0.0, None)
        prev_inertia = np.inf
        for _ in range(max_iter):
            d2 = self._pair_sqdist(X, centers)
            labels = np.argmin(d2, axis=1)
            inertia = d2[np.arange(n), labels].sum()
            new_centers = np.zeros_like(centers)
            counts = np.bincount(labels, minlength=k)
            for j in range(k):
                if counts[j] > 0:
                    new_centers[j] = X[labels == j].mean(axis=0)
                else:
                    far_idx = np.argmax(np.min(d2, axis=1))
                    new_centers[j] = X[far_idx]
            centers = new_centers
            if prev_inertia - inertia <= tol * max(1.0, prev_inertia):
                break
            prev_inertia = inertia
        d2 = self._pair_sqdist(X, centers)
        labels = np.argmin(d2, axis=1)
        inertia = d2[np.arange(n), labels].sum()
        return labels.astype(int), float(inertia)

    @staticmethod
    def _pair_sqdist(X, C):
        X2 = np.sum(X * X, axis=1, keepdims=True)
        C2 = np.sum(C * C, axis=1, keepdims=True).T
        return np.maximum(X2 + C2 - 2.0 * (X @ C.T), 0.0)

    def _row_min_sqdist(self, X, C):
        return np.min(self._pair_sqdist(X, C), axis=1)


    # ----- Clustering attachment (distance-based) -----
    def cluster_povm_dataset(self, dataset: dict, *, n_clusters: int | None = None,
                            methods: list[str] = ["kmedoids"]) -> dict:
        """Attach incompatibility distances and clustering labels to a prepared dataset.

        Parameters
        ----------
        dataset : dict
            Output of `prepare_povm_dataset`.
        n_clusters : int
            Target number of clusters for partitioning methods.
        methods : list[str]
            Clustering methods to run via `self.cluster_from_distance`.

        Returns
        -------
        dict
            The input dataset augmented in-place with 'D' and 'labels' per eta.
        """
        if "per_eta" not in dataset:
            raise ValueError("Dataset missing 'per_eta'. Pass the output of prepare_povm_dataset.")
        n_clusters = self.n_clusters if n_clusters is None else int(n_clusters)
        for eta_key, blob in dataset["per_eta"].items():
            D = self.cm.incompatibility_distance_matrix(blob["noisy_kraus"])
            # print(D)
            labels = {m: self.cluster_from_distance(D, n_clusters=n_clusters, method=m) for m in methods}
            blob["D"], blob["labels"] = D, labels
        return dataset

    # ----- Evaluation histogram (cluster accuracy per ordered block) -----
    def plot_cluster_accuracy_hist(self, dataset: dict, *, 
                                   n_clusters: int | None = None, method: str = "kmedoids",
                                etas: list[float] | None = None, annotate: bool = True, savepath: str | None = None) -> dict[float, np.ndarray]:
        """Plot, for each eta, a histogram (n_clusters bars) of % correctly clustered POVMs per ordered block.

        Logic
        -----
        For a given eta, take the list order of instances (noisy_kraus). Partition it into `n_clusters`
        consecutive, equal-size blocks. For each block, find the majority cluster label assigned by `method`
        and report the fraction (in %) of instances in that block carrying that majority label. Bars are
        in the existing order of blocks. One histogram is produced per eta.

        Parameters
        ----------
        dataset : dict
            Output of `prepare_povm_dataset` (and then `cluster_povm_dataset`).
        n_clusters : int
            Number of desired equal-size blocks and the number of bars in the histogram.
        method : str
            Key inside dataset['per_eta'][eta]['labels'][method] to use (e.g., "kmeans", "kmedoids", "hdbscan").
        etas : list[float] | None
            Which eta values to plot. If None, uses all etas present in dataset (sorted numerically).
        annotate : bool
            If True, writes the percentage value above each bar.

        Returns
        -------
        dict[float, np.ndarray]
            Mapping eta -> array of shape (n_clusters,) with the percentages per block.
        """
        if "per_eta" not in dataset:
            raise ValueError("Dataset missing 'per_eta'. Pass the output of prepare_povm_dataset/cluster_povm_dataset.")
        n_clusters = self.n_clusters if n_clusters is None else int(n_clusters)
        if savepath is None:
            savepath = f"Plots&Data/MLQS/unnamed.png"

        # resolve eta list in numeric sorted order
        per_eta = dataset["per_eta"]
        if etas is None:
            eta_list = sorted([float(k) for k in per_eta.keys()])
        else:
            # keep user-provided order but check existence
            eta_list = []
            for e in etas:
                # find matching key robustly
                key = next((k for k in per_eta if abs(float(k) - float(e)) < 1e-12), None)
                if key is None:
                    raise KeyError(f"eta={e} not found in dataset['per_eta'].")
                eta_list.append(float(key))

        results: dict[float, np.ndarray] = {}

        # compute percentages for all etas first
        for e in eta_list:
            # resolve stored key (string/float) robustly
            k_store = next(k for k in per_eta if abs(float(k) - e) < 1e-12)
            blob = per_eta[k_store]

            if "labels" not in blob or method not in blob["labels"]:
                raise KeyError(f"Labels for method='{method}' not found at eta={e}. Run cluster_povm_dataset first.")
            labels = np.asarray(blob["labels"][method])
            N = labels.shape[0]

            # enforce equal-size partition
            if N % n_clusters != 0:
                raise ValueError(f"Cannot partition {N} instances into {n_clusters} equal blocks (remainder {N % n_clusters}).")
            m = N // n_clusters

            pct = np.zeros(n_clusters, dtype=float) # percentage per block
            for g in range(n_clusters):
                sl = slice(g * m, (g + 1) * m)
                block = labels[sl]
                vals, counts = np.unique(block, return_counts=True)
                maj = vals[np.argmax(counts)]
                correct = np.count_nonzero(block == maj)
                pct[g] = 100.0 * correct / m
            results[e] = pct

        # --- plotting: single figure with 3 columns and as many rows as necessary ---
        n_eta = len(eta_list)
        cols = 3
        rows = int(np.ceil(n_eta / cols)) if n_eta > 0 else 1
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), constrained_layout=True)
        # flatten axes for easy indexing
        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        else:
            axes_flat = np.array([axes])

        cmap = plt.cm.get_cmap("tab10")
        for idx, e in enumerate(eta_list):
            ax = axes_flat[idx]
            pct = results[e]
            x = np.arange(n_clusters)
            colors = cmap(np.linspace(0.0, 1.0, n_clusters))
            bars = ax.bar(x, pct, width=0.8, color=colors)
            ax.set_xticks(x)
            ax.set_ylim(0.0, 100.0)
            ax.set_title(f"eta={e} ({method})", pad = 20)
            if annotate:
                for rect, val in zip(bars, pct):
                    ax.text(rect.get_x() + rect.get_width()/2.0, val + 1.5, f"{val:.1f}\\%", fontsize=9)
        # hide any unused axes
        for j in range(n_eta, rows * cols):
            axes_flat[j].axis("off")
        fig.supxlabel("Cluster index")
        fig.supylabel("Correctly clustered (\\%)")

        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=300)
        plt.show()
        # plt.close(fig)

        return results

    # ----- Visualizing higher dimensional POVMs (Bloch-slice rendering) -----
    def plot_povms(self, dataset: dict, *, eta: float,
                which_instances: slice | list[int] | None = None,
                subspace_pairs: list[tuple[int, int]] | None = None):
        """Render POVM effects on Bloch spheres (qubit) or SU(2) subspace slices (qudit).

        Parameters
        ----------
        dataset : dict
            Output of `prepare_povm_dataset`.
        eta : float
            Which noise scale to visualize.
        which_instances : slice | list[int] | None
            Subset of instances to plot; default plots all.
        subspace_pairs : list[tuple[int,int]] | None
            For d>2, list of level pairs (i,j) defining 2D subspaces to render.

        Returns
        -------
        Bloch | list[Bloch]
            A Bloch object (d=2) or a list of Bloch objects (one per (i,j) slice).
        """
        # resolve eta key robustly
        per_eta = dataset["per_eta"]
        eta_key = next((k for k in per_eta if abs(float(k) - float(eta)) < 1e-12), None)
        if eta_key is None:
            raise KeyError(f"eta={eta} not in dataset['per_eta'].")

        povms = per_eta[eta_key]["noisy_E"]
        N, d = len(povms), povms[0][0].shape[0]
        idxs = range(N) if which_instances is None else (range(N)[which_instances] if isinstance(which_instances, slice) else which_instances)

        def _bloch_vec_2x2(E2: Qobj) -> np.ndarray:
            tr = E2.tr()
            if abs(tr) < 1e-15: return np.zeros(3)
            rho = (E2 / tr + (E2 / tr).dag()) * 0.5
            return np.array([(rho * sigmax()).tr().real, (rho * sigmay()).tr().real, (rho * sigmaz()).tr().real]) # type: ignore

        def _proj_to_ij(E: Qobj, i: int, j: int) -> Qobj:
            P = Qobj(np.vstack([basis(d, i).full().T, basis(d, j).full().T]))  # 2×d
            return 0.5 * (P * E * P.dag() + (P * E * P.dag()).dag()) # type: ignore

        if d == 2:
            b = Bloch()
            for k in idxs:
                for Ei in povms[k]:
                    b.add_points(_bloch_vec_2x2(Ei))
            b.make_sphere()
            return b

        if not subspace_pairs:
            raise ValueError("For d>2, provide subspace_pairs=[(i,j), ...] to define SU(2) slices.")
        outs = []
        for (i, j) in subspace_pairs:
            b = Bloch()
            for k in idxs:
                for Ei in povms[k]:
                    b.add_points(_bloch_vec_2x2(_proj_to_ij(Ei, i, j)))
            b.make_sphere()
            outs.append(b)
        return outs

    def save_dataset(self, dataset, path_npz="Plots&Data/MLQS/noisy_obs_dataset.npz", path_pkl="Plots&Data/MLQS/noisy_obs_dataset.pkl"):
        os.makedirs(os.path.dirname(path_npz), exist_ok=True)
        meta = dataset["meta"]
        etas = sorted(dataset["per_eta"].keys())
        npz_payload = {"etas": np.array(etas, dtype=float),
                       "meta_spread_angle": meta["spread_angle"],
                       "meta_cone_seeds": np.array(meta["cone_seeds"], dtype=int), "meta_base_seed": meta["base_seed"]}
        for eta in etas:
            payload = dataset["per_eta"][eta]
            npz_payload[f"noisy_E_eta_{eta:.2f}"] = payload["noisy_E"]
            npz_payload[f"D_eta_{eta:.2f}"] = payload["D"]
            npz_payload[f"lam_vec_eta_{eta:.2f}"] = payload["lam_vec"]
            for m, labs in payload["labels"].items():
                npz_payload[f"labels_{m}_eta_{eta:.2f}"] = labs
        np.savez_compressed(path_npz, **npz_payload)
        with open(path_pkl, "wb") as f:
            pickle.dump(dataset, f, protocol=4)
        return path_npz, path_pkl
    
    def print_D_from_pkl(self, path_pkl="Plots&Data/MLQS/noisy_obs_dataset.pkl", eta: float | None = None):
        with open(path_pkl, "rb") as f:
            dataset = pickle.load(f)
        if "per_eta" not in dataset:
            raise ValueError("Dataset missing 'per_eta'. Pass the output of prepare_povm_dataset/cluster_povm_dataset.")
        if eta is not None:
            eta_key = next((k for k in dataset["per_eta"] if abs(float(k) - float(eta)) < 1e-12), None)
            if eta_key is None:
                raise KeyError(f"eta={eta} not found in dataset['per_eta'].")
            blob = dataset["per_eta"][eta_key]
            D = blob["D"]
            print(f"eta={eta_key}:")
            print(D)
            print()
            return
        for eta_key, blob in dataset["per_eta"].items():
            D = blob["D"]
            print(f"eta={eta_key}:")
            print(D)
            print()
        return
    
    def plot_eta_grid_from_dataset(self, dataset, methods=("kmeans","kmedoids","hdbscan"), savepath="Plots&Data/MLQS/unnamed.png", fontsize=18):
        if savepath is None:
            savepath = "Plots&Data/MLQS/unnamed.png"
        etas = sorted(dataset["per_eta"].keys())
        fig = plt.figure(figsize=(5*len(etas), 5*len(methods)))
        axes = [fig.add_subplot(len(methods), len(etas), i + 1, projection='3d') for i in range(len(methods) * len(etas))]
        for col, eta in enumerate(etas):
            payload = dataset["per_eta"][eta]
            noisy_E = payload["noisy_E"]
            for row, method in enumerate(methods):
                labels = payload["labels"][method]
                ax = axes[row * len(etas) + col]
                b = Bloch(fig=fig, axes=ax)
                labs = np.asarray(labels)
                n_clusters_detected = int(labs.max()) + 1 if labs.size > 0 else 1
                nc = max(n_clusters_detected, int(getattr(self, "n_clusters", n_clusters_detected)))
                if nc <= 10:
                    cmap = plt.cm.get_cmap("tab10")
                elif nc <= 20:
                    cmap = plt.cm.get_cmap("tab20")
                else:
                    cmap = plt.cm.get_cmap("hsv")
                n_effects = len(noisy_E[0]) if noisy_E and len(noisy_E) > 0 else 1
                color_by_label = [cmap(int(l) % cmap.N) for l in labs]
                b.vector_color = [col for col in color_by_label for _ in range(n_effects)]
                b.vector_width = 1
                Eij = [Qobj(noisy_E[i][j]) for i in range(len(noisy_E)) for j in range(len(noisy_E[i]))]  # type: ignore
                b.add_states(Eij)  # type: ignore
                b.render()
        for col, eta in enumerate(etas):
            ax = axes[col]
            pos = ax.get_position(); x_center = pos.x0/0.9 + pos.width/2
            fig.text(x_center, 0.9, rf"$\\eta$={eta:.2f}", ha='center', va='bottom', fontsize=fontsize)
        for row, method in enumerate(methods):
            ax = axes[row * len(etas)]
            pos = ax.get_position(); y_center = pos.y0 + pos.height / 2
            fig.text(pos.x0, y_center, method, rotation='vertical', ha='center', va='center', fontsize=fontsize)
        plt.tight_layout(rect=[0.1, 0, 1, 0.9])  # type: ignore
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        plt.show()


    # ---------- Helpers ----------

    # (Removed legacy static helper/dataset plotting methods to avoid duplication.)


__all__ = [
    "CompatibilityMeasure",
    "ClusteringToolkit",
]
