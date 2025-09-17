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

from qutip import Qobj, qeye, sigmax, sigmay, sigmaz, tensor, commutator, expect, Bloch
import os
import pickle
import matplotlib.pyplot as plt

_SIGMAS = [sigmax(), sigmay(), sigmaz()]

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
    return Qobj(A.full().reshape(-1, 1))


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

    def __init__(self, *, atol: float = 1e-9, method: str = "analytical", rng=None):
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

    def mutual_eigenspace_disturbance(self, povm1: Sequence[Qobj], povm2: Sequence[Qobj], *, state: Qobj | None = None) -> float:
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
        if not _is_hermitian_qobj(state, atol=self.atol):
            raise ValueError("State must be Hermitian.")
        state = state / state.norm()
        if method == "analytical":
            summation = 0.0
            state_mat = state.full()
            for E in povm1:
                E_mat = E.full()
                for F in povm2:
                    F_mat = F.full()
                    summation += np.trace(state_mat @ E_mat @ F_mat @ E_mat @ F_mat)
            return float(np.sqrt(1 - (summation.real / povm1[0].shape[0])))
        elif method == "numerical":
            C_tilda = 0
            for E in povm1:
                C_tilda += tensor(E, E.dag())
            dimensions = C_tilda.dims  # type: ignore
            D_list = [operator_double_ket(F) for F in povm2]
            D = 0
            for dvec in D_list:
                D += dvec * dvec.dag()  # type: ignore
            D.dims = dimensions  # type: ignore
            NCOM = np.sqrt(1 - (D * C_tilda * tensor(qeye(povm1[0].dims[0]), state.trans())).tr().real)  # type: ignore
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
                d = self.mutual_eigenspace_disturbance(povms[i], povms[j])
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
    axes : sequence of 3-vectors | None
        List of Bloch axes around which clusters are sampled. If None the
        default axes [[1,0,0], [0,0,1]] are used.

    Main methods
    ------------
    - sample_unit_vectors_cone(n, spread_angle, axis): sample Bloch directions in a cone
    - projective_qubit_povm_from_axis(n): build a binary projective POVM from a Bloch axis
    - generate_noisy_dataset(etas, n_clusters, ...): sample vectors, add noise, compute distances
    - cluster_from_distance(D, n_clusters): cluster objects given a distance matrix

    """

    def __init__(self, *, rng=None, cluster_method: str = "k-medoids", init: str = "kmeans++",
                 n_points: int = 50, spread_angle: float = 22.5, cm: CompatibilityMeasure | None = None,
                 axes: Sequence[Iterable[float]] | None = None, splits: int = 10):
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
        if cm is None:
            self.cm = CompatibilityMeasure()
        else:
            self.cm = cm
        if axes is None:
            self.axes = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])]
        else:
            self.axes = [np.asarray(a, dtype=float) for a in axes]

    # ----- Sampling -----
    def sample_unit_vectors_cone_x(self, n: int = 50, spread_angle: float = 22.5, center: str = 'positive') -> np.ndarray:
        """Sample `n` unit vectors in a cone about the +x axis.

        The cone opening is given by `spread_angle`. `center` controls whether the
        cone is centered in the +x, -x, or both directions.
        """
        if not (0.0 <= spread_angle <= 180.0):
            raise ValueError("spread_angle must be in [0, 180].")
        theta = np.deg2rad(spread_angle)
        u = self.rng.uniform(np.cos(theta), 1.0, size=n)
        beta = self.rng.uniform(0.0, 2*np.pi, size=n)
        sin_alpha = np.sqrt(np.maximum(0.0, 1.0 - u*u))
        v = np.column_stack((u, sin_alpha*np.cos(beta), sin_alpha*np.sin(beta)))
        if center == 'negative':
            v = -v
        elif center == 'both':
            signs = self.rng.choice([-1.0, 1.0], size=n)
            v = v * signs[:, None]
        elif center != 'positive':
            raise ValueError("center must be one of {'positive','negative','both'}.")
        return v

    def sample_unit_vectors_cone(self, n=None, spread_angle=None, axis=None, rng=None):
        """Sample `n` unit vectors in a cone around `axis`.

        Any of `n`, `spread_angle`, or `axis` can be omitted and will default to
        the corresponding instance attributes: `self.n_points`,
        `self.spread_angle`, and `self.axes[0]` respectively.
        """
        if n is None:
            n = self.n_points
        if spread_angle is None:
            spread_angle = self.spread_angle
        if axis is None:
            axis = self.axes[0]
        axis = np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        v_local = self.sample_unit_vectors_cone_x(n, spread_angle, center='positive')
        x_axis = np.array([1.0, 0.0, 0.0])
        if np.allclose(axis, x_axis):
            return v_local
        if np.allclose(axis, -x_axis):
            return -v_local
        rot_axis = np.cross(x_axis, axis); rot_axis /= np.linalg.norm(rot_axis)
        angle = np.arccos(np.clip(np.dot(x_axis, axis), -1.0, 1.0))
        K = np.array([[0, -rot_axis[2], rot_axis[1]],[rot_axis[2], 0, -rot_axis[0]],[-rot_axis[1], rot_axis[0], 0]])
        R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)
        return v_local @ R.T

    @staticmethod
    def projective_qubit_povm_from_axis(n: Iterable[float]) -> List[Qobj]:
        """Return the binary projective POVM aligned with Bloch vector `n`.

        Returns two Qobj POVM elements [E+, E-] with E+ = (I + n·σ)/2.
        """
        n = _normalize_direction(np.asarray(n, dtype=float))
        I = qeye(2)
        n_dot_sigma = n[0] * _SIGMAS[0] + n[1] * _SIGMAS[1] + n[2] * _SIGMAS[2]
        return [(I + n_dot_sigma) * 0.5, (I - n_dot_sigma) * 0.5]
    
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
        # Parameters and Returns are described in the detailed docstring
        # above; keep the implementation docstring minimal here.
        if not P:
            raise ValueError("Empty POVM.")
        d = P[0].shape[0]
        dimensions = P[0].dims
        if any(Q.shape != (d, d) for Q in P):
            raise ValueError("All POVM elements must be square Qobj of the same dimension.")
        I = qeye(dimensions[0])  

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

        S = 0
        for Ei in E:
            S = S + Ei * Ei.dag()
        corr = I - S
        if corr.norm() > 1e-12:
            # Add correction to last element to ensure sum_i E_i^2 = I
            E[-1] = E[-1] + corr

        return E, N

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

    # ----- Dataset generation (instance-based) -----
    def generate_noisy_dataset(self, n_clusters, *, etas: list[float] | None = None, n=None, spread_angle=None, methods=("kmeans","kmedoids","hdbscan"), cone_seeds=(42,43), base_seed=12345, noisy: bool = True):
        """Generate POVM datasets (noisy or noiseless) sampled around the instance's `axes`.

        Parameters
        ----------
        etas : iterable of float
            Noise scaling factors for which noisy datasets will be generated.
        n_clusters : int
            Number of clusters to request when clustering the distance matrix.
        n : int, optional
            Total number of samples; if None ``2 * self.n_points`` is used.
        spread_angle : float, optional
            Cone opening angle in degrees; if None ``self.spread_angle`` is used.
        methods : tuple[str], optional
            Which clustering methods to run for each eta.
        cone_seeds : tuple[int], optional
            Seeds used to initialize cone samplers per axis for reproducibility.
        base_seed : int, optional
            Seed used to construct the base randomness for noise vectors.
        noisy : bool, optional
            If True (default) generate noisy POVMs using the isotropic-noise model.
            If False, generate noiseless/projective POVMs (lam_vec will be zeros).

        Returns
        -------
        dict
            A dictionary with keys 'meta', 'vectors', and 'per_eta'. The
            'per_eta' sub-dictionary maps each eta to a dict containing the
            lam_vec, noisy_E (array of POVM matrices), pairwise D and
            the clustering labels for each requested method.
        """
        if noisy and (etas is None or len(etas) == 0):
            raise ValueError("At least one eta must be provided for noisy dataset generation.")
        else:
            etas = [0.0] if etas is None else etas
        if n is None:
            n = 2 * self.n_points
        if spread_angle is None:
            spread_angle = self.spread_angle

        # sample cones around each axis stored in self.axes
        if len(self.axes) < 2:
            raise ValueError("At least two axes must be provided in self.axes to generate a two-cluster dataset.")
        n_per_cluster = int(n // len(self.axes))
        vecs = []
        # allow deterministic cone seeds for reproducibility
        seeds = list(cone_seeds)
        for idx, ax in enumerate(self.axes):
            seed = seeds[idx] if idx < len(seeds) else None
            local_tool = ClusteringToolkit(rng=seed, n_points=n_per_cluster, spread_angle=spread_angle, cm=self.cm, axes=self.axes)
            v = local_tool.sample_unit_vectors_cone(n_per_cluster, spread_angle, ax)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            vecs.append(v)
        all_vectors = np.vstack(vecs)
        proj_povms = [self.projective_qubit_povm_from_axis(all_vectors[i]) for i in range(all_vectors.shape[0])]
        rng = np.random.default_rng(base_seed)
        base_Rl = rng.uniform(0.0, 1.0, size=all_vectors.shape[0])
        out = {"meta": {"spread_angle": float(spread_angle), "n": int(all_vectors.shape[0]), "etas": list(map(float, etas)), "cone_seeds": tuple(map(int, cone_seeds)), "base_seed": int(base_seed), "axes": [list(a) for a in self.axes], "noisy": bool(noisy)}, "vectors": all_vectors, "per_eta": {}}
        for eta in etas:
            if noisy:
                lam_vec = eta * base_Rl
            else:
                lam_vec = np.zeros_like(base_Rl)
            noisy_kraus = []
            noisy_arrays = []
            if noisy:
                for i, proj in enumerate(proj_povms):
                    E, N = self.noisy_povm_with_kraus(proj, lam=float(lam_vec[i]))
                    noisy_kraus.append([K for Ni in N for K in Ni])
                    noisy_arrays.append([Ei.full() for Ei in E])
            else:
                for i, proj in enumerate(proj_povms):
                    E = N = proj  # projective (noiseless) POVM
                    noisy_kraus.append(E)  # each Ni has one element
                    noisy_arrays.append([Ei.full() for Ei in E])
            D = self.cm.incompatibility_distance_matrix(noisy_kraus)
            labels = {m: self.cluster_from_distance(D, n_clusters=n_clusters, method=m) for m in methods}
            out["per_eta"][float(eta)] = {"lam_vec": lam_vec, "noisy_E": np.asarray(noisy_arrays, dtype=complex), "D": D, "labels": labels}
        return out

    def save_noisy_dataset(self, dataset, path_npz="Plots&Data/MLQS/noisy_obs_dataset.npz", path_pkl="Plots&Data/MLQS/noisy_obs_dataset.pkl"):
        os.makedirs(os.path.dirname(path_npz), exist_ok=True)
        meta = dataset["meta"]
        etas = sorted(dataset["per_eta"].keys())
        npz_payload = {"etas": np.array(etas, dtype=float), "vectors": dataset["vectors"],
                       "meta_spread_angle": meta["spread_angle"], "meta_n": meta["n"],
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

    def plot_eta_grid_from_dataset(self, dataset, methods=("kmeans","kmedoids","hdbscan"), savepath="Plots&Data/MLQS/unnamed.png", fontsize=18):
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
                b.vector_color = ['b' if lab == 0 else 'g' for lab in labels for _ in (0, 1)]
                b.vector_width = 1
                for i in range(noisy_E.shape[0]):
                    Ei = [Qobj(noisy_E[i, j]) for j in range(noisy_E.shape[1])]
                    b.add_states(Ei)  # type: ignore
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
