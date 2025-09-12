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
    """Convert operator A to its double-ket vector form |A⟩⟩."""
    d1, d2 = A.shape
    if d1 != d2:
        raise ValueError("Operator must be square.")
    return Qobj(A.full().reshape(-1, 1))


class CompatibilityMeasure():
    """Tools for constructing observables, measuring incompatibility, and clustering.

    The class is stateless; methods operate on provided inputs.
    """

    # ---------- Construction utilities ----------
    @staticmethod
    def pauli_operators() -> Tuple[Qobj, Qobj, Qobj]:
        """Return the qubit Pauli operators (σx, σy, σz)."""
        return sigmax(), sigmay(), sigmaz()

    @staticmethod
    def projective_qubit_povm_from_axis(n: Iterable[float]) -> List[Qobj]:
        """Return binary projective POVM along direction n: [E_plus, E_minus].

        E_± = (I ± n·σ)/2, where n is a unit 3-vector.
        """
        n = _normalize_direction(np.asarray(n, dtype=float))
        I = qeye(2)
        n_dot_sigma = n[0] * _SIGMAS[0] + n[1] * _SIGMAS[1] + n[2] * _SIGMAS[2]
        E_plus = (I + n_dot_sigma) * 0.5
        E_minus = (I - n_dot_sigma) * 0.5
        return [E_plus, E_minus]

    # ---------- Validation ----------
    @staticmethod
    def is_povm(povm: Sequence[Qobj], atol: float = 1e-7) -> bool:
        """Check if a list of effects forms a valid POVM."""
        if len(povm) == 0:
            return False
        d = _dimension_from_povm(povm)
        if any(not _is_psd_qobj(E, atol=atol) for E in povm):
            return False
        S = sum(povm[1:], povm[0] * 0)  # zero-like
        for E in povm:
            S = S + E
        return (S - qeye(d)).norm() <= atol

    # ---------- Analytic qubit incompatibility ----------
    @staticmethod
    def mutual_eigenspace_disturbance(povm1: Sequence[Qobj], povm2: Sequence[Qobj], state: Qobj = None, method: str = "analytical", atol: float = 1e-9) -> float: # type: ignore
        """Compute the mutual eigenspace disturbance between two POVMs.

        This measures how much the two POVMs disturb each other's eigenspaces.
        """
        if povm1[0].shape != povm2[0].shape:
            raise ValueError("POVMs must have the same dimension.")
        if state is None:
            # Default to the maximally mixed (identity) state for the given POVM dimension
            dimensions = povm1[0].dims
            state = qeye(dimensions[0]) / povm1[0].shape[0]
        if not _is_hermitian_qobj(state, atol=atol):
            raise ValueError("State must be Hermitian.")
        # Ensure state is normalized
        state = state / state.norm()
        if method == "analytical":
            summation = 0.0
            for E in povm1:
                for F in povm2:
                    summation += (state * E * F * E * F).tr() # pyright: ignore[reportOperatorIssue]
            MED = np.sqrt(1-summation.real/ povm1[0].shape[0])  
            return float(MED)
        elif method == "numerical":
            C_tilda = 0
            for E in povm1:
                C_tilda += tensor(E, E.dag())
            dimensions = C_tilda.dims # type: ignore
            D_list = [operator_double_ket(F) for F in povm2]
            D = 0
            for d in D_list:
                D += d * d.dag() # type: ignore
            D.dims = dimensions # type: ignore
            NCOM = np.sqrt(1 - (D * C_tilda * tensor(qeye(povm1[0].dims[0]), state.trans())).tr().real) # type: ignore
            return float(NCOM)
        elif method == "experimental":
            summation = 0.0
            for E in povm1:
                for F in povm2:
                    summation += (state * commutator(E, F) * commutator(E, F).dag()).tr() # type: ignore
            NCOM = np.sqrt(summation/ 2)
            return float(NCOM)
        else:
            raise ValueError(f"Unknown method: {method}")
        
    @staticmethod
    def incompatibility_distance_matrix(povms: Sequence[Sequence[Qobj]], method: str = "analytical", atol: float = 1e-9) -> np.ndarray:
        """Compute the pairwise distance matrix for a list of POVMs.

        The (i,j) entry is the mutual eigenspace disturbance between povms[i] and povms[j].
        """
        n = len(povms)
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                d = CompatibilityMeasure.mutual_eigenspace_disturbance(povms[i], povms[j], method=method, atol=atol)
                D[i, j] = d
                D[j, i] = d
        return D
        
    # ---------- Sampling and POVM generation ----------
    @staticmethod   
    def sample_unit_vectors_cone_x(n: int = 50, theta_deg: float = 22.5, center: str = 'positive', rng=None) -> np.ndarray:
        """
        Draw n unit vectors uniformly over the spherical cap of half-angle `theta_deg`
        about the x-axis.

        Parameters
        ----------
        n : int
            Number of vectors.
        theta_deg : float
            Maximum polar angle from the +x axis (degrees), 0 <= theta <= 180.
        center : {'positive', 'negative', 'both'}, default 'positive'
            - 'positive': cap centered at +x (angle from +x <= theta).
            - 'negative': cap centered at -x (angle from -x <= theta).
            - 'both': mixture of the two caps with equal probability.
        rng : None | int | np.random.Generator
            Random seed or Generator for reproducibility.

        Returns
        -------
        v : (n, 3) np.ndarray
            Unit vectors.
        """
        if not (0.0 <= theta_deg <= 180.0):
            raise ValueError("theta_deg must be in [0, 180].")
        theta = np.deg2rad(theta_deg)

        # RNG setup
        rng = np.random.default_rng(rng)

        # For uniform solid-angle sampling on a spherical cap around +x:
        # dΩ = sin(α) dα dβ  =>  u = cos(α) is uniform.
        u = rng.uniform(np.cos(theta), 1.0, size=n)   # u = cos(α), α ∈ [0, θ]
        beta = rng.uniform(0.0, 2*np.pi, size=n)      # azimuth about x-axis

        sin_alpha = np.sqrt(np.maximum(0.0, 1.0 - u*u))
        # x-axis as polar axis: (x, y, z) = (cos α, sin α cos β, sin α sin β)
        v = np.column_stack((u, sin_alpha*np.cos(beta), sin_alpha*np.sin(beta)))

        if center == 'negative':
            v = -v
        elif center == 'both':
            signs = rng.choice([-1.0, 1.0], size=n)
            v = v * signs[:, None]
        elif center != 'positive':
            raise ValueError("center must be one of {'positive','negative','both'}.")
        
        return v
    
    @staticmethod
    def sample_unit_vectors_cone(n, theta_deg, axis, rng=None):
        axis = np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        v_local = CompatibilityMeasure.sample_unit_vectors_cone_x(n, theta_deg, center='positive', rng=rng)
        x_axis = np.array([1.0, 0.0, 0.0])
        if np.allclose(axis, x_axis):
            return v_local
        if np.allclose(axis, -x_axis):
            return -v_local
        rot_axis = np.cross(x_axis, axis)
        rot_axis /= np.linalg.norm(rot_axis)
        angle = np.arccos(np.clip(np.dot(x_axis, axis), -1.0, 1.0))
        K = np.array([
            [0, -rot_axis[2], rot_axis[1]],
            [rot_axis[2], 0, -rot_axis[0]],
            [-rot_axis[1], rot_axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)
        return v_local @ R.T

    @staticmethod
    def generate_povms_from_bloch_vectors(vectors: np.ndarray) -> List[List[Qobj]]:
        """Generate binary projective qubit POVMs from a list of Bloch vectors.

        Each vector should be a 3D array-like representing a direction.
        Returns a list of POVMs, each being [E_plus, E_minus].
        """
        povms = []
        for n in vectors:
            povm = CompatibilityMeasure.projective_qubit_povm_from_axis(n)
            povms.append(povm)
        return povms
    
    @staticmethod
    def noisy_povm_with_kraus_qobj(P, lam, p=None, *, splits=1, random_split=False, rng=None):
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

        if splits < 1:
            raise ValueError("splits ≥ 1.")

        def split_mass(total, m):
            if m == 1:
                return np.array([total])
            w = rng.dirichlet(np.ones(m)) if random_split else np.full(m, 1.0 / m)
            return total * w

        E, N = [], []
        for i in range(k):
            a_parts = split_mass(1.0 - lam, splits)
            b_parts = split_mass(lam * p[i], splits)

            Ei = (1 - lam) * P[i] + (lam * p[i]) * I
            E.append(Ei)

            Ni = []
            for aj, bj in zip(a_parts, b_parts):
                # sqrt(a P_i + b I) = sqrt(a+b) P_i + sqrt(b) (I - P_i)
                K = np.sqrt(aj + bj) * P[i] + np.sqrt(bj) * (I - P[i])
                Ni.append(Qobj(K.full()))
            N.append(Ni)

        S = 0
        for Ei in E:
            S = S + Ei * Ei.dag()
        corr = I - S
        if corr.norm() > 1e-12:
            # Add correction to last element to ensure sum_i E_i^2 = I
            E[-1] = E[-1] + corr

        return E, N
    
    # ---------- Convenience wrappers ----------
    def bloch_direction_from_projective(self, povm: Sequence[Qobj]) -> np.ndarray:
        """Given a binary projective qubit POVM [E+, E-], return its Bloch direction.

        Assumes E+ = (I + n·σ)/2, returns n.
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
    """Tools for clustering observables based on incompatibility distance matrices.

    The class is stateless; methods operate on provided inputs.
    """
    @staticmethod
    def cluster_from_distance(
        D,
        n_clusters: int,
        method: str = "k-medoids",
        *,
        init: str = "kmeans++",      # <--- new
        random_state=None,
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
    ):
        """
        Cluster items given a pairwise distance matrix.

        Parameters
        ----------
        D : (n, n) array_like
            Symmetric, non-negative distance matrix with zeros on the diagonal.
        n_clusters : int
            Number of clusters (k).
        method : {'k-medoids','k-means', 'hdbscan'}, default 'k-medoids'
        init : {'kmeans++','linear++','random'}, default 'kmeans++'
            Initialization for k-medoids:
            - 'kmeans++' : pick next medoid with prob ∝ (nearest-distance)^2
            - 'linear++' : pick next medoid with prob ∝ (nearest-distance)
            - 'random'   : k distinct indices uniformly at random
            (Ignored for k-means, which always uses k-means++ in this implementation.)
        random_state : None | int | np.random.Generator
        n_init, max_iter, tol : standard meanings

        Returns
        -------
        labels : (n,) np.ndarray[int]
            Cluster labels 0..k-1.
        """
        D = np.asarray(D, dtype=float)
        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError("D must be square (n x n).")
        n = D.shape[0]
        if not (1 <= n_clusters <= n):
            raise ValueError("n_clusters must be in [1, n].")
        if np.any(D < -1e-12):
            raise ValueError("Distances must be non-negative.")
        if not np.allclose(D, D.T, atol=1e-10, rtol=0):
            D = 0.5 * (D + D.T)
        np.fill_diagonal(D, 0.0)

        rng = np.random.default_rng(random_state)

        if method.lower() in {"k-medoids", "kmedoids", "pam"}:
            best_labels, best_cost = None, np.inf
            for _ in range(max(1, n_init)):
                labels, medoids, cost = ClusteringToolkit._pam_kmedoids(D, n_clusters, rng, max_iter, init=init)
                if cost < best_cost - 1e-12:
                    best_labels, best_cost = labels, cost
            return best_labels

        elif method.lower() in {"k-means", "kmeans"}:
            X = ClusteringToolkit._classical_mds(D)
            best_labels, best_inertia = None, np.inf
            for _ in range(max(1, n_init)):
                labels, inertia = ClusteringToolkit._kmeans_lloyd(X, n_clusters, rng, max_iter, tol)
                if inertia < best_inertia - 1e-12:
                    best_labels, best_inertia = labels, inertia
            return best_labels
        
        elif method.lower() == "hdbscan":
            try:
                import hdbscan
            except ImportError as e:
                raise ImportError("hdbscan package is required for method='hdbscan'.") from e
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=5,  # Minimum cluster size is set to 5 by default
                metric='precomputed',
                cluster_selection_method='eom'
            )
            labels = clusterer.fit_predict(D)
            return labels

        else:
            raise ValueError("method must be one of {'k-medoids','k-means'}.")


    # ---------- Helpers ----------

    @staticmethod
    def _pam_kmedoids(D, k, rng, max_iter, init="kmeans++"):
        """
        PAM-style k-medoids with configurable initialization.
        Objective: sum_i distance(point_i, medoid_of_cluster_i)
        """
        n = D.shape[0]
        medoids = ClusteringToolkit._init_medoids(D, k, rng, mode=init)

        prev_cost = np.inf
        for _ in range(max_iter):
            # Assign to nearest medoid
            distances_to_medoids = D[:, medoids]          # (n, k)
            labels = np.argmin(distances_to_medoids, axis=1)
            cost = distances_to_medoids[np.arange(n), labels].sum()

            # Update medoids cluster-wise
            new_medoids = medoids.copy()
            for j in range(k):
                idx = np.where(labels == j)[0]
                if idx.size == 0:
                    # Re-seed empty cluster with farthest point from any current medoid
                    far_idx = np.argmax(np.min(D[:, medoids], axis=1))
                    new_medoids[j] = far_idx
                    continue
                subD = D[np.ix_(idx, idx)]
                within_sums = subD.sum(axis=1)
                new_medoids[j] = idx[np.argmin(within_sums)]

            # De-duplicate and refill if needed
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
        """
        Initialize medoids from the distance matrix.

        mode:
        - 'kmeans++' : pick next index with prob ∝ (min_distance_to_selected)^2
        - 'linear++' : pick next index with prob ∝ (min_distance_to_selected)
        - 'random'   : k distinct indices uniformly
        """
        n = D.shape[0]
        mode = mode.lower()
        if mode == "random":
            return np.array(rng.choice(n, size=k, replace=False), dtype=int)

        # Start with one uniformly
        medoids = [int(rng.integers(0, n))]

        for _ in range(1, k):
            dmin = np.min(D[:, medoids], axis=1)  # distance to nearest chosen medoid
            dmin[medoids] = 0.0                   # prevent re-choosing existing medoids

            if mode == "kmeans++":
                weights = dmin * dmin             # squared distance weighting
            elif mode in {"linear++", "kmedoids++"}:
                weights = dmin
            else:
                raise ValueError("init must be one of {'kmeans++','linear++','random'}.")

            total = weights.sum()
            if not np.isfinite(total) or total <= 1e-20:
                # Degenerate: all distances ~0 to chosen medoids -> pick a random non-medoid
                choices = np.setdiff1d(np.arange(n), np.array(medoids))
                medoids.append(int(rng.choice(choices)))
            else:
                probs = weights / total
                # draw from non-zero-probability indices; rng.choice handles zero probs
                idx = int(rng.choice(n, p=probs))
                # ensure uniqueness (rare if probs computed correctly)
                if idx in medoids:
                    choices = np.setdiff1d(np.arange(n), np.array(medoids))
                    idx = int(rng.choice(choices))
                medoids.append(idx)

        return np.array(medoids, dtype=int)

    @staticmethod
    def _classical_mds(D):
        """
        Classical MDS (Torgerson) from distances to centered Gram matrix, then eigendecomposition.
        Returns coordinates X with as many positive-eigenvalue dimensions as available.
        """
        n = D.shape[0]
        D2 = D ** 2
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ D2 @ J  # double-centering -> Gram matrix

        # Symmetrize for numerical stability
        B = 0.5 * (B + B.T)

        # Eigen-decompose (B is symmetric)
        w, V = np.linalg.eigh(B)
        idx = np.argsort(w)[::-1]  # descending
        w = w[idx]
        V = V[:, idx]

        # Keep strictly positive eigenvalues (tolerance guards round-off)
        pos = w > (1e-12 * w[0] if w[0] > 0 else 1e-12)
        if not np.any(pos):
            # Degenerate case: collapse to 1D zeros
            return np.zeros((n, 1), dtype=float)

        Lhalf = np.sqrt(w[pos])
        X = V[:, pos] * Lhalf  # scale each eigenvector by sqrt(eig)
        return X

    @staticmethod
    def _kmeans_lloyd(X, k, rng, max_iter, tol):
        if not np.isfinite(X).all():
            raise ValueError("Embedding contains non-finite values. "
                            "Your distance matrix may be far from Euclidean; "
                            "use k-medoids or check D.")

        n, d = X.shape
        centers = np.empty((k, d), dtype=X.dtype)

        # pick first center uniformly
        first = rng.integers(0, n)
        centers[0] = X[first]

        # distances to nearest chosen center (squared)
        closest_sq = ClusteringToolkit._row_min_sqdist(X, centers[0:1])
        closest_sq = np.clip(closest_sq, 0.0, None)   # <-- IMPORTANT

        for i in range(1, k):
            weights = np.clip(closest_sq, 0.0, None)  # ensure non-negative
            total = weights.sum()

            if not np.isfinite(weights).all() or total <= 1e-20:
                # degenerate case: pick uniformly
                idx = int(rng.integers(0, n))
            else:
                probs = weights / total
                # np.random.Generator.choice demands probs >= 0 and sum to 1
                idx = int(rng.choice(n, p=probs))

            centers[i] = X[idx]

            # update closest squared distances to nearest center so far
            new_d2 = ClusteringToolkit._row_min_sqdist(X, centers[i:i+1])
            closest_sq = np.minimum(closest_sq, new_d2)
            closest_sq = np.clip(closest_sq, 0.0, None)  # <-- IMPORTANT

        # ---- regular Lloyd iterations (unchanged except for numerical guards) ----
        prev_inertia = np.inf
        for _ in range(max_iter):
            d2 = ClusteringToolkit._pair_sqdist(X, centers)
            labels = np.argmin(d2, axis=1)
            inertia = d2[np.arange(n), labels].sum()

            new_centers = np.zeros_like(centers)
            counts = np.bincount(labels, minlength=k)
            for j in range(k):
                if counts[j] > 0:
                    new_centers[j] = X[labels == j].mean(axis=0)
                else:
                    # reseed to farthest point
                    far_idx = np.argmax(np.min(d2, axis=1))
                    new_centers[j] = X[far_idx]

            centers = new_centers

            if prev_inertia - inertia <= tol * max(1.0, prev_inertia):
                break
            prev_inertia = inertia

        d2 = ClusteringToolkit._pair_sqdist(X, centers)
        labels = np.argmin(d2, axis=1)
        inertia = d2[np.arange(n), labels].sum()
        return labels.astype(int), float(inertia)

    @staticmethod
    def _pair_sqdist(X, C):
        # squared Euclidean distances between rows of X (n,d) and C (k,d)
        X2 = np.sum(X * X, axis=1, keepdims=True)      # (n,1)
        C2 = np.sum(C * C, axis=1, keepdims=True).T    # (1,k)
        d2 = X2 + C2 - 2.0 * (X @ C.T)
        # Numerical guard: tiny negatives -> 0
        return np.maximum(d2, 0.0)

    @staticmethod
    def _row_min_sqdist(X, C):
        return np.min(ClusteringToolkit._pair_sqdist(X, C), axis=1)
    
    @staticmethod
    def generate_noisy_dataset(n, spread_angle, etas, n_clusters, methods= ('kmeans','kmedoids','hdbscan'),
                           cone_seeds=(42, 43), base_seed=12345):
        rng_cone_x = np.random.default_rng(cone_seeds[0])
        rng_cone_z = np.random.default_rng(cone_seeds[1])
        vx = CompatibilityMeasure.sample_unit_vectors_cone(n//2, theta_deg=spread_angle, axis=[1,0,0], rng=rng_cone_x)
        vz = CompatibilityMeasure.sample_unit_vectors_cone(n//2, theta_deg=spread_angle, axis=[0,0,1], rng=rng_cone_z)
        vx = vx / np.linalg.norm(vx, axis=1, keepdims=True)
        vz = vz / np.linalg.norm(vz, axis=1, keepdims=True)
        all_vectors = np.vstack([vx, vz])
        observables = [CompatibilityMeasure.projective_qubit_povm_from_axis(all_vectors[i]) for i in range(n)]

        rng = np.random.default_rng(base_seed)
        base_Rl = rng.uniform(0.0, 1.0, size=n)

        results = {"meta": {"spread_angle": float(spread_angle), "n": int(n),
                            "etas": list(map(float, etas)),
                            "cone_seeds": tuple(map(int, cone_seeds)),
                            "base_seed": int(base_seed)},
                "vectors": all_vectors,
                "per_eta": {}}

        for eta in etas:
            lam_vec = eta * base_Rl
            noisy_E = []
            kraus_flat = []
            for i, obs in enumerate(observables):
                Ei, Nij = CompatibilityMeasure.noisy_povm_with_kraus_qobj(obs, lam=float(lam_vec[i]))
                noisy_E.append([E.full() for E in Ei])  # shape (2,2,2)
                kraus_flat.append([K for Klist in Nij for K in Klist])

            D = CompatibilityMeasure.incompatibility_distance_matrix(kraus_flat)

            labels = {}
            for i, method in enumerate(methods):
                labels[method] = ClusteringToolkit.cluster_from_distance(D, n_clusters=n_clusters, method=method)

            # ready-to-plot payload for this eta
            results["per_eta"][float(eta)] = {
                "lam_vec": lam_vec,
                "noisy_E": np.asarray(noisy_E, dtype=complex),    # (n, 2, 2, 2)
                "D": D,
                "labels": labels
            }

        return results

    @staticmethod
    def save_noisy_dataset(dataset, path_npz="Plots&Data/MLQS/noisy_obs_dataset.npz",
                        path_pkl="Plots&Data/MLQS/noisy_obs_dataset.pkl"):
        os.makedirs(os.path.dirname(path_npz), exist_ok=True)
        # NPZ for arrays + small metadata; PKL for full dict (simple reload)
        meta = dataset["meta"]
        vectors = dataset["vectors"]
        etas = list(dataset["per_eta"].keys())
        npz_payload = {"vectors": vectors, "etas": np.array(etas, dtype=float)}
        for eta in etas:
            key = f"E_eta_{eta:.2f}"
            npz_payload[key] = dataset["per_eta"][eta]["noisy_E"]
            npz_payload[f"D_eta_{eta:.2f}"] = dataset["per_eta"][eta]["D"]
            npz_payload[f"labels_kmeans_eta_{eta:.2f}"] = dataset["per_eta"][eta]["labels"]["kmeans"]
            npz_payload[f"labels_kmedoids_eta_{eta:.2f}"] = dataset["per_eta"][eta]["labels"]["kmedoids"]
            npz_payload[f"lam_vec_eta_{eta:.2f}"] = dataset["per_eta"][eta]["lam_vec"]
        # minimal meta
        npz_payload["meta_spread_angle"] = meta["spread_angle"]
        npz_payload["meta_n"] = meta["n"]
        npz_payload["meta_cone_seeds"] = np.array(meta["cone_seeds"], dtype=int)
        npz_payload["meta_base_seed"] = meta["base_seed"]
        np.savez_compressed(path_npz, **npz_payload)

        with open(path_pkl, "wb") as f:
            pickle.dump(dataset, f, protocol=4)

        return path_npz, path_pkl

    @staticmethod
    def plot_eta_grid_from_dataset(dataset, methods=('kmeans','kmedoids', 'hdbscan'),
                                savepath="Plots&Data/MLQS/unnamed.png", fontsize=24):
        etas = sorted(dataset["per_eta"].keys())
        fig = plt.figure(figsize=(5*len(etas), 5*len(methods)))
        axes = [fig.add_subplot(len(methods), len(etas), i + 1, projection='3d') for i in range(len(methods) * len(etas))]

        for col, eta in enumerate(etas):
            payload = dataset["per_eta"][eta]
            noisy_E = payload["noisy_E"]           # (n,2,2,2)
            for row, method in enumerate(methods):
                labels = payload["labels"][method]  # (<n,)
                ax = axes[row * 3 + col]
                b = Bloch(fig=fig, axes=ax)
                b.vector_color = ['b' if lab == 0 else 'g' for lab in labels for _ in (0, 1)]
                b.vector_width = 1
                for i in range(noisy_E.shape[0]):
                    # reconstruct Qobj list [E0, E1] per observable
                    Ei = [Qobj(noisy_E[i, j]) for j in range(noisy_E.shape[1])]
                    b.add_states(Ei)  # type: ignore
                b.render()

        # Add big column labels (eta) at the top of the whole figure
        col_count = 3
        for col, eta in enumerate(etas):
            ax = axes[col]
            pos = ax.get_position()
            x_center = pos.x0/0.9 + pos.width/2
            fig.text(x_center, 0.9, rf"$\eta$={eta:.2f}", ha='center', va='bottom', fontsize=fontsize)

        # Add row labels on the left side (vertical)
        row_labels = methods
        for row in range(len(row_labels)):
            ax = axes[row * col_count]
            pos = ax.get_position()
            y_center = pos.y0 + pos.height / 2
            # place a vertical label slightly left of the leftmost subplot
            fig.text(pos.x0, y_center, row_labels[row], rotation='vertical',
                    ha='center', va='center', fontsize=fontsize)

        plt.tight_layout(rect=[0.1, 0, 1, 0.9])  # type: ignore # leave space for the top/left labels
        if savepath is not None:
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
            plt.savefig(savepath, dpi=300, bbox_inches='tight')
        else:
            savepath="Plots&Data/MLQS/unnamed.png"
            plt.savefig(savepath, dpi=300, bbox_inches='tight')
        plt.show()


    # -------- Example run in your notebook --------
    # spread_angle = 22.5
    # etas = [0.25, 0.50, 0.75]
    # dataset = generate_noisy_dataset(cm, ct, n, spread_angle, etas)
    # save_noisy_dataset(dataset)  # optional
    # plot_eta_grid_from_dataset(dataset)


__all__ = [
    "CompatibilityMeasure",
    "ClusteringToolkit",
]
