"""Convex spectral cutting-plane fits and explicitly qualified metrology designs.

SM Eqs. (S9), (S23)--(S27). The cutting-plane step has a certified
LP lower bound and an eigenvalue upper bound; the outer shot allocation
is numeric. This is NOT the unavailable original multi-start global search.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import linprog, minimize
from .observables import Candidate, hermitian


@dataclass
class SpectralFit:
    coefficients: np.ndarray
    intercept: float
    distance_upper: float
    objective_lower: float
    objective_upper: float
    ranges: np.ndarray
    iterations: int
    converged: bool
    def description(self):
        return dict(distance_upper=float(self.distance_upper),
                    objective_lower=float(self.objective_lower),
                    objective_upper=float(self.objective_upper),
                    certificate_gap=float(self.objective_upper-self.objective_lower),
                    iterations=int(self.iterations), converged=bool(self.converged),
                    intercept=float(self.intercept),
                    coefficients=self.coefficients.tolist(),
                    ranges=self.ranges.tolist())


def confidence_factor(M: int = 15, alpha: float = 0.01) -> float:
    return math.sqrt(2 * math.log(2 * M / alpha))


def _group_constraints(candidate: Candidate, idx_z: int, ncols: int):
    """Linearize joint record range and single-observable spectral range."""
    rows, b = [], []
    for s, group in enumerate(candidate.groups):
        z = idx_z + s
        if group.kind == 'record':
            idx = list(group.indices) + [None]  # p4 coefficient set to zero
            for p in idx:
                for q in idx:
                    if p is q:
                        continue
                    row = np.zeros(ncols)
                    if p is not None:
                        row[p] += 1.0
                    if q is not None:
                        row[q] -= 1.0
                    row[z] = -1.0
                    rows.append(row)
                    b.append(0.0)
        else:
            p = group.indices[0]
            for sign in (-1, 1):
                row = np.zeros(ncols)
                row[p] = sign * group.spectral_range
                row[z] = -1.0
                rows.append(row)
                b.append(0.0)
    return rows, b


def spectral_fit(candidate: Candidate, target: np.ndarray, *,
                 kappa: float = 1.0, shot_counts=None,
                 M: int = 15, alpha: float = 0.01,
                 anchor_state=None, max_cuts: int = 150,
                 atol: float = 5e-8) -> SpectralFit:
    """Fit A=a0 I+sum c_i A_i to target in spectral norm plus metrology.

    With `shot_counts=None`, solves pure min spectral distance d_R(X).
    With fixed positive shots N_s, minimizes kappa*d_R(X) plus sum_s
    b_s(c)/sqrt(N_s), where b_s uses setting-level measurement ranges.
    Optional `anchor_state` imposes Tr(rho_anchor A)=Tr(rho_anchor X).

    Every eigenvector cut is a necessary linear constraint. The LP optimal
    value is a lower bound; direct diagonalization supplies the upper bound.
    """
    if kappa <= 0:
        raise ValueError('kappa must be positive')
    A = candidate.operators
    n = len(A)
    ns = len(candidate.groups)
    nvar = n + 2 + ns
    ia0, it, iz = n, n + 1, n + 2
    X = hermitian(np.asarray(target, dtype=complex))
    d = X.shape[0]
    eye = np.eye(d, dtype=complex)
    w = np.zeros(ns)
    if shot_counts is not None:
        Ns = np.asarray(shot_counts, dtype=float)
        if Ns.shape != (ns,) or np.any(Ns <= 0):
            raise ValueError('one positive shot count per measurement group required')
        w = confidence_factor(M, alpha) / np.sqrt(Ns)
    objective = np.zeros(nvar)
    objective[it] = kappa
    objective[iz:] = w
    static_rows, static_b = _group_constraints(candidate, iz, nvar)

    A_eq = b_eq = None
    if anchor_state is not None:
        rho = hermitian(np.asarray(anchor_state))
        row = np.zeros(nvar)
        for j, op in enumerate(A):
            row[j] = float(np.trace(rho @ op).real)
        row[ia0] = 1.0
        A_eq = [row]
        b_eq = [float(np.trace(rho @ X).real)]

    rows = static_rows[:]
    rhs = static_b[:]
    bounds = [(None, None)] * n + [(None, None), (0.0, None)] + [(0.0, None)] * ns

    def cutting_plane(v, sign):
        # sign=+1: <X - A> <= t; sign=-1: <A - X> <= t
        vals = np.array([np.vdot(v, op @ v).real for op in A])
        row = np.zeros(nvar)
        row[:n] = -sign * vals
        row[ia0] = -sign
        row[it] = -1
        y = -sign * float(np.vdot(v, X @ v).real)
        return row, y

    # Start with two extremal target eigenvectors and one computational vector.
    evals, vecs = np.linalg.eigh(X)
    for idx in (0, -1):
        for sign in (-1, 1):
            row, b = cutting_plane(vecs[:, idx], sign)
            rows.append(row)
            rhs.append(b)
    v0 = eye[:, 0]
    for sign in (-1, 1):
        row, b = cutting_plane(v0, sign)
        rows.append(row)
        rhs.append(b)

    best = None
    for iteration in range(1, max_cuts + 1):
        res = linprog(objective, A_ub=rows, b_ub=rhs,
                      A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                      method='highs')
        if not res.success:
            raise RuntimeError('spectral LP failed: ' + res.message)
        co = res.x[:n]
        offset = res.x[ia0]
        residual = hermitian(X - offset * eye - sum((c * op for c, op in zip(co, A)), np.zeros_like(X)))
        ev, vs = np.linalg.eigh(residual)
        true_norm = float(max(abs(ev[0]), abs(ev[-1])))
        true_ranges = candidate.setting_ranges(co)
        lb = float(res.fun)
        ub = kappa * true_norm + float(w @ true_ranges)
        best = SpectralFit(co.copy(), float(offset), true_norm, lb, ub,
                           true_ranges, iteration, ub - lb <= atol)
        if best.converged:
            break
        # If the current residual violates the LP's claimed bound, add its
        # extremal eigenvectors as valid cuts. Add both signs for robustness.
        for idx in (0, -1):
            for sign in (-1, 1):
                row, b = cutting_plane(vs[:, idx], sign)
                rows.append(row)
                rhs.append(b)
    return best


def optimize_shared_shots(fits: list[SpectralFit], epsilon: float,
                          *, kappa: float = 0.04,
                          lifts=None, M: int = 15, alpha: float = 0.01,
                          maxiter: int = 4000):
    """Shared experimental settings across MANY fixed event-time fits.

    Minimize total Ns with constraints sum_s b_ks/sqrt(Ns) <=
    epsilon - delta_lift,k - kappa*d_k. Numerically convex in Ns>0;
    SLSQP is used with analytic gradients. If its returned solution is not
    feasible, the known feasible uniform allocation is used and flagged.
    """
    if not fits:
        raise ValueError('at least one fit is required')
    B = confidence_factor(M, alpha) * np.vstack([f.ranges for f in fits])
    n, ns = B.shape
    if lifts is None:
        lifts = np.zeros(n)
    margin = epsilon - np.asarray(lifts, dtype=float) - kappa * np.array([f.distance_upper for f in fits])
    if np.any(margin <= 0):
        return dict(feasible=False, reason='intrinsic/lift budget already exhausted',
                    margins=margin.tolist())
    uniform_n = max(1.0, float(np.max(np.sum(B, axis=1) / margin) ** 2))
    # Rescale optimization variables to avoid numbers around 1e8.
    scale = max(1.0, uniform_n)
    x_start = np.ones(ns) * 1.05
    def work(x):
        return float(np.sum(x))
    def jac(x):
        return np.ones_like(x)
    def constraint(x):
        return margin - np.sum(B / np.sqrt(scale * x[None, :]), axis=1)
    def cj(x):
        return (B / (2 * np.sqrt(scale) * x[None, :]**1.5))
    cons = [{'type': 'ineq', 'fun': constraint, 'jac': cj}]
    res = minimize(work, x_start, jac=jac, bounds=[(1e-12, None)] * ns,
                   constraints=cons, method='SLSQP',
                   options={'maxiter': maxiter, 'ftol': 1e-11})
    x = res.x if res.success and np.all(constraint(res.x) >= -1e-8) else x_start
    shot_counts = np.maximum(1, np.ceil(scale * x)).astype(np.int64)
    gaps = margin - np.sum(B / np.sqrt(shot_counts[None, :]), axis=1)
    # As a safety net, use a slightly enlarged analytic uniform allocation.
    if np.any(gaps < -1e-10):
        shot_counts = np.full(ns, int(np.ceil(scale * 1.05)), dtype=np.int64)
        gaps = margin - np.sum(B / np.sqrt(shot_counts[None, :]), axis=1)
    return dict(feasible=bool(np.min(gaps) >= -1e-10),
                total_shots=int(np.sum(shot_counts)), shots=shot_counts.tolist(),
                min_constraint_slack=float(np.min(gaps)),
                optimization_succeeded=bool(res.success and np.all(constraint(res.x) >= -1e-8)),
                message=res.message, n_constraints=n, n_settings=ns)


def scalar_shot_allocation(b, margin: float):
    """Exact single-constraint allocation, manuscript Eq. (S25)."""
    b = np.asarray(b, dtype=float)
    if margin <= 0 or np.any(b < 0):
        raise ValueError('require margin>0 and b>=0')
    weight = b ** (2 / 3)
    return weight * (weight.sum() ** 2) / (margin ** 2)
