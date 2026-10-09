"""FQET retained operators, historical pullback, event targets, geometry.

Source: FQET SM Notes 2--4, 7, 11, 13. No claim is made that any one
candidate representation is universally optimal.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .models import FiniteSystem
from .dynamics import transition


def hermitian(x):
    return (x + x.conj().T) / 2


def event_masks():
    """Exactly 15 nontrivial binary events up to complements for five outcomes."""
    return tuple(range(1, 16))


def event_operator(projectors, event_mask: int):
    if not (1 <= event_mask <= 15):
        raise ValueError('event_mask must be 1..15 (p4 supplied by complement)')
    out = np.zeros_like(projectors[0])
    for r in range(4):
        if (event_mask >> r) & 1:
            out += projectors[r]
    return out


@dataclass
class MeasurementGroup:
    name: str
    kind: str     # 'record' for all four independent projected probabilities; 'individual' otherwise
    indices: tuple[int, ...]
    spectral_range: float = 1.0


@dataclass
class Candidate:
    label: str
    operators: list[np.ndarray]  # all non-identity operators
    names: list[str]
    groups: list[MeasurementGroup]
    decision_time: float
    histories: tuple[float, ...]

    def matrix(self, coefficients: np.ndarray, intercept: float = 0.0):
        result = np.eye(self.operators[0].shape[0]) * intercept
        for c, op in zip(coefficients, self.operators):
            result = result + c * op
        return hermitian(result)

    def setting_ranges(self, coefficients: np.ndarray):
        c = np.asarray(coefficients)
        out = []
        for s in self.groups:
            if s.kind == 'record':
                # c_4 = 0 by retaining p0..p3; the omitted p4 is reconstructed.
                eig = [float(c[i]) for i in s.indices] + [0.0]
                out.append(float(max(eig) - min(eig)))
            elif s.kind == 'individual':
                out.append(float(abs(c[s.indices[0]]) * s.spectral_range))
            else:
                raise ValueError(s.kind)
        return np.asarray(out)


def _at(samples, time):
    """Look up a sampled propagator while tolerating harmless FP time roundoff."""
    if time in samples:
        return samples[time]
    keys = np.asarray(list(samples), dtype=float)
    ix = int(np.argmin(np.abs(keys - time)))
    if abs(keys[ix] - time) > 1e-10:
        raise KeyError(f'missing U({time:.12g},0) in sampled trajectory')
    return samples[float(keys[ix])]


def make_candidate(system: FiniteSystem, *, tc: float, U_samples: dict[float, np.ndarray],
                   history_lags=(), order=0) -> Candidate:
    """E,R and optional history or first/second direct derivative layers.

    `U_samples[t]` is the numerical propagator U(t,0) for t=tc and earlier
    history times. Energy E=H(tc) is always a retained thermodynamic setting.
    """
    if (history_lags and order) or (order not in (0, 1, 2)):
        raise ValueError('choose history lags OR derivative order 1 or 2')
    labels = ['E']
    ops = [system.H(tc).copy()]
    grps = [MeasurementGroup('E', 'individual', (0,),
                             np.ptp(np.linalg.eigvalsh(ops[0])))]
    labels += [f'P{r}(tc)' for r in range(4)]
    i0 = len(ops)
    ops += [system.projectors[r].copy() for r in range(4)]
    grps.append(MeasurementGroup('record(tc)', 'record', tuple(range(i0, i0 + 4))))
    for lag in history_lags:
        if lag <= 0 or lag > tc:
            raise ValueError('history lag must obey 0 < s <= tc')
        U = transition(_at(U_samples, tc), _at(U_samples, float(tc - lag)))
        i0 = len(ops)
        ops += [hermitian(U @ P @ U.conj().T) for P in system.projectors[:4]]
        labels += [f'P{r}(tc-{lag:.6g})' for r in range(4)]
        grps.append(MeasurementGroup(f'R(tc-{lag:.6g})', 'record', tuple(range(i0, i0 + 4))))
    if order:
        H = system.H(tc)
        for k in (1, 2)[:order]:
            for r, P in enumerate(system.projectors[:4]):
                comm = H @ P - P @ H
                A = 1j * comm if k == 1 else -(H @ comm - comm @ H)
                A = hermitian(A)
                n = len(ops)
                ops.append(A)
                labels.append(f'D{k}_{r}')
                grps.append(MeasurementGroup(f'D{k}_{r}', 'individual', (n,),
                                             float(np.ptp(np.linalg.eigvalsh(A)))))
    if history_lags:
        label = ('H2' if len(history_lags) == 1 else 'H3')
    else:
        label = 'ERJ' if order == 1 else 'ERJK2' if order == 2 else 'ER'
    return Candidate(label, ops, labels, grps, tc, tuple(history_lags))


def target_at(system: FiniteSystem, *, tc: float, horizon: float,
              U_samples: dict[float, np.ndarray], event_mask: int) -> np.ndarray:
    U = transition(_at(U_samples, float(tc + horizon)), _at(U_samples, tc))
    P = event_operator(system.projectors, event_mask)
    return hermitian(U.conj().T @ P @ U)


def required_times(tc: float, lags=(), horizons=()):
    return sorted({0.0, float(tc)} | {float(tc - s) for s in lags} |
                  {float(tc + tau) for tau in horizons})


def hs_vectors(ops):
    """Hermitian matrices as real vectors; dot-product equals Tr(A B)."""
    rows = []
    for op in ops:
        flat = np.asarray(op).ravel()
        rows.append(np.concatenate((flat.real, flat.imag)))
    return np.stack(rows) if rows else np.zeros((0, 0))


def orthospan(ops, conditioning=None, relative_cutoff=1e-10):
    """HS-orthonormal columns for `ops` after removing `conditioning` span."""
    vec = hs_vectors(ops).T
    if conditioning is not None and np.size(conditioning):
        q0 = np.asarray(conditioning)
        vec = vec - q0 @ (q0.T @ vec)
    if not vec.size:
        return vec[:, :0]
    u, sig, _ = np.linalg.svd(vec, full_matrices=False)
    if not len(sig) or sig[0] == 0:
        return u[:, :0]
    return u[:, sig > max(1e-13, relative_cutoff * sig[0])]


def principal_geometry(subspace_a, subspace_b, conditioning):
    """Return mean cos^2 angle and mean angle after conditioning common span."""
    base = orthospan(conditioning)
    a = orthospan(subspace_a, conditioning=base)
    b = orthospan(subspace_b, conditioning=base)
    q = min(a.shape[1], b.shape[1])
    if q == 0:
        raise ValueError('empty conditioned subspace')
    cos = np.linalg.svd(a.T @ b, compute_uv=False)[:q].clip(0, 1)
    angles = np.degrees(np.arccos(cos))
    return dict(overlap=float(np.mean(cos**2)), mean_angle_deg=float(np.mean(angles)),
                angles_deg=[float(v) for v in angles],
                dim_a=a.shape[1], dim_b=b.shape[1])


def gram_condition(operators, common, *, relative_cutoff=1e-11):
    """Condition number for normalized operators after common directions removed."""
    B = orthospan(common) if len(common) else None
    V = hs_vectors(operators).T
    if B is not None:
        V = V - B @ (B.T @ V)
    norms = np.linalg.norm(V, axis=0)
    V = V[:, norms > relative_cutoff]
    V = V / np.linalg.norm(V, axis=0)
    vals = np.linalg.eigvalsh(V.T @ V)
    vals = vals[vals > relative_cutoff * vals[-1]]
    return float(vals[-1] / vals[0]) if len(vals) else float('inf')


def cross_platform_geometry(system, tc, U_samples, s1, s2):
    """First-history/current, conditioned second-history/curvature geometry."""
    hist = make_candidate(system, tc=tc, U_samples=U_samples, history_lags=(s1, s2))
    inst = make_candidate(system, tc=tc, U_samples=U_samples, order=2)
    I = system.identity
    common = [I, *inst.operators[:5]]
    h1, h2 = hist.operators[5:9], hist.operators[9:13]
    d1, d2 = inst.operators[5:9], inst.operators[9:13]
    first = principal_geometry(h1, d1, common)
    # The layer-resolved comparison is asymmetric: the second history layer
    # is conditioned on the FIRST HISTORY LAYER; K^(2) is conditioned on
    # the FIRST DERIVATIVE LAYER. This is essential to match SM Table S5.
    h2_orth = orthospan(h2, conditioning=orthospan([*common, *h1]))
    k2_orth = orthospan(d2, conditioning=orthospan([*common, *d1]))
    cos = np.linalg.svd(h2_orth.T @ k2_orth, compute_uv=False)
    cos = np.clip(cos[:min(h2_orth.shape[1], k2_orth.shape[1])], 0, 1)
    angles = np.degrees(np.arccos(cos))
    second = dict(overlap=float(np.mean(cos**2)),
                  mean_angle_deg=float(np.mean(angles)),
                  angles_deg=angles.tolist(),
                  dim_a=h2_orth.shape[1], dim_b=k2_orth.shape[1])
    # The reported Gram condition number is computed from the ENTIRE
    # retained (E,R,H1,H2) list after centering off the identity.
    # Projecting out the common R first gives a different (incorrect here)
    # condition number by many orders of magnitude.
    cond = gram_condition(hist.operators, [I])
    return dict(first_layer=first, second_layer=second, normalized_gram_cond=cond,
                decision_time=tc, lags=[s1, s2])
