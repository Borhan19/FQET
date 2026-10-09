"""Independently reconstructed FQET finite-system Hamiltonians.

Source: FQET manuscript (9 October 2026), SM Notes 7 and 11.
All energies and times are in J=1, hbar=1 units; no environment or
partial trace enters these isolated finite-system simulations.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import comb, sqrt
import numpy as np


def boson_basis(L: int = 4, Q: int = 4) -> list[tuple[int, ...]]:
    """All weak compositions of Q into L parts, in fixed deterministic order."""
    def rec(n: int, sites: int):
        if sites == 1:
            yield (n,)
        else:
            for v in range(n + 1):
                for rest in rec(n - v, sites - 1):
                    yield (v,) + rest
    basis = list(rec(Q, L))
    assert len(basis) == comb(L + Q - 1, Q)
    return basis


@dataclass(frozen=True)
class FiniteSystem:
    kind: str
    basis: tuple
    hopping: np.ndarray
    interaction: np.ndarray
    offset: np.ndarray
    projectors: tuple[np.ndarray, ...]
    temperature: float
    offset_initial: float
    offset_final: float
    ramp_duration: float
    J: float = 1.0
    interaction_strength: float = 0.0
    transverse_field: float = 0.0

    @property
    def dim(self) -> int:
        return len(self.basis)

    @property
    def identity(self) -> np.ndarray:
        return np.eye(self.dim, dtype=complex)

    def H(self, time: float, *, clamp: bool = True) -> np.ndarray:
        if clamp:
            s = float(np.clip(time / self.ramp_duration, 0.0, 1.0))
        else:
            s = float(time / self.ramp_duration)
        lam = self.offset_initial + (self.offset_final - self.offset_initial) * s
        return self.hopping + self.interaction + lam * self.offset

    @property
    def offset_derivative(self) -> float:
        return (self.offset_final - self.offset_initial) / self.ramp_duration

    def canonical(self) -> np.ndarray:
        ev, v = np.linalg.eigh(self.H(0.0))
        weights = np.exp(-(ev - ev.min()) / self.temperature)
        weights /= weights.sum()
        rho = (v * weights) @ v.conj().T
        return 0.5 * (rho + rho.conj().T)

    def check(self, tol: float = 1e-12) -> None:
        d = self.dim
        assert self.hopping.shape == (d, d)
        for op in [self.hopping, self.interaction, self.offset, *self.projectors]:
            assert np.linalg.norm(op - op.conj().T, 2) < tol
        assert np.linalg.norm(sum(self.projectors) - self.identity, 2) < tol
        for i, p in enumerate(self.projectors):
            assert np.linalg.norm(p @ p - p, 2) < tol
            assert np.linalg.norm(self.offset @ p - i * p, 2) < tol
        for p, q in product(self.projectors, self.projectors):
            if p is not q:
                assert np.linalg.norm(p @ q, 2) < tol


def bose_hubbard(*, L: int = 4, Q: int = 4, U: float = 1.3 / 3,
                 lam0: float = 20.0, tau: float = 20.0,
                 temperature: float = 5.0, J: float = 1.0) -> FiniteSystem:
    if L != 4 or Q != 4:
        raise ValueError('FQET benchmark uses L=Q=4; update record conventions for other sizes.')
    basis = boson_basis(L, Q)
    index = {b: i for i, b in enumerate(basis)}
    d = len(basis)
    T = np.zeros((d, d), dtype=complex)
    D = np.zeros((d, d), dtype=complex)
    NR = np.zeros((d, d), dtype=complex)
    P = [np.zeros((d, d), dtype=complex) for _ in range(Q + 1)]
    for j, state in enumerate(basis):
        D[j, j] = sum(n * (n - 1) for n in state)
        nR = state[2] + state[3]
        NR[j, j] = nR
        P[nR][j, j] = 1.0
        for site in range(L - 1):
            if state[site + 1] > 0:
                dst = list(state)
                dst[site] += 1
                dst[site + 1] -= 1
                T[index[tuple(dst)], j] += sqrt((state[site] + 1) * state[site + 1])
            if state[site] > 0:
                dst = list(state)
                dst[site] -= 1
                dst[site + 1] += 1
                T[index[tuple(dst)], j] += sqrt((state[site + 1] + 1) * state[site])
    obj = FiniteSystem('Bose-Hubbard', tuple(basis), -J * T, U * D / 2.0,
                       NR, tuple(P), temperature, lam0, 0.0, tau,
                       J=J, interaction_strength=U)
    obj.check()
    return obj


def transverse_ising(*, L: int = 4, g: float = 1.5, J: float = 1.0,
                      lam0: float | None = None, lamf: float | None = None,
                      tau: float = 4.0, temperature: float = 2.0) -> FiniteSystem:
    """Direct computational-basis construction; |1> is Z=+1 (spin up)."""
    if L != 4:
        raise ValueError('FQET benchmark specifies L=4.')
    if lam0 is None:
        lam0 = 2.0 * J
    if lamf is None:
        lamf = -2.0 * J
    states = tuple(product([0, 1], repeat=L))
    idx = {s: i for i, s in enumerate(states)}
    d = len(states)
    zz = np.zeros((d, d), dtype=complex)
    xt = np.zeros((d, d), dtype=complex)
    nup = np.zeros((d, d), dtype=complex)
    P = [np.zeros((d, d), dtype=complex) for _ in range(L + 1)]
    for j, bits in enumerate(states):
        z = [2 * bit - 1 for bit in bits]
        zz[j, j] = sum(z[k] * z[k + 1] for k in range(L - 1))
        r = sum(bits)
        nup[j, j] = r
        P[r][j, j] = 1.0
        for k in range(L):
            flip = list(bits)
            flip[k] ^= 1
            xt[idx[tuple(flip)], j] += 1.0
    obj = FiniteSystem('Ising', states, -g * xt, -J * zz,
                       nup, tuple(P), temperature, lam0, lamf, tau,
                       J=J, transverse_field=g)
    obj.check()
    return obj


def ising_independent_kron(*, g: float = 1.5, J: float = 1.0,
                            lam: float = 0.0) -> np.ndarray:
    """Independent Pauli-tensor Hamiltonian used to cross-check the bit construction."""
    I = np.eye(2, dtype=complex)
    Z = np.diag([-1.0, 1.0]).astype(complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    N = (I + Z) / 2
    def local(k, a):
        ops = [I] * 4
        ops[k] = a
        out = ops[0]
        for b in ops[1:]:
            out = np.kron(out, b)
        return out
    out = sum(-g * local(k, X) + lam * local(k, N) for k in range(4))
    for k in range(3):
        ops = [I] * 4
        ops[k] = Z
        ops[k + 1] = Z
        term = ops[0]
        for b in ops[1:]:
            term = np.kron(term, b)
        out = out - J * term
    return out


def independently_rebuilt_hamiltonian(system: FiniteSystem, time: float):
    """Separate audit Hamiltonian assembled without system.hopping matrices.

    For Ising the full Pauli tensor products are rebuilt from scratch. For
    Bose-Hubbard the Fock-sector hopping actions are reconstructed directly.
    This source-level independence is narrower than independent experimental
    validation, but it prevents accidentally reusing precomputed H matrices.
    """
    lam = system.offset_initial + system.offset_derivative * float(np.clip(time, 0, system.ramp_duration))
    if system.kind == 'Ising':
        return ising_independent_kron(g=system.transverse_field, J=system.J, lam=lam)
    if system.kind != 'Bose-Hubbard':
        raise ValueError(system.kind)
    basis = list(system.basis)
    indices = {state: k for k, state in enumerate(basis)}
    output = np.zeros((len(basis), len(basis)), dtype=complex)
    for k, n in enumerate(basis):
        output[k, k] = sum(.5 * system.interaction_strength * x*(x-1) for x in n) + lam*(n[2]+n[3])
        for bond in range(3):
            for direction in (-1, +1):
                origin = bond if direction == 1 else bond+1
                dest = bond+1 if direction == 1 else bond
                if n[origin] == 0:
                    continue
                new = list(n)
                new[origin] -= 1
                new[dest] += 1
                output[indices[tuple(new)], k] -= system.J * np.sqrt(n[origin] * (n[dest]+1))
    return output
