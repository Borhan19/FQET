"""Independent unitary solvers (DOP853 and fourth-order two-node Gauss-Magnus).

The Gauss-Magnus implementation intentionally uses independent matrix exponentials
rather than SciPy's adaptive integration. Source: FQET SM Notes 7, 10, 12.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from .models import FiniteSystem, independently_rebuilt_hamiltonian


def evolve_at(system: FiniteSystem, times, *, solver: str = 'dop853',
              rtol: float = 2e-9, atol: float = 2e-11,
              max_step: float = 0.0025, independent_model: bool = False) -> np.ndarray:
    """Return U(t,0) at sorted requested nonnegative times, including t=0.

    Solver A is DOP853 with paper tolerances. Solver B is Gauss-Magnus with
    an adjustable maximum step, shortened to land exactly on every requested
    sample time. Hamiltonian convention is dU/dt = -i H(t) U.
    """
    ts = np.asarray(times, dtype=float)
    if ts.ndim != 1 or len(ts) < 1 or np.any(ts < -1e-14):
        raise ValueError('times must be a nonempty vector of nonnegative values')
    if np.any(np.diff(ts) < 0):
        raise ValueError('times must be sorted')
    if max_step <= 0:
        raise ValueError('max_step must be positive')
    d = system.dim
    eye = system.identity
    hamiltonian = (lambda t: independently_rebuilt_hamiltonian(system, t)) if independent_model else system.H
    ans = np.empty((len(ts), d, d), dtype=complex)
    if ts[-1] == 0:
        ans[:] = eye
        return ans
    if solver == 'dop853':
        def rhs(t, flat):
            return (-1j * (hamiltonian(t) @ flat.reshape((d, d)))).ravel()
        # solve_ivp disallows duplicated t_eval; restore duplicates afterwards.
        unique, inverse = np.unique(ts, return_inverse=True)
        sol = solve_ivp(rhs, (0, float(ts[-1])), eye.ravel(), method='DOP853',
                        t_eval=unique, rtol=rtol, atol=atol)
        if not sol.success or sol.y.shape[1] != len(unique):
            raise RuntimeError('DOP853 failed: ' + sol.message)
        return sol.y.T.reshape((len(unique), d, d))[inverse]
    if solver != 'magnus':
        raise ValueError('solver must be dop853 or magnus')
    t_prev = 0.0
    U = eye.copy()
    s3 = np.sqrt(3.0)
    for i, t_target in enumerate(ts):
        duration = float(t_target - t_prev)
        if duration < -1e-13:
            raise ValueError('unsorted times')
        if duration > 0:
            n = int(np.ceil(duration / max_step))
            h = duration / n
            for k in range(n):
                t = t_prev + k * h
                t1 = t + (0.5 - s3 / 6.0) * h
                t2 = t + (0.5 + s3 / 6.0) * h
                H1 = hamiltonian(t1)
                H2 = hamiltonian(t2)
                A1 = -1j * H1
                A2 = -1j * H2
                # Standard fourth-order Gauss-Magnus logarithm:
                # h(A1+A2)/2 - sqrt(3)*h^2[A1,A2]/12.
                Omega = (h / 2) * (A1 + A2) - (s3 * h * h / 12) * (A1 @ A2 - A2 @ A1)
                U = expm(Omega) @ U
        ans[i] = U
        t_prev = float(t_target)
    return ans


def transition(U_late: np.ndarray, U_early: np.ndarray) -> np.ndarray:
    """U(late, early) from U(t,0). Valid for late >= early."""
    return U_late @ U_early.conj().T


def state_at(rho0: np.ndarray, U: np.ndarray) -> np.ndarray:
    rho = U @ rho0 @ U.conj().T
    return (rho + rho.conj().T) / 2


def expectation(rho: np.ndarray, A: np.ndarray) -> float:
    return float(np.trace(rho @ A).real)


def diagnostics(system: FiniteSystem, times, unitaries, *, rho0=None) -> dict:
    if rho0 is None:
        rho0 = system.canonical()
    ts = np.asarray(times)
    U = np.asarray(unitaries)
    identity = system.identity
    defects = [np.linalg.norm(u.conj().T @ u - identity, ord=2) for u in U]
    states = [state_at(rho0, u) for u in U]
    ps = np.array([[expectation(r, p) for p in system.projectors] for r in states])
    ns = ps @ np.arange(len(system.projectors))
    energies = np.array([expectation(r, system.H(t)) for t, r in zip(ts, states)])
    work = float(np.trapezoid(system.offset_derivative * ns, ts)) if len(ts) > 1 else 0.0
    delta_e = float(energies[-1] - energies[0])
    return dict(unitarity_max=float(max(defects)), prob_sum_max=float(np.max(np.abs(ps.sum(axis=1)-1))),
                minimum_probability=float(ps.min()), energy_change=delta_e,
                work_trapezoid=work, work_integration_error=float(abs(work-delta_e)),
                probabilities=ps, energies=energies)
