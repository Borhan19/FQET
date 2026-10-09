"""Unit tests of reconstruction identities; these do not revalidate old CSV data."""
import numpy as np
import pytest
from fqet_reconstructed.models import bose_hubbard, transverse_ising, ising_independent_kron, independently_rebuilt_hamiltonian
from fqet_reconstructed.dynamics import evolve_at, transition, state_at, expectation, diagnostics
from fqet_reconstructed.observables import (MeasurementGroup, Candidate, make_candidate,
    target_at, event_masks, event_operator, required_times, principal_geometry, cross_platform_geometry)
from fqet_reconstructed.optimization import spectral_fit, scalar_shot_allocation, optimize_shared_shots


def test_models_and_energy_force():
    bh = bose_hubbard()
    Is = transverse_ising()
    assert bh.dim == 35 and Is.dim == 16
    assert np.allclose(Is.H(0.32), ising_independent_kron(g=1.5, lam=Is.offset_initial+Is.offset_derivative*.32))
    assert np.linalg.norm(sum(r * P for r, P in enumerate(bh.projectors)) - bh.offset) < 1e-13
    for x in [bh, Is]:
        rho = x.canonical()
        assert abs(np.trace(rho)-1) < 1e-12
        assert np.linalg.eigvalsh(rho).min() >= -1e-14


@pytest.mark.parametrize('kind', ['bh', 'ising'])
def test_independent_propagators(kind):
    m = bose_hubbard() if kind == 'bh' else transverse_ising()
    tt = np.linspace(0, .35, 8)
    A = evolve_at(m, tt, solver='dop853')
    B = evolve_at(m, tt, solver='magnus', max_step=.015, independent_model=True)
    assert np.linalg.norm(m.H(.2) - independently_rebuilt_hamiltonian(m, .2), ord=2) < 1e-12
    assert np.linalg.norm(A[-1] - B[-1], ord=2) < 2e-7
    summary = diagnostics(m, tt, A)
    assert summary['unitarity_max'] < 5e-8
    assert summary['prob_sum_max'] < 5e-8
    assert abs(summary['work_integration_error']) < .02  # coarse trapezoid grid


def test_history_pullback_derivatives():
    m = bose_hubbard()
    tc = .20
    s1 = .10
    horizon = .05
    tt = required_times(tc, lags=[s1], horizons=[horizon])
    Us = dict(zip(tt, evolve_at(m, tt)))
    hist = make_candidate(m, tc=tc, U_samples=Us, history_lags=(s1,))
    deriv = make_candidate(m, tc=tc, U_samples=Us, order=2)
    rho0 = m.canonical()
    rhoc = state_at(rho0, Us[tc])
    rhop = state_at(rho0, Us[float(tc-s1)])
    assert abs(expectation(rhoc, hist.operators[5]) - expectation(rhop, m.projectors[0])) < 3e-9
    future = target_at(m, tc=tc, horizon=horizon, event_mask=7, U_samples=Us)
    rhof = state_at(rho0, Us[tc+horizon])
    assert abs(expectation(rhoc, future) - expectation(rhof, event_operator(m.projectors, 7))) < 3e-9
    # First derivative via symmetric finite difference at t=tc.
    h = 2e-4
    sample = [tc-h, tc, tc+h]
    U3 = evolve_at(m, sample)
    rminus = state_at(rho0, U3[0])
    rplus = state_at(rho0, U3[-1])
    derivative = (expectation(rplus, m.projectors[0])-expectation(rminus, m.projectors[0]))/(2*h)
    assert abs(derivative - expectation(rhoc, deriv.operators[5])) < 5e-7


def test_event_family_and_analytic_qubit_distance():
    assert event_masks() == tuple(range(1, 16))
    I = np.eye(2, dtype=complex)
    Z = np.diag([1, -1]).astype(complex)
    Y = np.array([[0,-1j],[1j,0]], dtype=complex)
    tau = 0.71
    w = 1.7
    X = (I + np.cos(w*tau)*Z + np.sin(w*tau)*Y)/2
    cand = Candidate('qubit-E-R', [Z], ['Z'], [MeasurementGroup('Z','individual',(0,),2)], 0, ())
    fit = spectral_fit(cand, X, atol=1e-9)
    assert fit.converged
    assert abs(fit.distance_upper - abs(np.sin(w*tau))/2) < 1e-8
    assert fit.objective_upper - fit.objective_lower <= 1.1e-9
    assert np.allclose(fit.coefficients[0], np.cos(w*tau)/2, atol=1e-4)


def test_scalar_shots_and_shared_shots():
    b = np.array([2.,3.,4.])
    shots = scalar_shot_allocation(b, .1)
    assert abs(np.sum(b / np.sqrt(shots)) - .1) < 1e-12
    m = bose_hubbard()
    d = .04
    Us = dict(zip([0.,.2,.21], evolve_at(m, [0.,.2,.21])))
    c = make_candidate(m, tc=.2, U_samples=Us, history_lags=(.2,))
    X = target_at(m, tc=.2, horizon=.01, event_mask=1, U_samples=Us)
    fit = spectral_fit(c, X, shot_counts=np.full(len(c.groups), 1e8), kappa=.04, max_cuts=160)
    assert fit.converged
    alloc = optimize_shared_shots([fit], .01, kappa=.04)
    assert alloc['feasible']
    assert alloc['n_settings'] == 3


def test_trilinear_and_geometry():
    m = bose_hubbard()
    times = required_times(.2, lags=(.025,.05,.1,.2))
    Us = dict(zip(times, evolve_at(m, times)))
    geo = cross_platform_geometry(m, .2, Us, .1, .2)
    for key in ['first_layer', 'second_layer']:
        assert 0 <= geo[key]['overlap'] <= 1
    assert geo['normalized_gram_cond'] >= 1
    assert geo['first_layer']['overlap'] == pytest.approx(0.313262, abs=2e-5)
    assert geo['second_layer']['overlap'] == pytest.approx(0.271243, abs=2e-5)
    assert geo['normalized_gram_cond'] == pytest.approx(2305.8896, rel=2e-5)


def test_exact_reported_geometry_four_cases():
    # Independent reconstruction of the SM Table S5 and cross-platform table.
    reference = [
        ('bh', .2, .025, .05, .942613004, .914842165, 392026.2828),
        ('bh', .2, .1, .2, .313262032, .271243446, 2305.88955),
        ('ising', 1., .06, .2, .969165101, .724417293, 29006.5795),
        ('ising', 1., .06, .7, .969165101, .113790187, 1565.41487),
    ]
    for kind,tc,s1,s2,O1,O2,cond in reference:
        m = bose_hubbard() if kind == 'bh' else transverse_ising()
        tt = required_times(tc, (s1,s2))
        u = dict(zip(tt, evolve_at(m, tt)))
        geom = cross_platform_geometry(m,tc,u,s1,s2)
        assert geom['first_layer']['overlap'] == pytest.approx(O1, abs=2e-7)
        assert geom['second_layer']['overlap'] == pytest.approx(O2, abs=2e-7)
        assert geom['normalized_gram_cond'] == pytest.approx(cond, rel=2e-6)


def test_trilinear_interpolation_and_positive_tube():
    from fqet_reconstructed.datasheet import interpolate, CORNERS, _random_tube_states
    coeffs = np.array([np.array([i,j,k,i*j + k]) for (i,j,k) in CORNERS],float)
    for i, corner in enumerate(CORNERS):
        param = np.array([.9 if c == 0 else 1.1 for c in corner])
        assert np.allclose(interpolate(coeffs,param),coeffs[i])
    assert np.allclose(interpolate(coeffs,np.ones(3)), coeffs.mean(axis=0))
    rho = bose_hubbard().canonical()
    preps = _random_tube_states(rho,count=3,seed=1902)
    assert len(preps) == 4
    for x in preps:
        assert np.linalg.eigvalsh(x).min() > -1e-12
        assert np.trace(x).real == pytest.approx(1.,abs=1e-12)
    for x in preps[1:]:
        trace_dist = .5 * np.sum(np.abs(np.linalg.eigvalsh(x-rho)))
        assert trace_dist == pytest.approx(.0098,abs=1e-10)


def test_magnus_step_halving():
    m = transverse_ising(g=1.5)
    tt = [0.0, 0.2, 1.0]
    exact = evolve_at(m, tt, solver='dop853')[-1]
    e1 = np.linalg.norm(exact-evolve_at(m, tt, solver='magnus', max_step=.08, independent_model=True)[-1],ord=2)
    e2 = np.linalg.norm(exact-evolve_at(m, tt, solver='magnus', max_step=.04, independent_model=True)[-1],ord=2)
    assert e2 < e1 / 10 # expected ~16x for 4th order method


def test_extended_support_inheritance():
    from fqet_reconstructed.architectures import screen_architectures
    sys=transverse_ising()
    result=screen_architectures(sys,1.0,[.03],[1],history_grid=[.06,.2],rounds=1,max_cuts=120)
    by_kind={r['architecture']:r for r in result['results'] if r['architecture'].startswith('ER')}
    assert by_kind['ERJK2']['effective_shots'] <= by_kind['ERJ']['effective_shots']
    h2={r['lags'][0]:r for r in result['results'] if r['architecture']=='H2'}
    h3=next(r for r in result['results'] if r['architecture']=='H3')
    assert h3['effective_shots'] <= min(v['effective_shots'] for v in h2.values())
