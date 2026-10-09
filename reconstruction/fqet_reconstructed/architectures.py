"""Task-aware representation comparison and experimental resource accounting.

Uses the exact finite-system operators, convex fixed-shot coefficient fits,
and a numerically optimized shared allocation. The alternation and lag search
are explicit HEURISTICS: they cannot certify a global optimum of SM Eq. (S27).
"""
from __future__ import annotations
from itertools import combinations
import json
import hashlib
from pathlib import Path
import numpy as np
from .dynamics import evolve_at, state_at
from .observables import required_times, make_candidate, target_at
from .optimization import spectral_fit, optimize_shared_shots


def architecture_candidates(tc, *, history_grid=(.05,.1,.2), max_pairs=None):
    """Screen both nearby and nonlocal lags; avoid only-local searches."""
    grid = sorted(set(float(s) for s in history_grid if 0 < s <= tc))
    out = [('ERJ', (), 1), ('ERJK2', (), 2)]
    out += [('H2', (s,), 0) for s in grid]
    pairs = list(combinations(grid, 2))
    if max_pairs:
        pairs = pairs[:max_pairs]
    out += [('H3', p, 0) for p in pairs]
    return out


def evaluate_architecture(system, tc, horizons, event_masks, *,
                          lags=(), derivative_order=0, initial_shots=1e6,
                          rounds=2, tolerance=.01, prep_radius=.01,
                          family_confidence=.99, max_cuts=200, gap=2e-7,
                          anchor_nominal=True):
    """Run fixed-lag operator fits and shared metrology iterations.

    All targets share the same physical Ns per setting, but have separately
    optimized constitutive coefficients. Re-running the spectral problems
    with the new Ns approximates joint coefficient/shot co-design.
    """
    hs = np.array(sorted(set(map(float, horizons))), dtype=float)
    masks = np.array(sorted(set(map(int, event_masks))), dtype=int)
    if not len(hs) or not len(masks):
        raise ValueError('need at least one horizon and event')
    tt = required_times(tc, lags=lags, horizons=hs)
    U = dict(zip(tt, evolve_at(system, tt, solver='dop853')))
    cand = make_candidate(system, tc=tc, U_samples=U, history_lags=lags,
                          order=derivative_order)
    rhoc = state_at(system.canonical(), U[tc])
    targets = [target_at(system, tc=tc, horizon=float(h), event_mask=int(k), U_samples=U)
               for h in hs for k in masks]
    Ns = np.full(len(cand.groups), initial_shots, dtype=float)
    fits = []
    alloc = {}
    history = []
    for iteration in range(int(rounds)):
        fits = [spectral_fit(cand, X, shot_counts=Ns, kappa=min(2.,4*prep_radius),
                             M=len(hs)*len(masks), alpha=1-family_confidence,
                             anchor_state=rhoc if anchor_nominal else None,
                             max_cuts=max_cuts, atol=gap)
                for X in targets]
        alloc = optimize_shared_shots(fits, tolerance, kappa=min(2.,4*prep_radius),
                                      M=len(hs)*len(masks), alpha=1-family_confidence)
        history.append(dict(round=iteration+1, feasible=bool(alloc['feasible']),
                            total_shots=alloc.get('total_shots'),
                            largest_spectral_distance=float(max(f.distance_upper for f in fits)),
                            unconverged_fits=sum(not f.converged for f in fits)))
        if not alloc['feasible']:
            break
        Ns = np.asarray(alloc['shots'], dtype=float)
    return dict(model=system.kind, architecture=cand.label,
                lags=list(lags), derivative_order=derivative_order,
                horizons=hs.tolist(), masks=masks.tolist(),
                setting_names=[g.name for g in cand.groups],
                feasible=bool(alloc['feasible']),
                total_shots=alloc.get('total_shots'),
                setting_shots=alloc.get('shots'),
                largest_spectral_distance=float(max(f.distance_upper for f in fits)),
                largest_cut_gap=float(max(f.objective_upper-f.objective_lower for f in fits)),
                nonconverged=sum(not f.converged for f in fits),
                optimization_status=alloc.get('message', alloc.get('reason','')),
                iterations=history,
                note='Reconstructed task-aware ALTERNATING/SCREENED optimum, not manuscript global-search source')


def screen_architectures(system, tc, horizons, event_masks, *,
                         history_grid, max_pairs=None, checkpoint=None, **kwargs):
    specs = architecture_candidates(tc, history_grid=history_grid,max_pairs=max_pairs)
    parameters = dict(kind=system.kind, tc=float(tc), g=float(system.transverse_field),
                      horizons=list(map(float,horizons)), masks=list(map(int,event_masks)),
                      architecture_specs=[[a,list(b),c] for a,b,c in specs],options=kwargs)
    fingerprint = hashlib.sha256(json.dumps(parameters,sort_keys=True).encode()).hexdigest()
    path = Path(checkpoint) if checkpoint else None
    results = []
    if path and path.exists():
        saved=json.loads(path.read_text())
        if saved.get('fingerprint') != fingerprint:
            raise ValueError('checkpoint does not match current architecture task')
        results=saved['raw_results']
        print(f'  resuming from checkpoint with {len(results)} completed architectures',flush=True)
    for label, lags, order in specs[len(results):]:
        out = evaluate_architecture(system, tc, horizons, event_masks,
                                    lags=lags, derivative_order=order, **kwargs)
        results.append(out)
        print(f'  {label:6s} lags={str(lags):18s} shots={out["total_shots"]}', flush=True)
        if path:
            path.parent.mkdir(parents=True,exist_ok=True)
            tmp=path.with_suffix(path.suffix+'.tmp')
            tmp.write_text(json.dumps(dict(fingerprint=fingerprint,parameters=parameters,
                                           raw_results=results),indent=2)+'\n')
            tmp.replace(path)
    # Enforce support nesting explicitly. A richer candidate can always set
    # its added coefficients and repetitions to zero, so its feasible cost
    # cannot exceed the cost of an embedded smaller feasible candidate.
    # This does not assert the raw alternating optimization found the optimum.
    for result in results:
        result['raw_alternating_shots'] = result['total_shots']
        result['active_support'] = result['architecture']
        result['effective_shots'] = result['total_shots']
        result['inherited_from'] = None
    for result in results:
        if not result['feasible']:
            continue
        subcases = []
        if result['architecture'] == 'ERJK2':
            subcases = [r for r in results if r['architecture'] == 'ERJ']
        elif result['architecture'] == 'H3':
            # Either historical time alone can be retained.
            subcases = [r for r in results if r['architecture'] == 'H2'
                        and len(r['lags']) == 1 and r['lags'][0] in result['lags']]
        for base in subcases:
            if base['feasible'] and base['effective_shots'] < result['effective_shots']:
                result['effective_shots'] = base['effective_shots']
                result['active_support'] = base['active_support']
                result['inherited_from'] = {'architecture':base['architecture'],
                                            'lags':base['lags'],
                                            'note':'zero coefficients for extra layer; do not pay for inactive settings'}
    best_hist = min((r for r in results if r['architecture'].startswith('H') and r['feasible']),
                    key=lambda r:r['effective_shots'], default=None)
    best_inst = min((r for r in results if r['architecture'].startswith('ERJ') and r['feasible']),
                    key=lambda r:r['effective_shots'], default=None)
    return dict(results=results, best_history=best_hist, best_instantaneous=best_inst,
                global_optimality_certified=False,
                method='finite lag screen + fixed-shot spectral LP + alternating shared shot allocation')
