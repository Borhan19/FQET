"""Reconstructed eight-corner Bose-Hubbard table and independent holdout.

Implements the physics of FQET SM Notes 7, 9, and 10. Original optimizer
seeds, numerical coefficient arrays, tube directions, sensitivity envelopes,
and hidden validation points were not supplied. NEW results from this module
are therefore reconstructed, not a rerun of the original published campaign.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path
import json
import numpy as np
from scipy.stats import qmc
from .models import bose_hubbard
from .dynamics import evolve_at, state_at, expectation
from .observables import event_masks, event_operator, make_candidate, target_at, required_times
from .optimization import spectral_fit


CORNERS = tuple(product([0, 1], repeat=3))
LAGS = (0.10, 0.20)
TC = 0.20


def param_coordinates(x):
    x = np.asarray(x, dtype=float)
    if x.shape != (3,) or np.any(x < 0.9-1e-12) or np.any(x > 1.1+1e-12):
        raise ValueError('parameters must lie inside the declared [0.9, 1.1]^3 box')
    return np.clip((x - 0.9) / 0.2, 0, 1)


def interpolate(corners, scales):
    """Trilinear response, FQET Eq. (S77), first axis ordered as CORNERS."""
    t = param_coordinates(scales)
    a = np.asarray(corners)
    if a.shape[0] != 8:
        raise ValueError('need eight corner arrays, in CORNERS order')
    out = np.zeros(a.shape[1:], dtype=float)
    for i, corner in enumerate(CORNERS):
        weight = np.prod([t[j] if corner[j] else 1-t[j] for j in range(3)])
        out += weight * a[i]
    return out


def system_at(scales):
    u, l, t = np.asarray(scales, dtype=float)
    return bose_hubbard(U=(1.3 / 3.0) * u, lam0=20.0 * l, tau=20.0 * t)


def calibrate_bh(out_path, *, horizons, masks,
                 assumed_shots_per_setting=1e6, max_cuts=250,
                 gap=2e-7, anchor=True, resume=False, max_corners=None):
    """Compute eight operator-fit coefficient tables from scratch.

    The coefficient objective is a reconstructed fixed-shot spectral LP,
    not the unpublished original simultaneous outer global optimization.
    The campaign checkpoints after EVERY forecast horizon (15 fits in full
    mode) so hours of computational work are not lost if interrupted.
    
    In full mode: 8 corners x 81 future times x 15 binary events = 9720 fits.
    """
    horizons = np.array(sorted(set(map(float, horizons))), dtype=float)
    masks = np.array(sorted(set(map(int, masks))), dtype=int)
    if not len(horizons) or np.any(horizons < 0) or np.any(horizons > 0.2+1e-12):
        raise ValueError('horizons must lie in [0,0.2]')
    if any(int(k) not in event_masks() for k in masks):
        raise ValueError('event masks must be in 1..15')
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m0=system_at(np.ones(3))
    dummy_times=required_times(TC,lags=LAGS)
    dummy_U=dict(zip(dummy_times,evolve_at(m0,dummy_times)))
    nc=len(make_candidate(m0,tc=TC,U_samples=dummy_U,history_lags=LAGS).operators)
    shape=(8,len(horizons),len(masks))
    metadata = {'status':'RECONSTRUCTED, not original reported coefficients',
                'origin':'FQET 9-Oct-2026 SM Notes 7-10',
                'optimization':'fixed-shot spectral cutting-plane with optional nominal anchor; NOT original joint co-design',
                'eps_prep':0.01, 'eps_task':0.01, 'family_confidence':0.99,
                'tc':TC, 'lags':list(LAGS),
                'assumed_shots_per_setting':assumed_shots_per_setting,
                'masks':masks.tolist(), 'horizons':horizons.tolist(),
                'params_axes':['U/U0','Lambda0/Lambda00','tau_BC/tau_BC0'],
                'corner_order':[list(c) for c in CORNERS],
                'anchor_nominal_state':bool(anchor), 'cut_gap_requested':gap}
    if resume:
        if not out_path.exists():
            raise FileNotFoundError('cannot resume: no previous checkpoint: '+str(out_path))
        old=np.load(out_path,allow_pickle=False)
        previous=json.loads(str(old['metadata']))
        if previous != metadata:
            raise ValueError('resume settings mismatch (horizons, events or optimization)')
        coeffs=old['coefficients'].copy()
        intercepts=old['intercepts'].copy()
        distances=old['distance_upper'].copy()
        gaps=old['objective_gap'].copy()
        converged=old['converged'].copy()
        completed=old['completed'].copy()
        old.close()
    else:
        coeffs=np.zeros((*shape,nc))
        intercepts=np.zeros(shape)
        distances=np.zeros(shape)
        gaps=np.zeros(shape)
        converged=np.zeros(shape,dtype=bool)
        completed=np.zeros(shape[:2],dtype=bool)
    def checkpoint():
        # Save in the target directory, then atomically replace finished file.
        temp=out_path.with_suffix(out_path.suffix+'.tmp')
        with temp.open('wb') as stream:
            np.savez_compressed(stream,coefficients=coeffs,intercepts=intercepts,
                               distance_upper=distances,objective_gap=gaps,
                               converged=converged,completed=completed,
                               horizons=horizons,masks=masks,metadata=json.dumps(metadata))
        temp.replace(out_path)
    executed=0
    for ix, corner in enumerate(CORNERS):
        if max_corners is not None and executed >= max_corners:
            break
        if np.all(completed[ix]):
            continue
        param = np.array([0.9 if x == 0 else 1.1 for x in corner])
        m = system_at(param)
        tt = required_times(TC, lags=LAGS, horizons=horizons)
        U = dict(zip(tt, evolve_at(m, tt, solver='dop853')))
        candidate = make_candidate(m, tc=TC, U_samples=U, history_lags=LAGS)
        rho_c = state_at(m.canonical(), U[TC])
        for j, h in enumerate(horizons):
            if completed[ix,j]:
                continue
            for k, mask in enumerate(masks):
                X = target_at(m, tc=TC, horizon=float(h), event_mask=int(mask), U_samples=U)
                fit = spectral_fit(candidate, X, kappa=0.04,
                                   shot_counts=np.full(len(candidate.groups), assumed_shots_per_setting),
                                   M=len(masks)*len(horizons), alpha=.01,
                                   anchor_state=rho_c if anchor else None,
                                   max_cuts=max_cuts, atol=gap)
                coeffs[ix,j,k] = fit.coefficients
                intercepts[ix,j,k] = fit.intercept
                distances[ix,j,k] = fit.distance_upper
                gaps[ix,j,k] = fit.objective_upper-fit.objective_lower
                converged[ix,j,k] = fit.converged
            completed[ix,j]=True
            checkpoint()
        executed+=1
        print(f'  reconstructed corner {ix+1}/8: scales={param}, '
              f'max spectral distance={distances[ix].max():.6g}, '
              f'nonconverged={int(np.sum(~converged[ix]))}', flush=True)
    return {'path':str(out_path),'corner_count':8, 'horizon_count':len(horizons),
            'event_count':len(masks),'completed_corner_horizon_tasks':int(completed.sum()),
            'required_corner_horizon_tasks':int(completed.size),
            'all_completed':bool(np.all(completed)),
            'nonconverged_among_completed':int(np.sum(~converged & completed[:,:,None])),
            'max_spectral_distance':float(distances.max()), 'metadata':metadata}


def _random_tube_states(rho0, count=3, radius=.0098, seed=0):
    """Positive perturbations at prescribed trace distance from canonical rho0."""
    rng = np.random.default_rng(seed)
    d = rho0.shape[0]
    out = [rho0]
    for _ in range(count):
        z = rng.normal(size=d) + 1j*rng.normal(size=d)
        z = z / np.linalg.norm(z)
        sigma = np.outer(z, z.conj())
        one_dist = np.sum(np.abs(np.linalg.eigvalsh(sigma-rho0)))/2
        eta = radius/one_dist
        assert 0 <= eta <= 1
        other = (1-eta)*rho0+eta*sigma
        assert abs(np.sum(np.abs(np.linalg.eigvalsh(other-rho0)))/2-radius) < 1e-9
        out.append(other)
    return out


def _holdout_points(seed=20261009):
    interior = 0.9 + 0.2 * qmc.Sobol(d=3, scramble=True, seed=seed).random_base2(m=6)
    near_corner = np.array(list(product([0.905,1.095], repeat=3)))
    return [(x, 'sobol') for x in interior] + [(x, 'stress') for x in near_corner]


def validate_bh_sheet(npz_path, *, max_points=72, solver_step=.0025,
                      seed=20261009, output_csv=None):
    """Independent GM propagated holdout, with entirely NEW random seeds.

    Full 72x4x81x15 requires an 81-horizon, 15-event sheet and max_points=72.
    It is inappropriate to claim this reproduces original 0.1278% result.
    """
    import pandas as pd
    z = np.load(npz_path, allow_pickle=False)
    if 'completed' not in z or not np.all(z['completed']):
        raise RuntimeError('Do not run a holdout on an incomplete or uncheckpointed response table')
    if not np.all(z['converged']):
        raise RuntimeError('Response table has unresolved cutting-plane gaps; inspect before holdout')
    meta = json.loads(str(z['metadata']))
    t0 = meta['tc']
    lags = tuple(meta['lags'])
    horizons = z['horizons']
    masks = z['masks']
    coeffs = z['coefficients']
    intercepts = z['intercepts']
    points = _holdout_points(seed)[:max_points]
    records = []
    for n, (param, kind) in enumerate(points):
        m = system_at(param)
        tt = required_times(t0, lags=lags, horizons=horizons)
        U = dict(zip(tt, evolve_at(m, tt, solver='magnus', max_step=solver_step, independent_model=True)))
        candidate = make_candidate(m, tc=t0, U_samples=U, history_lags=lags)
        co = interpolate(coeffs, param)
        ab = interpolate(intercepts, param)
        preps = _random_tube_states(m.canonical(), count=3, seed=seed + 193*n)
        worst = 0.0
        worst_detail = None
        for i_prep, rho0 in enumerate(preps):
            rc = state_at(rho0, U[t0])
            x = np.array([expectation(rc, a) for a in candidate.operators])
            for i, h in enumerate(horizons):
                rf = state_at(rho0, U[float(t0+h)]) if float(t0+h) in U else state_at(rho0, U[min(U, key=lambda q:abs(q-(t0+h)))])
                for j, mask in enumerate(masks):
                    pred = float(ab[i,j] + co[i,j] @ x)
                    truth = expectation(rf, event_operator(m.projectors, int(mask)))
                    err = abs(pred-truth)
                    if err > worst:
                        worst = err
                        worst_detail = (i_prep, float(h), int(mask), truth, pred)
        r = dict(point=n, source=kind, U_scale=float(param[0]),
                 Lambda_scale=float(param[1]), Tau_scale=float(param[2]),
                 max_abs_event_error=float(worst),
                 worst_prep=int(worst_detail[0]) if worst_detail else 0,
                 worst_tau=worst_detail[1] if worst_detail else 0,
                 worst_event=worst_detail[2] if worst_detail else 0,
                 worst_exact=worst_detail[3] if worst_detail else 0,
                 worst_pred=worst_detail[4] if worst_detail else 0)
        records.append(r)
        print(f'  new holdout {n+1}/{len(points)}: error={worst:.6g}', flush=True)
    df = pd.DataFrame.from_records(records)
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    return dict(status='RECONSTRUCTED NEW HOLDOUT; not original frozen coefficients or seeds',
                points=len(points), preparations_per_point=4,
                horizons=len(horizons), events=len(masks),
                predictions=len(points)*4*len(horizons)*len(masks),
                max_abs_error=float(df.max_abs_event_error.max()),
                result_csv=str(output_csv) if output_csv else None)


def audit_parameter_grid(npz_path, *, grid_size=5, output_csv=None):
    """Finite parameter-grid audit of a reconstructed BH response table.

    This is a SAMPLED grid check, not a proof of an envelope between anchors.
    Numerically estimating directional finite differences does not certify
    the derivative supremum needed for the manuscript's 0.7304% result.
    """
    import pandas as pd
    z=np.load(npz_path, allow_pickle=False)
    if 'completed' not in z or not np.all(z['completed']) or not np.all(z['converged']):
        raise RuntimeError('Need a finished, gap-converged eight-corner sheet')
    meta=json.loads(str(z['metadata']))
    horizons=z['horizons']
    masks=z['masks']
    coeffs=z['coefficients']
    ints=z['intercepts']
    rows=[]
    tc=float(meta['tc'])
    lags=tuple(meta['lags'])
    if grid_size<2:
        raise ValueError('grid_size >= 2 required')
    axes=np.linspace(.9,1.1,grid_size)
    maxd=np.zeros((grid_size,)*3)
    for point, idx in enumerate(product(range(grid_size),repeat=3)):
        scales=np.array([axes[k] for k in idx])
        m=system_at(scales)
        ts=required_times(tc,lags=lags,horizons=horizons)
        U=dict(zip(ts,evolve_at(m,ts,solver='dop853')))
        c=make_candidate(m,tc=tc,U_samples=U,history_lags=lags)
        rho_c=state_at(m.canonical(),U[tc])
        x=np.array([expectation(rho_c,op) for op in c.operators])
        c_grid=interpolate(coeffs,scales)
        intercept_grid=interpolate(ints,scales)
        local_max=0.0
        for i,h in enumerate(horizons):
            for j,mask in enumerate(masks):
                target=target_at(m,tc=tc,horizon=float(h),event_mask=int(mask),U_samples=U)
                approximation=c.matrix(c_grid[i,j],intercept_grid[i,j])
                residual=target-approximation
                ev=np.linalg.eigvalsh(residual)
                d=float(max(abs(ev[0]),abs(ev[-1])))
                bias=float(np.trace(rho_c@residual).real)
                pred=float(intercept_grid[i,j]+np.dot(c_grid[i,j],x))
                truth=expectation(rho_c,target)
                assert abs((truth-pred)-bias)<1e-9
                local_max=max(local_max,d)
                rows.append(dict(U_scale=scales[0],Lambda_scale=scales[1],Tau_scale=scales[2],
                                 horizon=float(h),event_mask=int(mask),
                                 distance=d,nominal_bias=bias,
                                 pairwise_tube_width_upper=.04*d,
                                 single_state_tube_error_upper=abs(bias)+.02*d))
        maxd[idx]=local_max
        if (point+1)%25==0 or point==0:
            print(f'  empirical parameter grid {point+1}/{grid_size**3}: max d={local_max:.6g}',flush=True)
    spacing=float(axes[1]-axes[0])
    fd_envelope=[]
    for dim in range(3):
        # Adjacent-point difference quotients are only numerical estimates.
        fd_envelope.append(float(np.max(abs(np.diff(maxd,axis=dim))/spacing)))
    df=pd.DataFrame.from_records(rows)
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True,exist_ok=True)
        df.to_csv(output_csv,index=False)
    return dict(status='EMPIRICAL FINITE GRID ONLY; not a continuous domain certificate',
                grid_size=grid_size,parameter_points=grid_size**3,
                event_time_cases=len(df),
                max_operator_distance=float(df.distance.max()),
                max_pairwise_tube_width_upper_on_grid=float(df.pairwise_tube_width_upper.max()),
                max_single_state_tube_error_upper_on_grid=float(df.single_state_tube_error_upper.max()),
                max_nominal_abs_error_on_grid=float(abs(df.nominal_bias).max()),
                max_adjacent_grid_slopes=fd_envelope,
                warning='These are sampled finite differences, NOT certified global Lipschitz constants; no 0.7304% claim',
                result_csv=str(output_csv) if output_csv else None)
