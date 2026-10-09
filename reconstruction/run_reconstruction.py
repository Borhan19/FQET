#!/usr/bin/env python3
"""FQET reconstructed numerical pipeline (NOT the original simulation code).

Examples:
    python run_reconstruction.py smoke
    python run_reconstruction.py geometry
    python run_reconstruction.py compare --model ising --g 1.5 --horizons 0.03 --events 1 3
    python run_reconstruction.py calibrate --horizons 0.05 0.1 0.2 --events 1 3
    python run_reconstruction.py holdout --sheet results/bh_sheet.npz --max-points 2

Full 81x15x8 table (very expensive, do not run as a quick test):
    python run_reconstruction.py calibrate --full --sheet results/bh_sheet_full.npz
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from fqet_reconstructed.models import bose_hubbard, transverse_ising, independently_rebuilt_hamiltonian
from fqet_reconstructed.dynamics import evolve_at, diagnostics
from fqet_reconstructed.observables import cross_platform_geometry, required_times
from fqet_reconstructed.architectures import screen_architectures
from fqet_reconstructed.datasheet import calibrate_bh, validate_bh_sheet, audit_parameter_grid


def json_plain(o):
    if isinstance(o, dict):
        return {str(k):json_plain(v) for k,v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_plain(x) for x in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def save_result(output, record):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(json_plain(record), indent=2, allow_nan=False)+'\n')
    print('Saved:', output)


def run_geometry():
    cases = [('bh-local', bose_hubbard(),.2,.025,.05),
             ('bh-practical', bose_hubbard(),.2,.1,.2),
             ('ising-local',transverse_ising(),1.,.06,.2),
             ('ising-nonlocal',transverse_ising(),1.,.06,.7)]
    out = []
    for label, sys, tc, s1, s2 in cases:
        times=required_times(tc,lags=(s1,s2))
        U=dict(zip(times,evolve_at(sys,times)))
        g=cross_platform_geometry(sys,tc,U,s1,s2)
        g['case']=label
        out.append(g)
        print(f'{label:16s} O1={g["first_layer"]["overlap"]:.9f} '
              f'O2={g["second_layer"]["overlap"]:.9f} '
              f'Gram={g["normalized_gram_cond"]:.6f}')
    return dict(status='recomputed microscopic operator geometry', cases=out)


def run_smoke():
    data = []
    for model in [bose_hubbard(), transverse_ising()]:
        times = np.linspace(0,.4,41)
        A = evolve_at(model,times,solver='dop853')
        B = evolve_at(model,times,solver='magnus',max_step=.01,independent_model=True)
        normdiff = np.linalg.norm(A[-1]-B[-1],2)
        den = diagnostics(model,times,A)
        d = dict(model=model.kind,dimension=model.dim,
                 maximum_H_difference=float(np.linalg.norm(model.H(.2)-independently_rebuilt_hamiltonian(model,.2),2)),
                 dop853_vs_independent_magnus=float(normdiff),
                 maximum_unitarity_defect=den['unitarity_max'],
                 work_minus_energy=den['work_integration_error'])
        data.append(d)
        print(model.kind, json.dumps(d,indent=2))
    return dict(status='independent propagation and isolated work check',cases=data)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    subs=p.add_subparsers(dest='command',required=True)
    subs.add_parser('smoke')
    subs.add_parser('geometry')
    comp=subs.add_parser('compare')
    comp.add_argument('--model',choices=['bh','ising'],default='ising')
    comp.add_argument('--g',type=float,default=1.5)
    comp.add_argument('--horizons',type=float,nargs='+',default=[.03])
    comp.add_argument('--events',type=int,nargs='+',default=[1])
    comp.add_argument('--history-grid',type=float,nargs='+',default=None)
    comp.add_argument('--rounds',type=int,default=2)
    comp.add_argument('--max-cuts',type=int,default=250)
    comp.add_argument('--max-pairs',type=int,default=None)
    comp.add_argument('--checkpoint',type=str,default=None,help='save/resume completed architecture evaluations')
    cal=subs.add_parser('calibrate')
    cal.add_argument('--full',action='store_true')
    cal.add_argument('--horizons',type=float,nargs='+',default=[.05,.10,.20])
    cal.add_argument('--events',type=int,nargs='+',default=[1,3])
    cal.add_argument('--sheet',default='results/bh_sheet.npz')
    cal.add_argument('--max-cuts',type=int,default=250)
    cal.add_argument('--resume',action='store_true',help='resume a matching incomplete response sheet checkpoint')
    cal.add_argument('--max-corners',type=int,default=None,help='process at most this many unfinished corners per run')
    cal.add_argument('--assumed-shots-per-setting',type=float,default=1e6)
    hold=subs.add_parser('holdout')
    hold.add_argument('--sheet',default='results/bh_sheet.npz')
    hold.add_argument('--max-points',type=int,default=2)
    hold.add_argument('--magnus-step',type=float,default=.0025)
    hold.add_argument('--seed',type=int,default=20261009)
    hold.add_argument('--out-csv',default='results/reconstructed_holdout.csv')
    grid=subs.add_parser('grid-audit')
    grid.add_argument('--sheet',default='results/bh_sheet_15events.npz')
    grid.add_argument('--grid-size',type=int,default=5)
    grid.add_argument('--out-csv',default='results/reconstructed_parameter_grid.csv')
    p.add_argument('--out-json',default=None)
    a=p.parse_args()
    if a.command=='smoke': result=run_smoke()
    elif a.command=='geometry':result=run_geometry()
    elif a.command=='compare':
        sys=bose_hubbard() if a.model=='bh' else transverse_ising(g=a.g)
        tc=.2 if a.model=='bh' else 1.0
        grid=a.history_grid if a.history_grid is not None else ([.05,.1,.2] if a.model=='bh' else [.06,.2,.7])
        result=screen_architectures(sys,tc,a.horizons,a.events,history_grid=grid,
                                    max_pairs=a.max_pairs,rounds=a.rounds,max_cuts=a.max_cuts,
                                    checkpoint=a.checkpoint)
    elif a.command=='calibrate':
        hs=np.linspace(0,.2,81) if a.full else a.horizons
        ev=list(range(1,16)) if a.full else a.events
        result=calibrate_bh(a.sheet,horizons=hs,masks=ev,max_cuts=a.max_cuts,
                             assumed_shots_per_setting=a.assumed_shots_per_setting,
                             resume=a.resume,max_corners=a.max_corners)
    elif a.command=='holdout':
        result=validate_bh_sheet(a.sheet,max_points=a.max_points,solver_step=a.magnus_step,
                                 seed=a.seed,output_csv=a.out_csv)
    elif a.command=='grid-audit':
        result=audit_parameter_grid(a.sheet,grid_size=a.grid_size,output_csv=a.out_csv)
    if a.out_json:
        save_result(a.out_json,result)
    else:
        print(json.dumps(json_plain(result),indent=2,allow_nan=False)[:16000])

if __name__=='__main__':main()
