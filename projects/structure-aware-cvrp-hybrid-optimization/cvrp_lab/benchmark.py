"""Reproducible ablations; all recorded objectives are independently audited."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
from time import perf_counter

import numpy as np
import scipy

from .adapters import availability, external
from .domain import Instance, audit, exact_cost, generated, initial_routes, load
from .moves import Move, State
from .search import metrics, run


def micro(sizes=(10, 30, 60), seeds=(11, 12, 13), count=5000):
    rows = []
    for n in sizes:
        for seed in seeds:
            base = generated(seed, n)
            p = Instance(base.costs, np.r_[0, np.ones(n, dtype=int)], n, 1)
            rng = random.Random(seed)
            route = list(range(1, n+1))
            rng.shuffle(route)
            state = State(p, [route])
            moves = []
            for _ in range(count):
                i = rng.randrange(n-1)
                moves.append(Move('two_opt', 0, i, 0, rng.randrange(i+2, n+1)))
            totals, timings = {}, {}
            # Alternate execution order. This is a kernel experiment, not end-to-end speed.
            order = ('full', 'incremental') if seed % 2 else ('incremental', 'full')
            for mode in order:
                state.evaluate(moves[0], mode)
                tick = perf_counter()
                totals[mode] = [state.evaluate(m, mode).delta for m in moves]
                timings[mode] = perf_counter()-tick
            if totals['full'] != totals['incremental']:
                raise AssertionError('delta mismatch')
            rows.append(dict(n=n, seed=seed, evaluations=count, **timings,
                             ratio_full_over_incremental=timings['full']/timings['incremental']))
    return rows


def root_alns(sizes=(18, 36), seeds=(11, 12, 13), iterations=100):
    from alns_vrp.alns import ALNSConfig, solve_alns
    from alns_vrp.data import demo_instance
    rows = []
    for n in sizes:
        for seed in seeds:
            p = demo_instance(seed, n)
            results, seconds = {}, {}
            for mode in (('full', 'incremental') if seed % 2 else ('incremental', 'full')):
                tick = perf_counter()
                results[mode] = solve_alns(p, ALNSConfig(iterations=iterations,
                    segment_length=20, local_search_evaluation=mode), seed=seed)
                seconds[mode] = perf_counter()-tick
            a, b = results['full'], results['incremental']
            costs_equal = np.allclose([x.best_cost for x in a.records],
                                     [x.best_cost for x in b.records], rtol=1e-10, atol=1e-8)
            choices_equal = all((x.destroy_operator, x.repair_operator, x.accepted) ==
                                (y.destroy_operator, y.repair_operator, y.accepted)
                                for x, y in zip(a.records, b.records))
            if not (costs_equal and choices_equal and a.best_solution.routes == b.best_solution.routes):
                raise AssertionError('ALNS trajectories differ; do not attribute speed to evaluator alone')
            rows.append(dict(n=n, seed=seed, iterations=iterations, identical_trajectory=True,
                             objective=a.best_cost, full=seconds['full'], incremental=seconds['incremental'],
                             ratio_full_over_incremental=seconds['full']/seconds['incremental']))
    return rows


def suite(sizes, seeds, seconds, proposals):
    rows, inputs = [], []
    for n in sizes:
        for seed in seeds:
            tick = perf_counter()
            p = generated(seed, n, clustered=(seed % 2 == 0))
            input_seconds = perf_counter()-tick
            initial = initial_routes(p)
            inputs.append(dict(instance=p.as_dict(), sha256=p.fingerprint(),
                               input_construction_seconds=input_seconds))
            group = []
            variants = ['full', 'incremental', 'expanded']
            for variant in variants:
                r = run(p, variant, seed=seed, proposals=proposals, initial=initial)
                group.append(dict(r, regime='fixed_proposals'))
            if group[0]['routes'] != group[1]['routes']:
                raise AssertionError('fixed-proposal evaluation ablation diverged')
            # Rotate order to reduce systematic warm-up/order bias.
            variants = ['full', 'incremental', 'expanded', 'sequential', 'iterative', 'mip']
            rotate = seed % len(variants)
            for variant in variants[rotate:]+variants[:rotate]:
                r = run(p, variant, seed=seed, seconds=seconds, initial=initial,
                        mip_slice=min(.25, seconds*.4), mip_every=300)
                group.append(dict(r, regime='fixed_time'))
            reference = exact_cost(p) if n <= 11 else min(r['objective'] for r in group)
            for r in group:
                if audit(p, r['routes']) != r['objective']:
                    raise AssertionError('final audit failed')
                r.update(metrics(r, reference, proven=n <= 11))
                r.update(instance_name=p.name, instance_sha256=p.fingerprint(), n=n,
                         start_protocol='common_supplied_initial',
                         reference_type='exact_DP' if n <= 11 else 'best_observed_in_this_run')
                rows.append(r)
    return rows, inputs


def provenance():
    root = Path(__file__).parent
    return {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
            'platform': platform.platform(), 'processor': platform.processor(),
            'cpu_count': os.cpu_count(), 'OMP_NUM_THREADS': os.getenv('OMP_NUM_THREADS'),
            'OPENBLAS_NUM_THREADS': os.getenv('OPENBLAS_NUM_THREADS'),
            'highs_threads': 1, 'source_sha256': {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))}}


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if report.get('runs'):
        keys = ['instance_name', 'n', 'seed', 'regime', 'variant', 'initial_objective', 'objective',
                'elapsed_seconds', 'overrun_seconds', 'proposals', 'valid_evaluations',
                'reference_type', 'reference_gap', 'global_lower_bound', 'certified_gap',
                'observed_time_to_target', 'target_not_reached']
        with path.with_suffix('.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(report['runs'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['suite', 'micro', 'alns', 'external'], default='suite')
    parser.add_argument('--out', default='results/run.json')
    parser.add_argument('--sizes', type=int, nargs='+', default=[6, 30, 60])
    parser.add_argument('--seeds', type=int, nargs='+', default=[11, 12, 13])
    parser.add_argument('--seconds', type=float, default=.5)
    parser.add_argument('--proposals', type=int, default=5000)
    parser.add_argument('--instance', help='common integer-cost JSON instance for external comparison')
    args = parser.parse_args()
    if not args.sizes or min(args.sizes) < 2 or not args.seeds or min(args.seeds) < 0:
        parser.error('valid sizes and nonnegative seeds required')
    report = {'environment': provenance(), 'external_availability': availability(),
              'protocol': vars(args), 'runs': []}
    if args.mode == 'suite':
        report['runs'], report['inputs'] = suite(args.sizes, args.seeds, args.seconds, args.proposals)
    elif args.mode == 'micro':
        report['micro'] = micro(args.sizes, args.seeds, args.proposals)
    elif args.mode == 'alns':
        report['alns'] = root_alns(args.sizes, args.seeds, args.proposals)
    else:
        p = load(args.instance) if args.instance else generated(args.seeds[0], args.sizes[0])
        report['instance'] = p.as_dict()
        # Commercial/dedicated comparisons are cold-start, separate from native warm comparisons.
        report['external'] = [external(p, name, args.seconds, seed, focus=focus)
            for seed in args.seeds for name, focus in [('gurobi', 0), ('gurobi', 1),
                                                       ('hexaly', 0), ('pyvrp', 0)]]
    write_report(args.out, report)


if __name__ == '__main__':
    main()
