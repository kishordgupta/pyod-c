#!/usr/bin/env python3
"""Time PyOD and native ports on shared CSV data. See docs/BENCHMARKING.md."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

MODELS = ('ABOD', 'COPOD', 'ECOD', 'HBOS', 'IForest', 'KNN', 'LOF', 'PCA')


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--c-root', type=Path)
    parser.add_argument('--rs-root', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--train', type=positive, default=2000)
    parser.add_argument('--test', type=positive, default=1000)
    parser.add_argument('--features', type=positive, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--repeats', type=positive, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--threads', type=positive, help='limit Python BLAS/OpenMP pools; omitted = library defaults')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error('--warmup must be nonnegative')
    if args.train < 30 or args.test < 30:
        parser.error('use at least 30 training and test rows')
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('--out must be a new or empty directory')

    # Set backend-specific limits before numerical libraries are loaded.
    if args.threads:
        for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                         'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[variable] = str(args.threads)
    import numpy as np
    from contextlib import nullcontext
    from sklearn.metrics import average_precision_score, roc_auc_score
    from threadpoolctl import threadpool_info, threadpool_limits
    from pyod.utils.data import generate_data
    from pyod.models.abod import ABOD
    from pyod.models.copod import COPOD
    from pyod.models.ecod import ECOD
    from pyod.models.hbos import HBOS
    from pyod.models.iforest import IForest
    from pyod.models.knn import KNN
    from pyod.models.lof import LOF
    from pyod.models.pca import PCA
    constructors = dict(ABOD=ABOD, COPOD=COPOD, ECOD=ECOD, HBOS=HBOS,
                        IForest=lambda: IForest(random_state=42), KNN=KNN, LOF=LOF, PCA=PCA)
    binaries = {}
    for backend, root, relative in (
        ('pyod-c', args.c_root, 'build/examples/benchmark_csv'),
        ('pyod-rs', args.rs_root, 'target/release/examples/benchmark_csv'),
    ):
        if root:
            binary = (root / relative).resolve()
            if not binary.is_file():
                parser.error(f'build the helper first: {binary}')
            binaries[backend] = binary
    args.out.mkdir(parents=True, exist_ok=True)
    x_train, x_test, y_train, y_test = generate_data(
        n_train=args.train, n_test=args.test, n_features=args.features,
        contamination=0.1, random_state=args.seed)
    paths = {name: args.out / f'{name}.csv' for name in ('train', 'test')}
    np.savetxt(paths['train'], x_train, delimiter=',', fmt='%.17g')
    np.savetxt(paths['test'], x_test, delimiter=',', fmt='%.17g')
    np.savetxt(args.out / 'test_labels.csv', y_test, fmt='%d')
    metadata = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'platform': platform.platform(), 'machine': platform.machine(),
        'python': sys.version, 'command': sys.argv,
        'packages': {n: importlib.metadata.version(n) for n in ('pyod', 'numpy', 'scipy', 'scikit-learn', 'numba', 'threadpoolctl')},
        'configuration': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'data_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
        'native': {},
        'numpy_configuration': np.show_config(mode='dicts'),
        'timed_region': 'fit + decision_function; excludes construction, imports, CSV loading, serialization and metrics; C output allocation included',
        'seed_note': 'seed sets shared data; IForest model seed is 42 in all backends, but random streams differ; repeats do not measure seed variability',
    }
    for backend, binary in binaries.items():
        root = args.c_root if backend == 'pyod-c' else args.rs_root
        def git(*words):
            return subprocess.check_output(['git', '-C', str(root), *words], text=True).strip()
        metadata['native'][backend] = {
            'commit': git('rev-parse', 'HEAD'), 'dirty': bool(git('status', '--porcelain')),
            'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
        }
    raw, summary = [], []
    with (threadpool_limits(limits=args.threads) if args.threads else nullcontext()):
        metadata['threadpools'] = threadpool_info()
        for model in args.models:
            reference = None
            records = []
            for iteration in range(args.warmup + args.repeats):
                detector = constructors[model]()
                start = time.perf_counter()
                detector.fit(x_train)
                fit = time.perf_counter() - start
                start = time.perf_counter()
                scores = detector.decision_function(x_test)
                score = time.perf_counter() - start
                if iteration >= args.warmup:
                    records.append(dict(fit_seconds=fit, score_seconds=score, scores=scores.tolist()))
            reference = np.asarray(records[0]['scores'])
            all_records = {'PyOD': records}
            for backend, binary in binaries.items():
                completed = subprocess.run([str(binary), model, str(paths['train'].resolve()),
                                            str(paths['test'].resolve()), str(args.repeats), str(args.warmup)],
                                           capture_output=True, text=True, check=True)
                all_records[backend] = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
            for backend, records in all_records.items():
                if len(records) != args.repeats:
                    raise RuntimeError(f'{backend}: incorrect repetition count')
                enriched = []
                for iteration, record in enumerate(records):
                    scores = np.asarray(record['scores'])
                    if scores.shape != y_test.shape or not np.isfinite(scores).all():
                        raise RuntimeError(f'{backend} {model}: invalid scores')
                    row = dict(backend=backend, model=model, repeat=iteration, **record)
                    row.update(total_seconds=row['fit_seconds'] + row['score_seconds'],
                               roc_auc=float(roc_auc_score(y_test, scores)),
                               average_precision=float(average_precision_score(y_test, scores)),
                               max_abs_score_difference=float(np.max(np.abs(scores - reference))),
                               scores_match_pyod=(bool(np.allclose(scores, reference, rtol=1e-7, atol=1e-8))
                                                  if model != 'IForest' else None))
                    raw.append(row)
                    enriched.append(row)
                item = dict(backend=backend, model=model,
                            median_fit_seconds=statistics.median(r['fit_seconds'] for r in enriched),
                            median_score_seconds=statistics.median(r['score_seconds'] for r in enriched),
                            median_total_seconds=statistics.median(r['total_seconds'] for r in enriched),
                            min_total_seconds=min(r['total_seconds'] for r in enriched),
                            max_total_seconds=max(r['total_seconds'] for r in enriched),
                            median_roc_auc=statistics.median(r['roc_auc'] for r in enriched),
                            median_average_precision=statistics.median(r['average_precision'] for r in enriched),
                            max_abs_score_difference=max(r['max_abs_score_difference'] for r in enriched),
                            scores_match_pyod=(all(r['scores_match_pyod'] for r in enriched) if model != 'IForest' else None))
                summary.append(item)
            pyod_time = next(r['median_total_seconds'] for r in summary if r['backend'] == 'PyOD' and r['model'] == model)
            for row in summary:
                if row['model'] == model:
                    row['speedup_vs_pyod'] = pyod_time / row['median_total_seconds']
                    print(f"{model:8} {row['backend']:8} {row['median_total_seconds']*1000:9.3f} ms  {row['speedup_vs_pyod']:6.2f}x  AUC {row['median_roc_auc']:.4f}", flush=True)
    (args.out / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    with (args.out / 'raw.jsonl').open('w') as stream:
        for row in raw:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    with (args.out / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == '__main__':
    main()
