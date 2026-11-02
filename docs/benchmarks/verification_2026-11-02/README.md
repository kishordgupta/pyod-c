# Supplementary local verification on November 2 2026

This is a separate eight-model verification of the new documented workflow. It does not replace the October 31 M3 Max benchmark reported in the repository README.

Configuration: 2,000 training rows, 1,000 test rows, 10 features, 10% contamination, PyOD `generate_data(random_state=42)`, one warm-up, three measured fresh-detector repetitions. `--threads 1` requested one thread through backend environment variables and threadpoolctl. Native detectors ran serially. See `manifest.json` for versions, backend, data hashes and build provenance.

The seven deterministic methods ABOD, COPOD, ECOD, HBOS, KNN, LOF and PCA matched PyOD test scores at `rtol=1e-7, atol=1e-8` in both ports. IForest had AUC 1.0 in all three implementations on this fixture, but its independent random streams and score convention preclude an elementwise parity claim.

LOF had AUC 0.206311 in all three implementations. Compatibility therefore does not imply useful anomaly ranking on every dataset. This fixture is a validation example, not a real-world accuracy benchmark.

`summary.csv` contains medians/ranges and metrics. `raw_times.csv` preserves all 72 measured backend/model/repetition records without redundant test-score vectors. The shared inputs can be regenerated with `scripts/benchmark_pyod.py`; verify the saved SHA256 hashes. The source commit IDs refer to the base algorithm snapshots; `dirty=true` records that the new helper/examples/documentation were present during the run. Binary hashes identify the exact benchmark executables. No neural timing or neural PyOD parity was rerun here.
