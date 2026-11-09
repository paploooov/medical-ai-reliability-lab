# Project status — 8 November 2026

Completed: dataset validation, linear and 64-unit neural training, validation selection, ten-condition test evaluation, paired bootstrap intervals, figures, portable numeric weights, and metric checks. Full findings: [pilot report](research/PILOT.md).

Next milestone: related-work review and higher-resolution data assessment, followed by declared multi-seed and calibration experiments. The pilot test set has been seen; do not represent further tuning as independent confirmation. No external validation or Apple app has been completed. Infrastructure expenditure: none.

Reproduction commands are in the project README. Model provenance is recorded in reports/metrics/run.json. Colab UI execution has not been tested; the underlying data workflow and CPU experiment were run locally in the cloud environment. Telegram export is not configured.

Follow-up: confidence-based error detection and validation-selected rejection evaluated for both models across all ten conditions. Six metric tests pass. Model classification and licensing status documented. Repeated training, detection uncertainty intervals, calibration and external evaluation remain outstanding.

CNN follow-up completed: three seeds × 25 epochs, validation-selected checkpoints, temperature scaling, ten conditions each, 60 result rows. Eight unit tests passed; all three stored CNN checkpoints reproduce clean predictions. Learning curves, calibration plots, per-image predictions and full study are saved. No independent external validation yet.

Audit completed: 80 metric rows recomputed successfully; eight exact 28×28 train/validation shared image hashes and three excess test repeats identified. Further claims must include these limitations. Six-source foundational literature map and revised resolution/rejection study plan added. No further models trained during the audit.

MedMNIST+ 64×64 archive downloaded, checksum verified, all split labels matched to 28×28. Exact overlap counts persist at 64×64. A same-case resolution figure is in reports/audit. Higher-resolution training is pending revised duplicate/split handling.

Resolution sensitivity analysis completed: 4682 fit / 262 selection / 262 calibration / 621 unique test cases; six runs × 15 epochs; fixed 21249-parameter adaptive CNN; paired bootstrap primary endpoint and PDF figures saved. 30 metric rows, six weights and six cutoffs independently verified. All 12 tests pass. Results are conditional on fixed fitted models and a known benchmark, not patient-level external validation. See RESOLUTION_STUDY.md for interpretation and convergence limitations.

Study 05 completed: equal-coverage sensitivity analysis, 180 rows, paired bootstrap, class breakdown, deterministic error atlas, and full-condition plots. No new training. 18 tests pass. Three external sources assessed from public documentation; no restricted data accessed. README now leads with study 05.

9 November: study 06 finished, six × 45-epoch runs; all 90 prefix entries match study 04. Noise preprocessing reproduced; zero final-hidden vectors identified and bias-only acceptance counted. All six new weights and 30 result rows verified. 108-condition frozen-model noise sweep completed. Twenty tests pass. Longer training did not solve noise failure.
