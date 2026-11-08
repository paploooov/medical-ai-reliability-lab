# Project status — 8 November 2026

Completed: dataset validation, linear and 64-unit neural training, validation selection, ten-condition test evaluation, paired bootstrap intervals, figures, portable numeric weights, and metric checks. Full findings: [pilot report](research/PILOT.md).

Next milestone: related-work review and higher-resolution data assessment, followed by declared multi-seed and calibration experiments. The pilot test set has been seen; do not represent further tuning as independent confirmation. No external validation or Apple app has been completed. Infrastructure expenditure: none.

Reproduction commands are in the project README. Model provenance is recorded in reports/metrics/run.json. Colab UI execution has not been tested; the underlying data workflow and CPU experiment were run locally in the cloud environment. Telegram export is not configured.

Follow-up: confidence-based error detection and validation-selected rejection evaluated for both models across all ten conditions. Six metric tests pass. Model classification and licensing status documented. Repeated training, detection uncertainty intervals, calibration and external evaluation remain outstanding.

CNN follow-up completed: three seeds × 25 epochs, validation-selected checkpoints, temperature scaling, ten conditions each, 60 result rows. Eight unit tests passed; all three stored CNN checkpoints reproduce clean predictions. Learning curves, calibration plots, per-image predictions and full study are saved. No independent external validation yet.
