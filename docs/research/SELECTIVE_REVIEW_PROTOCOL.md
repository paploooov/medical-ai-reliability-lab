# Study 05 — equal-coverage sensitivity analysis

The preceding study showed less deterioration at 64 pixels, but different fractions of predictions were accepted. This follow-up asks whether that observation persists when exactly the same fraction is accepted. It reuses the six frozen models and predictions from study 04; no retraining and no independent confirmation are claimed.

Primary coverage: 80%. Secondary descriptive curves: 50%, 60%, 70%, 90%, 100%. Within each model/condition, accept the largest confidence values. At a tied boundary assign fractional weights equally to the whole tie group, equivalent to expected random selection among ties. Label values never resolve ties. The denominator is exactly coverage × sample count; this can be fractional. Cases accepted at each resolution need not be the same cases.

Primary contrast: mean across seeds of (blur risk − clean risk) at 64 pixels minus the corresponding mean at 28. Resample the same unique test representatives jointly across conditions, resolutions and seeds, 2,000 times, seed 20261108; recalculate ranking and fractional tie weights within each bootstrap sample. Report percentile intervals conditional on the six models. These do not include retraining, calibration uncertainty or patient clustering. No clinical risk guarantee or corrected multi-endpoint significance is claimed.

Fixed coverage requires the test confidence distribution. It is a retrospective ranking diagnostic, not an operational rule fixed on clean calibration data. Keep both views side by side. Report seed-level results and class-specific accepted/rejected counts. Show deterministic failure examples chosen by a stated rule, never as evidence that highlighted pixels explain a model decision.

The strongest noise failures will be described with saved outputs and variation across seeds; do not silently omit them because they weaken the model's appearance. No claim that Gaussian pixel noise models scanner physics is made.
