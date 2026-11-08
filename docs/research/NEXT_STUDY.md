# Next study: resolution and selective prediction

**Question:** On the same public pediatric radiographs, how do resolution and controlled degradation affect the error rate among predictions retained by a validation-selected rejection rule?

This is a proposed exploratory methods study, not a frozen preregistration. Do not describe it as clinical validation or an unseen-test experiment.

1. Audit corresponding 28/64 images and labels; record exact duplicate groups and investigate whether low-resolution matches persist. Retain immutable official-split baseline reports. A separately labelled de-duplicated sensitivity analysis must specify which groups are excluded from fitting/calibration and how repeated test images are weighted. Pixel hashes do not establish patient identity.
2. Use a small convolutional network with adaptive pooling so resolution does not silently multiply dense parameters. Keep architecture, seeds and selection rules fixed; report parameter count, compute and wall time separately. Start with 28 versus 64 on CPU; only expand to 128/224 if the analysis warrants the cost.
3. Separate model-selection and calibration data with group-aware assignments based on available duplicate identities. Document limited calibration sample size and remaining unknown patient overlap. Freeze these assignments and the experiment configuration before fitting.
4. Primary outcome: change from clean to prespecified blur in retained error rate using a clean calibration cutoff targeting 80% coverage. Also report actual coverage, error recall, review precision, Brier score, error AUROC/AP and class-specific sensitivity/specificity. A rule that rejects almost everything is not automatically useful.
5. Define blur relative to image width and show visual checks. Include a secondary fixed-pixel-sigma comparison, explicitly answering a different question. Noise and contrast remain secondary exploratory perturbations. No physical scanner equivalence is assumed.
6. Use paired resampling of available duplicate groups, across conditions, for uncertainty conditional on each fitted model. Report seed variation separately; do not treat predictions across seeds as new patients. Specify the primary comparison and label remaining comparisons exploratory.
7. Preserve untouched raw data, split manifests, configs, code commit, model checksums, environment, per-image outputs and a report of negative results. A separate patient-aware source is still needed for external validity; higher-resolution MedMNIST is not such a source.

Before execution: finalize duplicate handling, calibration split size, resolution-compatible architecture and a single primary blur strength. These are methodological decisions to record, not questions requiring the user to design the study.
