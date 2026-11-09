# Research scope and decision log

## Current research question

Does a confidence-based rejection rule, fixed using separate clean calibration data, retain its coverage and error rate when pediatric chest radiographs are blurred? Does increasing the benchmark resolution from 28 to 64 pixels change that behavior?

This project studies model reliability under a defined synthetic perturbation. It does not diagnose patients, validate a clinical workflow or establish that a model knows whether its own diagnosis is correct.

## Why the design changed

1. Initial simple models established a runnable pipeline, but the first plots counted confident errors rather than measuring error detection directly.
2. Confidence-ranking and risk–coverage analyses were added. They showed that good ranking and useful rejection are different questions.
3. Three CNN training seeds exposed variability. Temperature scaling changed probabilities without changing binary decisions; improved error ranking could not be expected mathematically from that transform alone.
4. An audit found exact pixel overlap between official train and validation splits and repetition within test. Historical results were preserved and labelled accordingly.
5. The current experiment uses one representative per exact-pixel group, removes train/validation overlap from training, and separates checkpoint selection from calibration. Resolution-specific inputs come from the same source cases.

## Evidence ladder

Completed evidence is limited to data and software verification, exploratory benchmark measurements and conditional statistical estimates. Next levels are robust comparison across training budgets, patient-aware independent data, clinical review of transformations and intended use, and prospective evaluation. These cannot be inferred from a model's parameter count or the number of charts.

## Practical boundaries

CPU-only development; no paid services. Source data and dependencies are not copied into Git. Small model weights, manifests, probabilities and reports are versioned. No clinical contact, paid resource or private patient data is required for the current stage. A public repository is a research record, not a substitute for publication peer review.

## Before a Facharbeit

Explain the label definition and source population; distinguish classification, calibration and error detection; state what was chosen before evaluation; reproduce the primary table from saved probabilities; discuss rejected correct cases and accepted errors; show how duplicate handling changes the interpretation; state remaining patient-identity and external-validity limitations. Negative findings belong in the report.
