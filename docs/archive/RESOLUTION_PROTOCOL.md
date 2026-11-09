# Resolution study: locked execution plan

This exploratory follow-up compares 28×28 with 64×64 on the same public pediatric radiographs. The benchmark test images were examined previously. No independent clinical validation or preregistration is claimed.

## Data policy

Build connected groups of pixel-identical arrays at either resolution, retaining one representative per group. Groups with conflicting labels or train/test overlap stop execution. Where training and validation overlap, retain the validation representative and remove all training members. Within test, keep one representative per exact group. Divide the remaining official validation representatives 50/50, stratified by label with seed 2026, into model-selection and calibration sets. This alters the official benchmark: results must be labelled a deduplicated sensitivity analysis, not directly substituted for official scores. Pixel groups are not patients. Raw archives remain unchanged. Commit the complete manifest before training.

## Training and evaluation

Same network and parameter count at both resolutions: two convolution/ReLU/max-pool blocks (16,32 channels), adaptive 4×4 pooling, dense 32-unit head and one logit. Fixed seeds 17/42/83, Adam 0.001, 15 epochs, batch 128. Select each checkpoint by model-selection BCE. This modest fixed compute budget is not a tuned best-model claim; equal epochs do not mean equal compute. Log durations and every epoch. Primary evaluation uses raw probabilities; temperature scaling, if later added, is a separate analysis.

Choose each model's confidence cutoff using only the calibration set, targeting 80% coverage (20th percentile). Evaluate clean, primary blur sigma=width/28, secondary fixed blur sigma=1 pixel, noise SD=0.1 and contrast factor=0.5 around intensity 0.5. Gaussian blur uses reflected boundaries. Transform each image independently; clip intensities where appropriate. The resolution-relative blur matches relative image width, not physical millimeters or a validated acquisition process. Primary and fixed blur coincide at 28 pixels; do not count them as independent evidence.

## Primary estimand and uncertainty

For each seed calculate retained-error risk under relative blur minus clean risk. Primary contrast: the mean of those changes at 64 minus the mean at 28. Retained sets may differ: this is an operational comparison of separately calibrated rules, not a comparison on an identical retained cohort. Always report coverage and its change alongside risk. A lower retained risk achieved by rejecting more images is not an unqualified benefit.

Paired bootstrap of the retained unique test representatives: 2,000 draws, seed 2026, same sampled indices across resolutions, conditions and seeds. Recalculate ratio endpoints inside each draw; discard and count draws with empty retained subsets. Report percentile 95% intervals conditional on the fitted models. With one representative per exact group this is equal-group resampling, not patient-level inference. Report seed-specific results separately; three seeds are not a population uncertainty estimate. No claim of corrected significance across secondary outcomes.

## Evidence requirements

Immutable input manifest and protocol commit; checksummed weights; model-selection history; per-case probabilities; threshold metadata; complete environment; recorded timings; accuracy, Brier, sensitivity/specificity, error AUROC/AP, coverage and retained risk. Check saved-model outputs and recalculated metrics, verify no pixel group crosses fitting/selection/calibration/test, and report failures. Preserve older results.
