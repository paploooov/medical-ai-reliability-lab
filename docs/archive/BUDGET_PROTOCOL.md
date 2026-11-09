# Study 06 — noise collapse and fixed training-budget extension

Date: 9 November 2026. This is a follow-up prompted by failures observed in studies 04–05, not an independent test or a prospective clinical study.

## Diagnostic question

For the six saved 15-epoch-budget models, verify that raw-to-[0,1] scaling, Gaussian noise standard deviation 0.1, clipping and float32 inference reproduce the stored predictions. Compare clean/noisy activations at each ReLU, the final hidden representation and output logits. In particular record cases whose entire final hidden vector is zero: their output reduces algebraically to the final layer bias. This is a computational mechanism, not a medical explanation or proof that every noise-induced error has the same cause.

## Training question

Repeat the same six initializations, architecture, data groups, Adam settings and minibatch order, increasing the budget from 15 to 45 epochs. Select the lowest model-selection BCE over all 45 epochs, never using test performance. Recompute the 80%-target confidence cutoff on the same separate clean calibration set. Preserve earlier outputs in place; save the new study to reports/budget and models/budget. Verify the first 15 epochs agree with the prior run, and record any discrepancy before making a matched-budget claim. No noise augmentation, new architecture or learning-rate changes.

## Outcomes

Primary: change in the mean noise-minus-clean retained error risk, 45-budget minus 15-budget, for the three 28px seeds. Lower is better for that specific endpoint, but coverage must be reported. Secondary: 64px results, other conditions, full-case accuracy, class-specific performance and collapse counts. Do not select a successful seed or call more epochs a proven fix.

Use 2,000 paired exact-test-group bootstrap replicates, seed 20261109. Recalculate ratio endpoints, holding the trained models and calibration cutoffs fixed; discard/count replicates with an empty accepted subset. The interval excludes training-population, patient clustering and calibration sampling uncertainty. Compare the same 621 representatives, conditions and seeds jointly. Any later tuning on these observations remains exploratory.

## Evidence

Store raw per-epoch losses and timing, selected checkpoints, thresholds, model hashes, complete per-case probabilities, activation diagnostics, diagnostic figures and a report of all conditions. Weight checks and independently recomputed metric rows are required. This study does not establish external clinical validity or architectural novelty.

## Additional diagnostic sweep (specified during the budget run)

For the frozen 15-budget models only, examine Gaussian noise SD 0, 0.025, 0.05, 0.1, 0.15 and 0.2 using noise seeds 7, 42 and 99. Scale the same sampled standard-normal field across severities within each resolution/noise seed. Record full-case accuracy and the fraction with an all-zero final hidden vector. This is an exploratory mechanism check prompted by the activation findings; it does not modify training, checkpoint selection, the primary endpoint or the operational cutoff. Plot separate training seeds and show the range across three noise realizations, not statistical confidence bands.
