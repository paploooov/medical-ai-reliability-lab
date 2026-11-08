# Focused literature map

Sources accessed 8 November 2026. This is a targeted reading of six foundational sources, not a systematic search or a claim to cover the latest literature. Publisher/arXiv abstracts were checked for all six; dataset construction and medical generalization passages were additionally read in the available full-text HTML. Full methodological appraisal of every paper remains outstanding.

| Source | Relevant finding or method | Implication here |
|---|---|---|
| [Hendrycks & Gimpel, ICLR 2017](https://arxiv.org/abs/1610.02136), *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks* | Maximum softmax probability is a baseline for detecting errors and out-of-distribution examples. | Our confidence warning is an established baseline, not a new self-awareness mechanism. Misclassification and out-of-distribution detection are different targets. |
| [Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html), *On Calibration of Modern Neural Networks* | Temperature scaling can improve probability calibration on many studied datasets. | It is a probability correction, not a guarantee under image degradation; binary ranking is invariant to positive temperature. |
| [Ovadia et al., NeurIPS 2019](https://arxiv.org/abs/1906.02530), *Can You Trust Your Model's Uncertainty?* | Benchmarks uncertainty under dataset shift; conventional post-hoc calibration can fall short. | Our observed calibration limitation is consistent with a known research concern, not proof of novelty. |
| [Geifman & El-Yaniv, ICML 2019](https://proceedings.mlr.press/v97/geifman19a.html), *SelectiveNet* | Jointly learns prediction and a reject option; evaluates risk–coverage trade-offs. | Our fixed confidence cutoff is a simpler baseline; we have not implemented or compared SelectiveNet. |
| [Zech et al., PLOS Medicine 2018](https://doi.org/10.1371/journal.pmed.1002683), *Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs* | Across three hospital systems, location/prevalence effects and external performance changes complicate evaluation. | Synthetic blur is not evidence about a hospital shift; independent medical data and label compatibility matter. |
| [Yang et al., Scientific Data 2023](https://doi.org/10.1038/s41597-022-01721-8), *MedMNIST v2* | Lightweight standardized biomedical benchmarks. PneumoniaMNIST uses pediatric radiographs, center cropping and resizing; source training is split 9:1. | Appropriate for method development, not a clinical validation dataset. The general patient-splitting description is not a substitute for independently verifying this dataset's identities. |

## Higher resolution

The [official MedMNIST+ construction note](https://github.com/MedMNIST/MedMNIST/blob/main/on_medmnist_plus.md) states that cases, indices and splits correspond across resolutions. Pneumonia images are reprocessed from the center-cropped source at 64, 128 or 224 pixels. The smallest cropped source is 127×127 and is upsampled for larger targets. Higher resolution may retain more detail but adds neither patients nor independent validation. It is an appropriate within-case resolution comparison.

## Defensible contribution at this stage

A reproducible educational replication and sensitivity analysis of confidence-based error detection under controlled image transformations. Novel medical conclusions are not established. A useful narrower question is whether a validation-selected rejection rule maintains its retained error rate and coverage across resolution and corruption severity, after explicit duplicate auditing.
