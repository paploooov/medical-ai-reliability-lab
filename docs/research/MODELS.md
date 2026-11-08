# Which models are actually in this repository?

| Model | Implementation | Training | Role |
|---|---|---|---|
| Logistic regression | scikit-learn, 784 pixel inputs, 785 learned parameters | Fit from scratch; C selected on validation | Simple reference |
| Neural MLP | scikit-learn, 784 → 64 ReLU → 1 sigmoid, 50,305 learned parameters | Fit from scratch for 40 epochs; validation selects checkpoint | First neural baseline |

Both are real fitted models with saved numeric parameters. Neither is a pretrained medical foundation model, a convolutional network, a new architecture or a clinically validated product. Training an established architecture produces project-specific weights; it does not establish architectural novelty.

The small 28×28 inputs and about 4,700 training examples explain why this CPU experiment takes little time. Longer runtime alone would not improve the scientific evidence. The important missing work is comparison, independent validation and analysis of failure modes.

## How to compare fairly

Start with performance on identical splits and preprocessing. Include sensitivity, specificity, discrimination, probability calibration and error detection; do not collapse these into a single league-table rank. Report training variation and paired comparisons. Published scores are only comparable when data versions, resolution, splits and evaluation protocols match.

Next candidates include a small CNN trained from scratch and an established convolutional architecture with documented pretraining. A larger pretrained chest-X-ray model would require a compatible higher-resolution dataset, label definitions, licensing review and leakage checks. No published-model benchmark or comprehensive literature review has been completed here.

## Evidence

`models/*.npz`: weights. `reports/metrics/validation.json`: checkpoint search. `run.json`: software versions, configuration and pre-evaluation commit. `predictions.npz`: test outputs. The saved MLP weights were checked against all 624 clean test outputs to numerical tolerance 1e-12.
