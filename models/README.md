# Model artifacts

Numeric weights trained on the official PneumoniaMNIST training split. `mlp.npz`: 784 inputs → 64 ReLU units → one sigmoid output; `logistic.npz`: linear logit. Inputs: grayscale 28×28, flattened row-major, divided by 255. Output: probability of dataset class 1 (pneumonia). Threshold 0.5; no probability calibration. See `reports/metrics/validation.json` for selection and `run.json` for provenance.

Research use only. The MLP has low specificity (59.8%) on the clean test split and predicts the positive class for every image under the strongest contrast reduction. The weights are not a clinical diagnostic system. A future Core ML export must preserve preprocessing and pass numerical equivalence checks. No Apple export is included yet.
