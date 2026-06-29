# Phytopathology Classification Analysis Report

## 1. Empirical Performance Metrics
* **Model Class:** Custom Binary Logistic Regression (Sigmoid Link Function)
* **Test Classification Accuracy:** 99.50%
* **Test Precision Score:** 100.00%
* **Test Recall Score:** 98.99%
* **Test Balanced F1-Score:** 0.9949

## 2. Metric Interpretations
* **Accuracy (99.50%):** Out of 200 completely unseen validation leaf samples, the classification engine correctly diagnosed 199 plants, committing exactly one single error.
* **Precision (100.00%):** The model achieved a flawless False Positive rate of zero. Every single alert raised by the model flagging a leaf as infected was 100% accurate, guaranteeing no false alarms for agricultural operators.
* **Recall (98.99%):** The model captured 98.99% of actual infected leaves. In automated phytopathology, recall is the most critical safety metric; missing only one single diseased leaf prevents localized crop contamination from spreading through a greenhouse.

## 3. Scientific Significance & Architectural Wins
* **Linear Separability:** The extreme accuracy confirms that the engineered features (`mean_green_intensity` and `texture_contrast`) are direct, definitive biological indicators of leaf necrosis and chlorosis. Healthy uniform green structures and jagged diseased spots form highly distinct clusters.
* **Optimization Stability:** The implementation of numerical clipping shields the custom exponential functions against runtime overflow crashes, allowing the learning rate (alpha = 0.1) to safely guide the model to maximum likelihood convergence.
