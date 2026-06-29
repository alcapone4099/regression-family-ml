# The Regression Family: Custom ML Optimization Engines

A mathematical tour de force implementing foundational machine learning architectures completely from scratch using raw NumPy. This repository bypasses high-level abstractions (like Scikit-Learn or PyTorch) to derive, build, and evaluate core optimization frameworks applied across real-world agricultural, phytopathologic, and taxonomic datasets.

## 🪐 Core Implementations & Mathematical Architectures

The framework is divided into three distinct pillars, each conquering a unique optimization paradigm:

### 1. Agronomic Yield Predictor (Continuous Linear Regression)
* **Objective:** Predict localized Rice yields ($kg/ha$) based on micro-climate metrics and soil chemistry.
* **Architectures Implemented:** Dual-Engine architecture comparing an algebraic **Closed-Form Ordinary Least Squares (OLS) Matrix Inversion** ($w = (X^T X)^{-1} X^T y$) against a step-by-step **Batch Gradient Descent (BGD)** solver.
* **Mathematical Insight:** Handles extreme natural variance via a **Natural Log Transformation** (y -> ln(y)), neutralizing severe biological outlier distortion (e.g., a single 120,000 kg/ha sample pulling the hyperplane out of bounds) and achieving an extraordinary optimization convergence precision delta of **$7.99 \times 10^{-15}$** (at the limits of 64-bit floating-point precision).

### 2. Phytopathology Classifier (Binary Logistic Regression)
* **Objective:** Classify automated greenhouse leaves as "Healthy" vs "Diseased" based on pixel color intensities and surface texture contrast.
* **Architectures Implemented:** Maximum Likelihood Estimation optimized via Vectorized Gradient Descent ($X^T (\sigma(Xw) - y)$) mapped through a non-linear **Sigmoid Link Function**.
* **Engineering Protections:** Hardcoded **numerical clipping boundaries** to prevent runtime exponential overflow (`NaN`) along with machine epsilon padding ($10^{-15}$) to prevent infinity crashes during logarithmic Binary Cross-Entropy (BCE) evaluation.
* **Performance:** Achieved **99.50% Classification Accuracy** on unseen validation samples, capturing a perfect **100.00% Precision** (Zero False Alarms) and a vital **98.99% Recall** safety net.

### 3. Taxonomic Evolutionary Engine (Multi-Class SVM)
* **Objective:** Classify botanical specimens into three distinct species (*Setosa*, *Versicolor*, *Virginica*) based on physical morphological traits.
* **Architectures Implemented:** Primal **Soft-Margin Support Vector Machine** utilizing **Conditional Subgradient Descent** to bypass non-smooth non-differentiable Hinge Loss boundaries.
* **Strategy:** Scaled from natively binary boundaries into multi-class dimensions via a **One-vs-Rest (OvR)** architecture, selecting assignments via maximum geometric margin distance confidence scores ($\arg\max w_k^T x + b_k$).
* **Performance:** Achieved **93.33% Test Accuracy** with a robust **94.87% Macro-Averaged Precision**, ensuring rare classes are isolated without label-imbalance skew.

---

## 📂 Repository Structure

```text
regression_family/
│
├── implementation/             # Raw mathematical class engines
│   ├── logistic_regression.py  # Stable BCE & Sigmoid Gradient loops
│   └── svm_classifier.py       # Binary SVM & Multi-Class OvR wrappers
│
├── run/                        # Execution and evaluation pipelines
│   ├── run_yield.py            # Closed-form vs BGD comparison
│   ├── run_plant.py            # Diagnostic classifier engine
│   └── run_iris.py             # Taxonomic SVM classifier
│
├── results/                    # Automated portfolio exports
│   ├── crop_yield/             # OLS vs GD weight logs & loss plots
│   ├── plant_disease/          # Phytopathology confusion matrices
│   └── iris/                   # Macro-averaged taxonomic reports
│
└── .gitignore                  # Keeps repository clean of bytecaches/npy files
