# Regression Family from Scratch: Agronomy, Pathology & Taxonomy Pipelines

A pure-NumPy engineering framework that implements foundational linear and geometric machine learning models from architectural first principles. By eliminating automated libraries like `scikit-learn`, this repository documents exactly how optimization algorithms traverse error landscapes, navigate non-smooth loss boundaries, and map mathematical frameworks directly to physical biological systems.

---

## 1. Project Objective & Core Concept

Modern data science makes it easy to execute `model.fit()` using automated libraries. However, biological data is natively noisy, multi-collinear, and highly variable. 

The core objective of this project is to build and evaluate the three fundamental archetypes of the regression family using absolute mathematical code (pure matrix algebra and vector calculations). By doing so, we verify optimization mechanics—proving that manual gradient descents land precisely on exact mathematical global minimums across different statistical problems.

---

## 2. Dataset & System Architecture

The nature of the biological question dictates the mathematical space of the model. This framework processes three distinct plant-related datasets across three independent paradigms:

* **Linear Regression (Crop Yield):** Continuous Value Space ($y \in \mathbb{R}$)
* **Logistic Regression (Plant Pathology):** Binary State Probability Space ($y \in [0, 1]$)
* **Support Vector Machine (Plant Taxonomy):** Multi-Class Vector Space ($y \in \{0, 1, 2\}$)

### I. Crop Yield Dataset (Agronomic Predictor)
* **Biological Context:** Mapping phenotypic plasticity—how environmental variables scale a crop's physical mass output.
* **Attributes processed:** Soil Chemistry inputs (Nitrogen, Phosphorus, Potassium), Environmental features (Rainfall, Temperature), and Crop Categorical identifiers.
* **Target Output:** Continuous numerical scale (Crop Yield in kg/ha, optimized using a natural logarithm conversion).

### II. PlantVillage Leaf Profiles (Phytopathology Classifier)
* **Biological Context:** Identifying active cellular infections (necrosis and chlorosis) using visual biometric traits.
* **Attributes processed:** `mean_green_intensity` (a direct proxy for healthy chlorophyll volume) and `texture_contrast` (a metric mapping jagged, localized cell death/spots).
* **Target Output:** Binary classification labels (`0.0` for Healthy, `1.0` for Diseased).

### III. UCI Iris Dataset (Evolutionary Taxonomist)
* **Biological Context:** Finding the precise geometric boundaries of natural evolutionary variance and structural divergence between sister species.
* **Attributes processed:** Four continuous physical measurements: Petal Length, Petal Width, Sepal Length, and Sepal Width.
* **Target Output:** Multi-class categorization markers (`0` for Setosa, `1` for Versicolor, `2` for Virginica).

---

## 3. Algorithmic Approaches & Implementation Detail

Each engine is built cleanly using vector reductions, matrix transformations, and raw loops without third-party mathematical short-cuts.

### Approach A: Ridge Linear Regression
* **Mathematical Space:** Infinite Euclidean Space ($\mathbb{R}$)
* **The Mechanics:** Assumes target outcomes scale linearly with structural inputs plus random Gaussian noise. It implements both a **Closed-Form Normal Equation Inversion** ($w^* = (X^T X + n\lambda I)^{-1} X^T y$) and a step-by-step **Batch Gradient Descent Loop**.
* **Regularization Safeguard:** Implements an $L_2$ Ridge penalty ($\lambda$) to force numerical stability down the diagonal matrix, ensuring it remains cleanly invertible even when soil data contains highly correlated parameters.

### Approach B: Binary Logistic Regression
* **Mathematical Space:** Probability Bound Space ($[0, 1]$)
* **The Mechanics:** Since physical conditions cannot be a fraction of a category, inputs are passed through a linear layer and squashed into a probability vector using the non-linear **Sigmoid Link Function** ($\sigma(z) = \frac{1}{1 + e^{-z}}$). Optimization updates are guided by a **Maximum Likelihood Estimation** derivative.
* **Numerical Safeguards:** Integrates an exponential value clipping window to completely block 64-bit floating-point overflow spikes, alongside a machine epsilon boundary ($10^{-15}$) to protect internal log operations against infinity calculations.

### Approach C: One-vs-Rest Support Vector Machine (SVM)
* **Mathematical Space:** High-Dimensional Separation Matrix ($\mathbb{R}^d$)
* **The Mechanics:** Instead of balancing overall probability averages, this structural model focuses exclusively on the hardest borderline plants to classify (the Support Vectors). It maximizes the physical gap ("clear air margin") between species boundaries using a non-smooth **Primal Hinge Loss Function** ($\max(0, 1 - y_i(w^T x_i + b))$).
* **Optimization Framework:** Because the hinge loss contains a sharp, non-differentiable kink at the margin valley, traditional derivatives fail. The system uses manual **Primal Subgradient Descent** paired with a **One-vs-Rest (OvR)** wrapper matrix to resolve three distinct plant lineages.

---

## 4. Evaluation Metrics & Verified Empirical Results

Below are the exact metrics used to audit each engine, alongside the real-world performance benchmarks generated by the pipeline.

### I. Crop Yield Predictions
* **Metrics Used:**
    * **$R^2$ (Coefficient of Determination):** Measures the proportion of variance explained by the features relative to a simple horizontal dataset average line.
    * **RMSE (Root Mean Squared Error):** Tracks the standard geometric distance from prediction to true coordinates.
* **Verified Results:**
    * **Test $R^2$ Score:** `0.0920` (Explains ~9.2% of complex environmental variance on unseen test data).
    * **Test RMSE:** `0.8779` (Log-scale error tracking).
    * **Convergence Identity:** The maximum delta between the Closed-Form matrix calculation and the 2,000-epoch Batch Gradient Descent weights was evaluated at **$7.9936 \times 10^{-15}$**, demonstrating flawless mathematical alignment at the absolute precision limits of modern computing.

### II. Plant Village Pathology Diagnostic
* **Metrics Used:**
    * **Accuracy:** Percentage of overall perfect classifications.
    * **Precision:** The reliability of raised alerts (minimizing False Positives).
    * **Recall:** The safety net catch-rate (minimizing missed infections / False Negatives).
    * **Balanced F1-Score:** The harmonic mean balancing precision and recall.
* **Verified Results:**
    * **Test Classification Accuracy:** `99.50%` (199 out of 200 validation leaves correctly identified).
    * **Test Precision:** `100.00%` (Zero false alarms; every flagged leaf was genuinely infected).
    * **Test Recall:** `98.99%` (Only one single active infection managed to slip by).
    * **Final F1-Score:** `0.9949`

### III. Taxonomic Iris Speciation
* **Metrics Used:** Macro-Averaged Precision, Recall, and Multi-Class Accuracy (giving rare and intersecting species equal weighting to eliminate baseline group distortions).
* **Verified Results:**
    * **Test Multi-Class Accuracy:** `93.33%` (Successfully separated intersecting morphologic structures across 28 out of 30 test records).
    * **Macro-Precision:** `94.87%`
    * **Macro-Recall:** `92.46%`
    * **Macro-Averaged F1-Score:** `0.9365`

---

## 5. Summary Portfolio Matrix

| Model Engine | Targeted Bio-Problem | Target Variable Type | Primary Math Tool | Core Strategic Value |
| :--- | :--- | :--- | :--- | :--- |
| **Ridge Regression** | Crop Yield Scaling | Continuous Value | Normal Inversion & GD | Captures broad continuous phenotypic shifts while managing high feature correlations. |
| **Logistic Regression** | Pathology Diagnostics | Binary Category | Sigmoid Link & MLE | Establishes sharp diagnostic thresholds over uniform lighting conditions. |
| **Support Vector Machine** | Species Identification | Multi-Class Index | Primal Hinge Subgradients | Resolves complex evolutionary boundary overlaps by emphasizing maximum margin borders. |
