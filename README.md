# Imbalance-Aware Credit Card Fraud Detection

<p align="center">
  <strong>LEAKAGE-SAFE • IMBALANCE-AWARE • EXPLAINABLE</strong><br/>
  <sub>A reproducible machine-learning study of credit-card fraud detection under extreme class imbalance.</sub>
</p>

<p align="center">
  <em>From a 1:577 class imbalance to a defensible evaluation protocol.</em>
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-6A5ACD?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-Tree%20Boosting-0F9D58?style=for-the-badge)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-FF6B35?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-black?style=for-the-badge)

</p>

> **Research note** — This repository treats class imbalance as a methodological problem, not merely a preprocessing step. The core principle is simple: **the held-out test set must never influence the solution to the imbalance problem.**

## At a glance

| | |
|---|---|
| **Problem** | Credit-card fraud detection under extreme class imbalance |
| **Benchmark** | ULB Credit Card Fraud Detection |
| **Models** | Logistic Regression · Random Forest · XGBoost |
| **Imbalance strategy** | SMOTE, applied only to training data |
| **Explainability** | SHAP TreeExplainer |
| **Best held-out model** | **Random Forest** |
| **Best F1-score** | **0.8763** |
| **Best ROC-AUC** | **0.9791** |

## Navigation

- [Overview](#overview)
- [Why this repository exists](#why-this-repository-exists)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Class imbalance](#class-imbalance)
- [Models](#models)
- [Evaluation](#evaluation)
- [Results](#results)
- [Explainability](#explainability)
- [Figures](#figures)
- [Repository structure](#repository-structure)
- [Reproducing the experiment](#reproducing-the-experiment)
- [Research context & author](#research-context--author)
- [Technical article](#technical-article)
- [Citation](#citation)
- [License](#license)

---

## Overview

Credit-card fraud detection is not a conventional accuracy-maximisation problem.

In the ULB Credit Card Fraud Detection benchmark, only a tiny fraction of transactions are fraudulent. A classifier can therefore achieve deceptively high accuracy while failing to identify the minority class that actually matters.

This project studies that problem through a reproducible pipeline built around:

- leakage-safe train/test separation;
- train-only feature scaling;
- **SMOTE applied only after the split**;
- Logistic Regression as a balanced linear baseline;
- Random Forest as an ensemble benchmark;
- XGBoost with leakage-safe cross-validation and randomized hyperparameter search;
- ROC-AUC, precision, recall and F1-score;
- SHAP-based model interpretation;
- publication-ready figures and an accompanying technical article.

The strongest held-out model was **Random Forest**, achieving:

| Model | AUC-ROC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9619 | 0.7431 | 0.6812 | 0.8163 |
| XGBoost | 0.9703 | 0.8216 | 0.8444 | 0.8000 |
| **Random Forest** | **0.9791** | **0.8763** | **0.8912** | **0.8618** |

The evaluation set contains **56,962 transactions**, including 98 fraudulent transactions.

---

## Why this repository exists

The interesting part of fraud detection is not simply training a classifier.

It is answering:

1. **Can the minority class be learned without leaking information from the test set?**
2. **Does oversampling actually improve minority-class performance?**
3. **Does a nonlinear ensemble outperform a linear baseline?**
4. **Which transaction features drive the final predictions?**
5. **Are the reported metrics trustworthy enough to reproduce?**

The repository is therefore organised around the experiment rather than around a generic machine-learning template.

---

## Dataset

The experiments use the public **ULB Credit Card Fraud Detection** benchmark.

### Dataset profile

| Property | Value |
|---|---:|
| Total transactions | 284,807 |
| Fraudulent transactions | 492 |
| Legitimate transactions | 284,315 |
| Fraud prevalence | ~0.172% |
| Approximate imbalance | 1 : 577 |
| Input features | 30 |
| Test partition | 56,962 |

The dataset contains anonymised PCA-derived variables `V1`–`V28`, together with `Time`, `Amount`, and the binary `Class` target.

> **Data note:** The raw dataset is not redistributed in this repository. Obtain it from the original public source and place it locally according to the project configuration.

---

## Methodology

The experiment follows a strict ordering:

```text
Raw transactions
       │
       ▼
Duplicate removal
       │
       ▼
Stratified train/test split
       │
       ├──────────────► Held-out test set
       │                    │
       │                    └── untouched until final evaluation
       ▼
Train-only scaling
       │
       ▼
SMOTE on training data only
       │
       ├──────────────► Logistic Regression
       ├──────────────► Random Forest
       └──────────────► XGBoost + leakage-safe CV
                              │
                              ▼
                       Final evaluation
                              │
                              ▼
                    SHAP interpretation
```

### The critical rule

**SMOTE is never applied before the train/test split.**

Applying SMOTE to the full dataset first allows synthetic training examples to be influenced by information derived from observations that later appear in the test set. That makes the final evaluation optimistic.

---

## Class imbalance

The original fraud prevalence is approximately 0.172%.

The imbalance ratio can be expressed as:

$$
R = \frac{N_{\text{legitimate}}}{N_{\text{fraud}}}
$$

For this benchmark:

$$
R \approx \frac{284315}{492} \approx 577:1
$$

This means that a model that predicts the majority class almost everywhere can still appear highly accurate.

That is why this project prioritises **F1-score, precision, recall and ROC-AUC** rather than accuracy alone.

### SMOTE

For a minority sample \(x_i\), SMOTE creates a synthetic point by interpolating between \(x_i\) and one of its minority-class neighbours \(x_{nn}\):

$$
x_{\text{new}} = x_i + \lambda(x_{nn}-x_i),
\qquad \lambda \sim U(0,1)
$$

In this experiment, SMOTE is applied exclusively to the training partition.

The resulting training partition contains:

- 227,451 legitimate examples;
- 227,451 minority-class examples.

The held-out test set remains untouched.

---

## Models

### Logistic Regression

A balanced Logistic Regression model provides the linear reference point:

$$
P(y=1\mid x)=\sigma(w^Tx+b)
$$

where

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

`class_weight="balanced"` is used so that the minority class is not ignored during optimisation.

### Random Forest

Random Forest combines multiple decision trees and aggregates their predictions:

$$
\hat{y}=\operatorname{mode}\{T_1(x),T_2(x),\ldots,T_B(x)\}
$$

The evaluated configuration used 200 estimators, maximum depth 10, and balanced class weighting.

### XGBoost

XGBoost constructs an additive ensemble:

$$
\hat{y}_i = \sum_{k=1}^{K} f_k(x_i)
$$

with the regularised objective:

$$
\mathcal{L} =
\sum_i l(y_i,\hat{y}_i)
+
\sum_k \Omega(f_k)
$$

The search used 100 randomized configurations with SMOTE placed **inside each cross-validation fold**, preventing synthetic observations from crossing fold boundaries.

---

## Evaluation

For binary classification:

### Precision

$$
\text{Precision}=\frac{TP}{TP+FP}
$$

### Recall

$$
\text{Recall}=\frac{TP}{TP+FN}
$$

### F1-score

$$
F_1 =
2\cdot
\frac{\text{Precision}\cdot\text{Recall}}
{\text{Precision}+\text{Recall}}
$$

### ROC-AUC

The ROC curve evaluates the relationship between the true-positive rate and false-positive rate over possible classification thresholds:

$$
TPR=\frac{TP}{TP+FN}
$$

$$
FPR=\frac{FP}{FP+TN}
$$

AUC summarises the area under this curve.

---

## Results

### Held-out performance

| Model | AUC-ROC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9619 | 0.7431 | 0.6812 | 0.8163 |
| XGBoost | 0.9703 | 0.8216 | 0.8444 | 0.8000 |
| **Random Forest** | **0.9791** | **0.8763** | **0.8912** | **0.8618** |

Random Forest produced the strongest overall F1-score and the best balance between false positives and false negatives on the held-out set.

---

## Explainability

The selected Random Forest model was interpreted using SHAP TreeExplainer.

For a prediction \(f(x)\), SHAP decomposes the output into a base value and feature contributions:

$$
f(x)=\phi_0+\sum_{i=1}^{M}\phi_i
$$

where:

- \(\phi_0\) is the base value;
- \(M=30\) is the number of model features;
- \(\phi_i\) represents the contribution of feature \(i\).

The strongest global predictors identified in the analysis include:

`V14`, `V4`, `V12`, `V17`, and `V10`.

---

## Figures

All publication figures are stored under `figures/` and are referenced directly by the article.

### Figure 1 — Class distribution

![Figure 1 — Class distribution](figures/fig1_class_distribution.png)

*Figure 1. Distribution of legitimate and fraudulent transactions in the original benchmark, illustrating the extreme minority-class imbalance.*

### Figure 2 — Feature correlation structure

![Figure 2 — Correlation heatmap](figures/fig2_correlation_heatmap.png)

*Figure 2. Feature correlation heatmaps for legitimate and fraudulent transactions.*

### Figure 3 — Feature distributions

![Figure 3 — Feature distributions](figures/fig3_feature_distributions.png)

*Figure 3. Distribution of selected high-impact features for legitimate and fraudulent transactions.*

### Figure 4 — SMOTE balancing

![Figure 4 — SMOTE balancing](figures/fig4_smote_balancing.png)

*Figure 4. Training-set class distribution before and after applying SMOTE exclusively to the training partition.*

### Figure 5 — ROC curves and confusion matrices

![Figure 5 — ROC and confusion matrices](figures/fig5_roc_confusion.png)

*Figure 5. ROC curves and confusion matrices for the evaluated classifiers on the untouched held-out test set.*

### Figure 6 — Metric comparison

![Figure 6 — Metric comparison](figures/fig6_metrics_comparison.png)

*Figure 6. Comparative AUC-ROC, F1-score, precision and recall across the three classifiers.*

### Figure 7 — SHAP summary

![Figure 7 — SHAP summary](figures/fig7_shap_summary.png)

*Figure 7. Global SHAP summary showing the magnitude and direction of feature contributions.*

### Figure 8 — SHAP waterfall

![Figure 8 — SHAP waterfall](figures/fig8_shap_waterfall.png)

*Figure 8. Local SHAP waterfall explanation for an individual transaction.*

### Figure 9 — SHAP feature importance

![Figure 9 — SHAP importance](figures/fig9_shap_importance.png)

*Figure 9. Global feature importance derived from absolute SHAP contributions.*

### Figure 10 — End-to-end pipeline

![Figure 10 — Pipeline architecture](figures/fig10_pipeline_architecture.png)

*Figure 10. End-to-end experimental pipeline from raw transactions through leakage-safe preprocessing, modelling, evaluation and explainability.*

---

## Repository structure

```text
imbalance-article-project/
│
├── README.md
├── LICENSE
├── article.md
├── requirements.txt
│
├── figures/
│   ├── fig1_class_distribution.png
│   ├── fig2_correlation_heatmap.png
│   ├── fig3_feature_distributions.png
│   ├── fig4_smote_balancing.png
│   ├── fig5_roc_confusion.png
│   ├── fig6_metrics_comparison.png
│   ├── fig7_shap_summary.png
│   ├── fig8_shap_waterfall.png
│   ├── fig9_shap_importance.png
│   └── fig10_pipeline_architecture.png
│
└── src/
    └── imbalance_analysis.py
```

> The figure filenames above match the analysis notebooks/source material available for this project. If your GitHub repository contains additional scripts or generated artefacts, keep those files in their existing locations rather than renaming them solely for documentation.

---

## Reproducing the experiment

### 1. Clone

```bash
git clone https://github.com/<your-username>/imbalance-article-project.git
cd imbalance-article-project
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add the dataset

Download the ULB Credit Card Fraud Detection dataset from its original public source and configure the path expected by the analysis script.

Do **not** commit the raw dataset if your repository policy or dataset licence does not permit redistribution.

### 5. Run the analysis

```bash
python src/imbalance_analysis.py
```

The script generates the analysis outputs and publication figures used throughout the repository.

---

## Reproducibility principles

This project follows several rules designed to keep the reported metrics defensible:

- split before oversampling;
- fit transformations only on training data;
- keep the held-out test set untouched;
- apply SMOTE within CV folds during XGBoost tuning;
- report multiple imbalance-aware metrics;
- preserve a fixed random seed where applicable;
- keep generated figures separate from source code;
- document the exact evaluation partition.

---

## Research context & author

<p align="center">
  <strong>Zain Ul Abideen</strong><br/>
  <sub>Computer Science • Applied Machine Learning • AI Research • Explainable AI</sub>
</p>

This repository is authored by **Zain Ul Abideen**, a Computer Science graduate focused on turning machine-learning ideas into reproducible, research-oriented systems. His work spans **applied ML, AI, computer vision, explainability, and research engineering**, with an emphasis on building projects that can be inspected, reproduced, and extended rather than treated as isolated experiments.

### Research & technical interests

- **Applied Machine Learning** — practical modelling, evaluation, and experimentation.
- **Artificial Intelligence** — learning systems, model development, and responsible evaluation.
- **Computer Vision** — deep learning, visual recognition, and explainable AI workflows.
- **Explainable AI** — understanding model behaviour through techniques such as SHAP and Grad-CAM.
- **Research Engineering** — connecting rigorous methodology with clean, reproducible implementation.
- **Deno Lab** — the broader project/research context associated with this body of work.
- **Open Research on GitHub** — documenting code, experiments, figures, and findings in a form that others can inspect and build upon.

### Author profile

| | |
|---|---|
| **Author** | Zain Ul Abideen |
| **Background** | Computer Science |
| **Focus** | Applied ML · AI · Computer Vision · Explainable AI |
| **Research style** | Reproducible · Experimental · Methodology-first |
| **Primary platform** | GitHub |
| **Project context** | Deno Lab |

> **Build the model. Challenge the evaluation. Explain the result. Share the work.**

This repository follows that philosophy: the model is only one part of the contribution. The data boundary, preprocessing order, imbalance strategy, evaluation protocol, visual evidence, and explanations are documented alongside it.

---

## Technical article

A companion long-form article explains the experiment, the class-imbalance problem, leakage risks, model comparison, equations, visual results and lessons learned.

**Medium article:**  
`[PASTE YOUR LIVE MEDIUM ARTICLE LINK HERE]`

Replace the placeholder above with the final published Medium URL.

---

## Key takeaways

> **1. Accuracy is not enough.**  
> With a ~577:1 class imbalance, majority-class predictions can look impressive while being operationally useless.

> **2. Resampling must respect the evaluation boundary.**  
> SMOTE belongs inside the training workflow — never before the train/test split.

> **3. Nonlinear ensembles captured the problem better.**  
> Random Forest produced the strongest held-out F1-score in this experiment.

> **4. Evaluation and explanation are separate questions.**  
> A strong classifier still needs an interpretable account of why individual predictions were made.

> **5. Reproducibility is part of the result.**  
> A metric is only meaningful when the data split, preprocessing order and evaluation protocol are clearly defined.

---

## Citation

If you use this repository in academic work, please cite the repository and the accompanying article.

```bibtex
@misc{ulabideen_imbalance_fraud_2026,
  author       = {Zain Ul Abideen},
  title        = {Imbalance-Aware Credit Card Fraud Detection},
  year         = {2026},
  howpublished = {GitHub},
  note         = {Leakage-safe machine learning study using SMOTE, Random Forest, XGBoost, Logistic Regression and SHAP}
}
```

---

## License

Released under the **MIT License**. See [`LICENSE`](LICENSE).

---

<p align="center">
  <sub>Built as a reproducible machine-learning study — where the evaluation protocol matters as much as the model.</sub>
</p>
