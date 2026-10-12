# FARS Road-Accident Severity Classification

### Four machine-learning pipelines compared with stratified cross-validation

Coursework project for Newcastle University's MSc Data Science & AI programme (module **CSC8635**, mini-project 1). It predicts the injury severity of people involved in US road traffic accidents from the FARS dataset, and compares logistic regression, an SVM, a random forest and k-NN under identical preprocessing and evaluation.

📓 **Notebook (code and written report):** [`notebooks/FARS_Accident_Severity_Classification.ipynb`](notebooks/FARS_Accident_Severity_Classification.ipynb)

---

## At a Glance

| | |
|---|---|
| **Task** | Multi-class classification of accident injury severity |
| **Data** | FARS (Fatality Analysis Reporting System): 100,968 records, 29 features, 8 severity classes |
| **Pipelines** | Logistic regression, SVM (RBF), random forest, k-NN |
| **Evaluation** | Stratified 5-fold cross-validation; accuracy and weighted precision, recall, F1 |
| **Best pipeline** | SVM: accuracy 0.7932, weighted F1 0.7862 |
| **Majority-class baseline** | 41.7% accuracy (always predicting "Fatal_Injury") |
| **Framework** | Python, scikit-learn |

## Table of Contents

1. [Background](#1-background)
2. [Problem Statement and Brief](#2-problem-statement-and-brief)
3. [Dataset](#3-dataset)
4. [Solution: Approach](#4-solution-approach)
5. [Implementation](#5-implementation)
6. [Results](#6-results)
7. [Limitations and Future Work](#7-limitations-and-future-work)
8. [Repository Structure](#8-repository-structure)
9. [Setup and Running](#9-setup-and-running)

---

## 1. Background

Road traffic accidents remain a major public-safety concern, and understanding which factors drive the severity of an injury supports better road-safety policy. The FARS dataset records the people involved in accidents together with demographic details, seating position, safety-equipment use, and alcohol and drug indicators. The dataset is large and mixes many categorical features, so it suits machine learning methods that can capture non-linear relationships, but it also raises practical questions about class imbalance and high-dimensional encodings.

## 2. Problem Statement and Brief

The CSC8635 brief asked for the following on the classification dataset:

1. Exploratory data analysis (EDA).
2. Data preprocessing.
3. A train/test split using cross-validation.
4. **Four** machine-learning pipelines for classifying accident severity.
5. Results presented with appropriate metrics.
6. A short report on which approaches were used, which worked best, and why.

The brief also encouraged a broad exploration of options (preprocessing, feature or instance selection, class-imbalance correction, several models, hyper-parameter tuning). The written report sits inside the notebook, as the brief required.

## 3. Dataset

`fars.csv`, provided through the university's course platform: **100,968 rows × 30 columns**. The last column, `INJURY_SEVERITY`, is the target. There are no null values; missing information is encoded as category values such as `Unknown` or `Not_Extricated`.

| Feature type | Count | Examples |
|---|---|---|
| Numerical | 5 | `AGE`, `ALCOHOL_TEST_RESULT`, `DRUG_TEST_RESULTS_(1_of_3)` |
| Categorical | 24 | `CASE_STATE` (51 values), `SEATING_POSITION` (26), `RESTRAINT_SYSTEM-USE`, `EJECTION`, `RACE` |

**Target distribution** (highly imbalanced):

| Severity class | Records | Share |
|---|---|---|
| Fatal_Injury | 42,116 | 41.7% |
| No_Injury | 20,007 | 19.8% |
| Incapaciting_Injury | 15,072 | 14.9% |
| Nonincapaciting_Evident_Injury | 13,890 | 13.8% |
| Possible_Injury | 8,674 | 8.6% |
| Unknown | 901 | 0.9% |
| Injured_Severity_Unknown | 299 | 0.3% |
| Died_Prior_to_Accident | 9 | <0.01% |

> **Not included** in this repository. The notebook expects `fars.csv` (see [Setup and Running](#9-setup-and-running)).

## 4. Solution: Approach

### 4.1 Preprocessing

All preprocessing lives **inside each pipeline**, so its parameters are learned from the training folds only during cross-validation. This prevents data leakage.

| Feature group | Steps |
|---|---|
| Numerical | Median imputation → standardisation (zero mean, unit variance) |
| Categorical | Most-frequent imputation → one-hot encoding (`handle_unknown="ignore"`) |

The two branches are combined with a scikit-learn `ColumnTransformer`.

### 4.2 The four pipelines

| # | Pipeline | Configuration | Imbalance handling |
|---|---|---|---|
| 1 | Logistic regression | `max_iter=1000` | `class_weight="balanced"` |
| 2 | Support vector machine | RBF kernel, `C=1.0` | `class_weight="balanced"` |
| 3 | Random forest | 200 trees | `class_weight="balanced"` |
| 4 | k-nearest neighbours | `k=15`, distance-weighted voting, Minkowski metric | Distance weighting |

All pipelines share the same preprocessing and the same evaluation setup, so differences in performance come from the model, not the experimental design.

### 4.3 Evaluation

- **Cross-validation:** `StratifiedKFold`, 5 splits, shuffled, `random_state=42`. Stratification keeps rare severity classes represented in every fold.
- **Metrics:** accuracy, plus **weighted** precision, recall and F1, so classes count in proportion to their frequency.

## 5. Implementation

- **Language and libraries:** Python, scikit-learn (`Pipeline`, `ColumnTransformer`, `cross_validate`), pandas, NumPy, Matplotlib
- **Reproducibility:** fixed seed (`42`) for cross-validation and for the models that use randomness
- **Environment:** developed on Google Colab

## 6. Results

Mean over the five stratified folds:

| Pipeline | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) |
|---|---|---|---|---|
| Logistic regression | 0.7828 | 0.7878 | 0.7828 | 0.7828 |
| **SVM (RBF)** | **0.7932** | **0.7940** | **0.7932** | **0.7862** |
| Random forest | 0.7755 | 0.7683 | 0.7755 | 0.7714 |
| k-NN | 0.7797 | 0.7664 | 0.7797 | 0.7705 |

The notebook also plots weighted F1 per pipeline and a grouped bar chart of all four metrics.

### What the results mean

- **All four pipelines clearly beat the majority-class baseline** of 41.7% accuracy, so the features carry real information about severity.
- **The SVM is best on accuracy and weighted F1.** The notebook explains this by the RBF kernel's ability to model non-linear interactions between demographic, safety-equipment and behavioural factors.
- **Logistic regression is a strong baseline.** It lands within about one percentage point of the SVM, which suggests a large share of the signal is additive once categorical features are one-hot encoded.
- **Random forest and k-NN score lower.** The notebook attributes this to the high-dimensional, sparse space created by one-hot encoding (distance-based methods suffer most), and to untuned default settings for the forest.
- **The gaps between pipelines are small** (under two points in F1). Fold-to-fold variance was not reported, so differences of this size should be read with caution.

## 7. Limitations and Future Work

**Limitations**

- Hyper-parameters are fixed at sensible defaults (`C=1`, `k=15`, 200 trees) and were **not tuned**, although the brief encouraged tuning. Results for the forest and k-NN in particular may improve with tuning.
- Only weighted metrics are reported. **Per-class results and a confusion matrix are not included**, so performance on the rare severity classes is not visible and weighted averages can hide it.
- The `Unknown` and similar categories are kept as ordinary categories. Because the data has no null values, the median and most-frequent imputers act as safeguards and have no effect.
- Class imbalance is handled with class weights only; resampling methods were not compared.

**Future work**

- Hyper-parameter search (grid or random) with nested cross-validation.
- Per-class metrics, a confusion matrix, and macro-averaged scores.
- Feature selection, and a comparison of imbalance-correction methods (for example SMOTE, undersampling).
- Gradient-boosted trees, which handle high-cardinality categorical data well.

## 8. Repository Structure

```
fars-accident-severity-classification/
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks/
    └── FARS_Accident_Severity_Classification.ipynb
```

The report is written inside the notebook, as the coursework required.

## 9. Setup and Running

The notebook was written for **Google Colab**: it mounts Google Drive and reads `fars.csv` from a personal Drive folder. To run it elsewhere:

1. **Get the data.** Obtain `fars.csv` (100,968 rows; last column `INJURY_SEVERITY`).
2. **Remove the Colab cell.** Delete the `google.colab` / `drive.mount(...)` cell. Set `data_path` to your local `fars.csv`.
3. **Install dependencies.**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the notebook top to bottom:** EDA → preprocessing → cross-validation setup → the four pipelines → results and discussion.

An RBF SVM scales poorly with 100k rows, so expect the cross-validation cell to take a long time.

---

*Author: Emaduddin Asdaq Syed Mohammed, School of Computing, Newcastle University.*
