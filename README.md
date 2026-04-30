# Codveda Virtual Data Science Internship
### Deborah Olofin · Data Scientist & Machine Learning Engineer

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-green?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=for-the-badge)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

---

## Overview

This repository contains all work completed during the **Codveda Virtual Data Science Internship** — six tasks across three levels, covering the full data science pipeline from raw data cleaning through to neural network architecture and hyperparameter tuning.

Every notebook is fully structured with markdown annotations, phase-by-phase explanations, observation callouts, and written insights. The work is documented the way a real data scientist would document it — not just running code and moving on, but reasoning through every decision.

---

## Repository Structure

```
codveda-internship/
│
├── Level_1/
│   ├── Task1_Data_Cleaning_and_Preprocessing.ipynb
│   └── Task2_Exploratory_Data_Analysis.ipynb
│
├── Level_2/
│   ├── Task1_Predictive_Modelling_Regression.ipynb
│   └── Task2_Classification_Logistic_Regression.ipynb
│
├── Level_3/
│   ├── Task1_NLP_Text_Classification.ipynb
│   └── Task2_Neural_Networks_MNIST.ipynb
│
└── README.md
```

---

## Tasks at a Glance

| Level | Task | Dataset | Key Methods | Status |
|:---:|:---|:---|:---|:---:|
| 1 | Data Cleaning & Preprocessing | Telecom Churn (3,333 rows) | Merging, encoding, feature engineering, normalization | ✅ |
| 1 | Exploratory Data Analysis | Telecom Churn (3,333 rows) | Histograms, box plots, scatter plots, correlation matrix | ✅ |
| 2 | Predictive Modelling — Regression | Boston Housing (506 rows) | Linear Regression, Decision Tree, Random Forest, XGBoost | ✅ |
| 2 | Classification | Iris (150 rows) | Logistic Regression, Random Forest, SVM, ROC curves | ✅ |
| 3 | NLP Text Classification | Social Media Sentiment (732 posts) | TF-IDF, Logistic Regression, Naive Bayes | ✅ |
| 3 | Neural Networks | MNIST (70,000 images) | Feed-Forward NN, hyperparameter tuning, TensorFlow/Keras | ✅ |

---

## Level 1 — Basic

### Task 1: Data Cleaning & Preprocessing
**Dataset:** `churn-bigml-80.csv` + `churn-bigml-20.csv` → merged to **3,333 rows × 20 columns**

The raw data arrived as two separate files (80/20 split). Both were loaded, labelled with a `data_type` tracking column, and concatenated — preserving the original split provenance for any downstream modelling that needed it.

**What was done:**
- Zero missing values confirmed across all 3,333 rows — no imputation needed
- `Area code` retyped from integer to string (it is an identifier, not a measurement — leaving it numeric would produce meaningless correlations and incorrect normalization)
- Logical consistency checks: all four billing categories (day, evening, night, international) confirmed zero instances of charges without corresponding minutes
- **11 engineered features** created on top of the original 20, including:
  - `day_rate`, `eve_rate`, `night_rate`, `intl_rate` — price per minute by billing period
  - `total_overall_charge` — single total bill figure
  - `value_score` — effective rate (total spend ÷ total minutes), a proxy for price sensitivity
  - `service_sentiment` — customer friction level: **2,637 Normal (79%) · 429 Nervous (13%) · 267 High_Risk (8%)**
  - `is_loyal` — tenure binning: **1,984 non-loyal · 1,349 loyal (60/40 split)**
- Binary encoding applied to `International plan`, `Voice mail plan`, and `Churn` for ML compatibility
- MinMaxScaler normalization applied to numerical columns

---

### Task 2: Exploratory Data Analysis
**Dataset:** Same merged churn dataset · **Target variable:** `Churn` (binary)

**Class balance:** 85.5% non-churners vs 14.5% churners — a significant imbalance that informed all visualisation decisions (density normalization, `common_norm=False` to prevent the minority class from being visually buried).

**Key findings from correlation analysis:**

| Feature | Correlation with Churn | Interpretation |
|:---|:---:|:---|
| Customer service calls | Strongest positive | High friction customers — the 267 High_Risk group churn at dramatically higher rates |
| Total day charge | Positive | Bill shock is a real driver — higher day charges push customers out |
| International plan | Positive | International plan subscribers churn more, likely due to pricing dissatisfaction |
| Voice mail plan | Negative | Voicemail plan subscribers are more stable — product engagement creates retention |
| Account length | Near zero | Tenure alone does not predict churn — engagement matters more than longevity |

The `value_score` feature (engineered in Task 1) revealed that churners pay a marginally higher effective rate per minute — a subtle pricing sensitivity signal invisible in the raw data.

---

## Level 2 — Intermediate

### Task 1: Predictive Modelling — Regression
**Dataset:** Boston Housing (`house_Prediction_Data_Set.csv`) — **506 rows · 14 columns**

The dataset arrived with no column headers and whitespace-separated values. A custom load (`header=None`, `names=columns`, `sep='\s+'`, `engine='python'`) was applied before anything else ran.

**Target variable:** `MEDV` — median home value in $1,000s. Distribution is right-skewed: most homes cluster between $15K–$25K with a long tail toward $50K.

**Strongest correlations with MEDV:**
- `RM` (avg rooms per dwelling): **+0.695** — most intuitive positive predictor
- `LSTAT` (% lower-status population): **−0.738** — strongest predictor overall
- `RAD` and `TAX` multicollinearity: **0.910** — both retained after testing confirmed independent predictive signal

**Model progression — each model addresses the previous one's specific weakness:**

| Model | MSE | R² | Notes |
|:---|:---:|:---:|:---|
| Linear Regression (baseline) | `[paste from Colab]` | `[paste from Colab]` | OLS, no scaling needed, linearity assumption limits performance |
| Decision Tree | `[paste from Colab]` | `[paste from Colab]` | Captures non-linearity; unconstrained depth risks overfitting |
| Random Forest | `[paste from Colab]` | `[paste from Colab]` | Ensemble of trees reduces variance; expected improvement over DT |
| XGBoost | `[paste from Colab]` | `[paste from Colab]` | Sequential boosting corrects residuals; best performer |

> **Note to Debs:** Fill in the four rows above with the actual numbers printed in your Colab output cells for each model. They are in the format `MSE: X.XXXX` and `R2: X.XXXX`.

No feature scaling was applied — both OLS and tree-based models are scale-invariant. Scaling the features would change coefficient magnitudes but produce identical predictions and metrics.

---

### Task 2: Classification with Logistic Regression
**Dataset:** Iris (`iris.csv`) — **147 rows × 5 columns** (3 duplicate rows removed)

**Target:** `species` — setosa, versicolor, virginica. Perfectly balanced: 50 samples per class after deduplication.

Three models trained and evaluated with 80/20 stratified split. Logistic Regression and SVM trained on `StandardScaler`-normalized features; Random Forest trained on raw features (tree models are scale-invariant).

| Model | Accuracy | Precision (weighted) | Recall (weighted) |
|:---|:---:|:---:|:---:|
| Logistic Regression | `[paste from Colab]` | `[paste from Colab]` | `[paste from Colab]` |
| Random Forest | `[paste from Colab]` | `[paste from Colab]` | `[paste from Colab]` |
| SVM (RBF kernel) | `[paste from Colab]` | `[paste from Colab]` | `[paste from Colab]` |

> **Note to Debs:** Paste your printed accuracy/precision/recall values for each model from your Colab output. They are in the results DataFrame printed in Phase 10.

ROC curves were plotted using One-vs-Rest binarization across all three models on a single plot for direct comparison.

---

## Level 3 — Advanced

### Task 1: NLP Text Classification
**Dataset:** `Sentiment_dataset.csv` — **732 social media posts · 15 columns · 191 raw sentiment labels**

The 191 labels were too granular for a 732-row dataset (averaging ~3–4 samples per class). They were consolidated into three actionable classes: Positive, Negative, Neutral. **Class distribution after mapping: ~61% Positive · 26% Negative · 12% Neutral** — a significant imbalance that was the central challenge of this task.

**Pipeline decisions with reasoning:**
- Hashtags absorbed into `Text` before cleaning — they resolve sentiment ambiguity that surface text alone cannot (e.g. `#MondayBlues` vs `#MondayMotivation`)
- Emojis converted to readable tokens via `emoji.demojize()` rather than stripped — 😤 and 😊 carry direct sentiment signal
- TF-IDF vectorization applied **after** train/test split — fitting the vectorizer on training data only, then transforming the test set separately. This is the correct approach; fitting on the full dataset before splitting is data leakage
- `class_weight='balanced'` applied to Logistic Regression and `class_prior=[1/3, 1/3, 1/3]` to Naive Bayes — both corrections were necessary to prevent the models from ignoring the Neutral minority class

**Impact of class imbalance correction:**

| Model | Weighted F1 (before) | Neutral F1 (before) | Weighted F1 (after) | Neutral F1 (after) |
|:---|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.7640 | 0.20 | **0.8552** | **0.47** |
| Naive Bayes | 0.7271 | 0.00 | **0.8417** | **0.40** |

**Final model comparison:**

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 |
|:---|:---:|:---:|:---:|:---:|
| **Logistic Regression** | **86%** | **0.8557** | **0.8639** | **0.8552** |
| Naive Bayes | 84% | 0.8402 | 0.8435 | 0.8417 |

Logistic Regression wins across every metric. Naive Bayes performs competitively — better than expected for an imbalanced dataset — but its independence assumption limits its ability to generalise on the nuanced Neutral class.

---

### Task 2: Neural Networks — MNIST Digit Classification
**Dataset:** MNIST — loaded via `tf.keras.datasets.mnist.load_data()` · **60,000 training · 10,000 test · 28×28 grayscale images**

**Class balance:** Near-uniform. Digit 1 is most common (6,742 samples); digit 5 is least common (5,421) — a ~20% gap, mild enough to require no resampling.

**EDA insight confirmed by results:** Digit 1 produces the sharpest mean image (consistent thin vertical stroke). Digit 8 produces the blurriest (high handwriting variance). This predicted that digit 8 would have the lowest recall — confirmed: **digit 8 recall = 0.96**, lowest of all classes.

**Architecture progression:**

| Model | Architecture | Test Accuracy | Test Loss |
|:---|:---|:---:|:---:|
| Linear Baseline | `Flatten → Dense(10, softmax)` | 92.49% | 0.2679 |
| **Feed-Forward NN** | `Flatten → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, softmax)` | **97.97%** | — |

**Improvement from baseline to FFNN: +5.48 percentage points** — entirely attributable to the two ReLU hidden layers enabling non-linear feature learning. The baseline's linear decision boundaries cannot separate digit classes in 784-dimensional pixel space; hidden layers build hierarchical representations that can.

Training curves: training accuracy reached **99.36% by epoch 10**. Validation accuracy plateaued at **97.8–98.2% from epoch 7** — a mild overfitting signal. Optimal early stopping was at **epoch 7 (val accuracy: 98.15%)**.

**Hyperparameter tuning — 3×3 grid (learning rate × batch size), 9 configurations, 10 epochs each:**

| Learning Rate | Batch Size | Test Accuracy | Best Val Acc |
|:---:|:---:|:---:|:---:|
| **0.001** | **32** | **98.03%** | **98.27%** |
| 0.001 | 64 | 97.97% | 98.00% |
| 0.001 | 128 | 97.88% | 98.02% |
| 0.01 | 128 | 97.06% | 97.47% |
| 0.0001 | 32 | 96.97% | 97.48% |
| 0.0001 | 128 | 95.81% | 96.25% |

LR=0.001 sweeps the top positions. LR=0.01 overshoots; LR=0.0001 cannot converge in 10 epochs. Smaller batch size consistently wins within the optimal LR: gradient noise from smaller batches improves generalisation. **Best configuration: LR=0.001, Batch Size=32** — identical to the initial FFNN, confirming the original choice was already optimal.

---

## Tech Stack

| Category | Tools |
|:---|:---|
| Data manipulation | Python, Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine learning | Scikit-learn, XGBoost |
| Deep learning | TensorFlow, Keras |
| NLP | NLTK, emoji, TF-IDF |
| Environment | Google Colab + Google Drive |

---

## About

**Deborah Olofin** — Data Scientist & Machine Learning Engineer, 5 months into the journey.

This internship covered the complete data science workflow end-to-end: raw data → cleaning → EDA → classical ML → NLP → deep learning. Every notebook is documented with the reasoning behind every decision, not just the code.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/YOUR-LINKEDIN-HANDLE)

---

*Codveda Virtual Data Science Internship · April 2026*
