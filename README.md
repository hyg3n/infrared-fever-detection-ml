# Predicting Oral Temperature and Fever Detection using Infrared Thermography

**Author:** Hygen Amoniku  

This repository contains a single end-to-end machine learning case study on predicting oral temperature and detecting fever from infrared thermography and contextual data. All code, analysis, and figures are implemented in a Jupyter notebook.

---

## 1. Project Overview

This project uses infrared thermography data to:

1. **Predict continuous oral temperature** (regression)
   - **Fast mode:** `aveOralF`
   - **Monitor mode:** `aveOralM`
2. **Detect fever** using clinically relevant thresholds (classification)
   - **feverF:** derived from `aveOralF`  
   - **feverM:** derived from `aveOralM`

The workflow is deliberately structured as a full ML pipeline:

- Data import and exploratory data analysis (EDA)  
- Data cleaning and target engineering  
- Train/validation/test splitting with stratification  
- Preprocessing via `ColumnTransformer` and pipelines  
- Model training and hyperparameter tuning  
- Threshold tuning for imbalanced classification  
- Final evaluation on a held-out test set  
- Model interpretation via feature importance and permutation importance

Throughout, the **test set is kept strictly untouched** until the very end to avoid data leakage.

---

## 2. Dataset

- **Source:** UCI Machine Learning Repository – *Infrared Thermography Temperature* dataset (ID 925)  
- **Instances:** 1,020 subjects  
- **Original features:** 33 (plus 2 target variables)  
- **Targets:**
  - `aveOralF` – Oral temperature measured in fast mode  
  - `aveOralM` – Oral temperature measured in monitor mode  

### Feature types

- **Demographics**
  - `Gender` (Male/Female)  
  - `Age` (age ranges, later cleaned and merged)  
  - `Ethnicity` (multiple categorical groups)

- **Environmental/contextual**
  - `T_atm` (ambient temperature)  
  - `Humidity` (relative humidity)  
  - `Distance` (subject–camera distance)  
  - `T_offset1` (blackbody temperature offset)

- **Infrared temperature measurements**
  - Canthus-related features (`T_RC1`, `T_LC1`, wet/dry subregions, maxima)  
  - Forehead regions (`T_FH*` variants)  
  - Mouth region (`T_OR1`, `T_OR_Max1`)  
  - Overall face maxima (`T_Max1`)  
  - Extended canthi aggregates (`canthiMax1`, `canthi4Max1`)

The final preprocessed design matrix expands to **39 features**, due to:

- 30 continuous numeric features  
- 2 one-hot-encoded `Gender` columns  
- 6 one-hot-encoded `Ethnicity` columns (present in the train split)  
- 1 ordinal-coded `Age` feature  

---

## 3. Prediction Tasks

### 3.1 Regression

- **Goal:** predict continuous oral temperature in °C  
- **Targets:**
  - `aveOralF` (fast mode – noisier, more challenging)  
  - `aveOralM` (monitor mode – more stable, easier to predict)

### 3.2 Classification

Binary fever labels are derived from oral temperatures using a clinical threshold:

- `feverF` = 1 if `aveOralF ≥ 37.5°C`, else 0  
- `feverM` = 1 if `aveOralM ≥ 37.5°C`, else 0  

Class balance:

- `feverF`: ~**6.37%** positives (highly imbalanced)  
- `feverM`: ~**10.88%** positives  

Because of this imbalance, **F1-score** (along with precision, recall and ROC-AUC) is used as the primary classification metric, rather than accuracy.

---

## 4. Methodology

### 4.1 Exploratory Data Analysis

The notebook performs:

- Head and summary statistics for features and targets  
- Missing value inspection (only two missing values in `Distance`)  
- Histograms for numeric features and both oral temperature targets  
- Bar plots for `Gender`, `Age`, and `Ethnicity`  
- A **correlation matrix** between all continuous features and both targets

Key observations:

- Strong positive correlations between oral temperatures and facial region temperatures (especially **canthi**, **mouth region**, and **T_Max1**).  
- `aveOralF` and `aveOralM` are strongly correlated (Pearson r ≈ 0.88).  
- Inconsistent age categories (`21-25`, `26-30`, `21-30`) are detected and later merged.

### 4.2 Data Cleaning & Target Engineering

- Created binary fever labels:
  - `feverF` and `feverM` as described above.  
- Fixed the **Age** feature by merging overlapping ranges into a single `21–30` category:
  - Final age categories: `18–20`, `21–30`, `31–40`, `41–50`, `51–60`, `>60`.

### 4.3 Train / Validation / Test Strategy

1. **Train/test split**

   - 80% train (816 instances), 20% test (204 instances)  
   - **Stratified on `feverF`** to preserve the rare positive rate across splits  
   - Because of the strong correlation between `aveOralF` and `aveOralM`, this also preserves the `feverM` distribution.

2. **Classification internal validation split**

   - From the training set, an additional **20% validation split** is created for classification tasks:
     - Training (classification): 652 rows  
     - Validation (classification): 164 rows  
   - Split is **again stratified on `feverF`**.  
   - This validation set is used for:
     - Hyperparameter tuning  
     - **Threshold optimisation** (for F1-score)  
     - Fair comparison across all classifiers  

3. **Regression** uses:
   - 10-fold cross-validation on the full training set (816 rows) for model selection and tuning.  
   - The test set is only evaluated **once** at the end.

### 4.4 Preprocessing Pipelines

Two separate preprocessing configurations are used:

- **Distance-based models (scaled):**
  - Numeric features: median imputation + `StandardScaler`  
  - Categorical features:
    - `Gender`, `Ethnicity`: `OneHotEncoder` with `handle_unknown='ignore'`  
    - `Age`: `OrdinalEncoder` with explicit category ordering  
  - Implemented with a `ColumnTransformer` named `preprocessor_scaled`.  
  - Outputs: `X_train_scaled`, `X_test_scaled`.

- **Tree-based models (unscaled):**
  - Numeric features: median imputation only (no scaling)  
  - Categorical features: same one-hot and ordinal encoders as above  
  - Implemented as `preprocessor_tree`.  
  - Outputs: `X_train_tree`, `X_test_tree`.

Note: In this notebook, the scaler is fitted on the full training set before cross-validation. This slightly overestimates CV performance, but **the test set remains untouched**, so final test metrics are still unbiased. In production, this would be wrapped in a `Pipeline` to re-fit scaling within each CV fold.

### 4.5 Reproducibility Setup

- Global seed `SEED = 42` applied to:
  - Python `random`  
  - NumPy  
  - TensorFlow/Keras (`tf.keras.utils.set_random_seed`)  
- TensorFlow deterministic ops enabled via:
  - `tf.config.experimental.enable_op_determinism()`  

---

## 5. Models

### 5.1 Regression Models

All regression models are trained on **scaled** features (`X_train_scaled`):

- **Linear Regression** (baseline)
- **Ridge Regression** (GridSearch over `alpha`)
- **Lasso Regression** (GridSearch over `alpha`)
- **Elastic Net** (GridSearch over `alpha` and `l1_ratio`)
- **Polynomial Regression (degree + Ridge)**  
  - Pipeline: `PolynomialFeatures` → `Ridge`  
  - GridSearch over polynomial degree (2–4) and `alpha`
- **k-NN Regression**  
  - GridSearch over `n_neighbors` (odd k from 3–21)
- **SGDRegressor**  
  - GridSearch over penalty, learning rate, `max_iter`, and tolerance
- **Neural Network Regressor (MLP using Keras)**  
  - Tuned with **Bayesian optimisation** using Keras Tuner  
  - Two-stage process:
    - Broad exploratory search over architecture and hyperparameters  
    - Refined Bayesian search around the best region  

Metrics: **RMSE, MAE, R²** on the *held-out test set*.

### 5.2 Classification Models

All classifiers are trained on **scaled** features (`X_train_scaled` → classification splits):

- **Logistic Regression**
  - `class_weight='balanced'`
  - GridSearch over `C`, `penalty` (`l1`, `l2`), solvers (`liblinear`, `saga`)
- **SGDClassifier**
  - `class_weight='balanced'`
  - GridSearch over `alpha`, penalty (`l1`, `l2`, `elasticnet`), etc.
- **MLPClassifier (scikit-learn)**
  - Separate hyperparameter grids for `feverF` and `feverM`  
  - GridSearch over `hidden_layer_sizes`, activation, `alpha`, early stopping
- **RandomForestClassifier**
  - `class_weight='balanced'`
  - GridSearch over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`

For each classifier:

1. Hyperparameters tuned via GridSearchCV (5-fold CV, F1-score).  
2. Performance evaluated on the **validation** set at default threshold 0.5.  
3. **Threshold tuning** on validation data:
   - Scan thresholds from 0.1 to 0.9  
   - Select the threshold that maximises F1-score  
4. Best model + optimal threshold per target are selected for **final test evaluation**.

Metrics: **F1, precision, recall, accuracy, ROC AUC**, plus confusion matrices.

---

## 6. Key Results

### 6.1 Regression – Oral Temperature Prediction

On the **held-out test set**, the following patterns emerge:

#### aveOralF (fast mode)

| Model             | R²    | RMSE  | MAE   |
|-------------------|-------|-------|-------|
| Linear Regression | 0.645 | 0.239 | 0.182 |
| Ridge             | 0.656 | 0.236 | 0.175 |
| Lasso             | 0.655 | 0.236 | 0.178 |
| Elastic Net       | 0.656 | 0.236 | 0.177 |
| Polynomial        | 0.581 | 0.260 | 0.197 |
| k-NN              | 0.639 | 0.241 | 0.171 |
| SGDRegressor      | 0.637 | 0.242 | 0.185 |
| **MLP**           | **0.671** | **0.230** | **0.172** |

- Regularised linear models (**Ridge, Lasso, Elastic Net**) all perform similarly and clearly outperform Polynomial Regression and SGD in terms of stability and R².
- The **MLP regressor** is the **best overall** model for `aveOralF`, achieving:
  - R² ≈ **0.671**
  - RMSE ≈ **0.230**
  - MAE ≈ **0.172**
- Polynomial Regression (degree 2) expands the feature space to ~819 features, which is comparable to the number of training samples, leading to overfitting and worse generalisation.

#### aveOralM (monitor mode)

| Model             | R²    | RMSE  | MAE   |
|-------------------|-------|-------|-------|
| Linear Regression | 0.759 | 0.267 | 0.204 |
| Ridge             | 0.762 | 0.266 | 0.201 |
| Lasso             | 0.763 | 0.265 | 0.200 |
| Elastic Net       | 0.762 | 0.265 | 0.200 |
| Polynomial        | 0.711 | 0.293 | 0.229 |
| k-NN              | 0.720 | 0.288 | 0.206 |
| SGDRegressor      | 0.759 | 0.267 | 0.203 |
| **MLP**           | **0.777** | **0.257** | **0.196** |

- Monitor mode is **easier to predict**: all models achieve higher R² than for fast mode.  
- Simple regularised linear models (Ridge/Lasso/Elastic Net) already perform very well.  
- The **MLP regressor** again achieves the **best performance**:
  - R² ≈ **0.777**
  - RMSE ≈ **0.257**
  - MAE ≈ **0.196**
- However, gains over Lasso/Ridge are relatively small and may not be clinically significant, so **linear models remain attractive for deployment** due to their interpretability.

---

### 6.2 Classification – Fever Detection

#### feverF – Final model: Random Forest (with threshold tuning)

**Test set (13 positive cases)**

- **Default threshold 0.50:**
  - Accuracy: 0.946  
  - Precision: 0.562  
  - Recall: 0.692  
  - F1-score: 0.621  
  - ROC AUC: 0.972  

- **Tuned threshold 0.41 (from validation):**
  - Accuracy: 0.951  
  - Precision: 0.579  
  - Recall: 0.846  
  - F1-score: 0.688  

The tuned threshold increases **recall** (captures more true fever cases) while keeping precision similar. This is a desirable trade-off in a medical triage context. The lower F1 on the test set compared to validation is largely due to the **very small number of positive cases**, which makes F1 and recall very sensitive to a handful of misclassifications.

#### feverM – Final model: Logistic Regression (default threshold)

**Test set (23 positive cases)**

- **Default threshold 0.50:**
  - Accuracy: 0.966  
  - Precision: 0.786  
  - Recall: 0.957  
  - F1-score: 0.863  
  - ROC AUC: 0.974  

- **Tuned threshold 0.62:**
  - Accuracy: 0.956  
  - Precision: 0.818  
  - Recall: 0.783  
  - F1-score: 0.800  

For `feverM`, the **default threshold (0.50)** offers the best balance between precision and recall, and thus the highest F1-score. Raising the threshold to 0.62 improves precision slightly but hurts recall and F1, which may be less desirable if missing fevers carries a higher cost than falsely flagging them.

---

### 6.3 Predictive Features

To understand model behaviour, the notebook computes:

- **Permutation importance** for the MLP Classifier (feverF) on the validation set  
- **Feature importance** for the Random Forest (feverM) on the validation set  

Key patterns:

- High-importance features include:
  - **Facial maximum temperature** (`T_Max1`)  
  - **Mouth region temperatures** (`T_OR1`, `T_OR_Max1`)  
  - **Canthi-related features** (`canthiMax1`, `canthi4Max1`, `T_RC*`, `T_LC*`)  
  - Several **forehead measurements** (`T_FHBC1`, `T_FHC_Max1`, etc.)

- Demographic features (`Gender`, `Ethnicity`) show smaller but non-zero contributions in some models, suggesting potential subgroup differences but also highlighting the need to monitor for bias.

Importantly, all interpretation is done on the **validation set** to avoid leaking information from the test set.

---

## 7. How to Use this Notebook

To run this project, you will need:
- **Python 3.5+**
- **Jupyter Notebook / JupyterLab** or **Google Colab**
- (Recommended): **[Google Colab](https://colab.research.google.com/)**, since no local setup is required.


**Core libraries**
`numpy`, `pandas`, `matplotlib`, `scikit-learn`, `tensorflow` / `keras`, `keras-tuner`, `ucimlrepo` (for fetching the UCI dataset)

## 7.1 Suggested Setup

The notebook is fully self-contained: all required libraries (including TensorFlow and Keras Tuner) are automatically installed when the notebook is run.

### Option A - **Google Colab**(Recommended)
- Upload the notebook to Google Colab
- Click **Runtime → Run all**
- All dependencies install automatically

### Option B - Local Setup

Install the required packages:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras keras-tuner ucimlrepo
```

Then open the notebook using Jupyter or VS Code and execute all cells.

## 8. Reproducibility Notes

- A fixed **`SEED = 42`** is used throughout the entire notebook.
- NumPy, Python's built-in random, and Keras/TensorFlow are all initialised using this seed.
- TensorFlow deterministic ops are enabled via:

```python
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()
```

- Some operations may still vary slightly across different hardware (CPU vs GPU, BLAS/LAPACK differences), but results will be stable **on the same machine/environment**.
- Preprocessing scalers are fitted once on the training set prior to cross-validation.  
  - This introduces very small optimism in CV metrics,  
  - **but test-set metrics remain fully unbiased** because the test data is never used during training.

---

## 9. Limitations & Future Work

### **Current Limitations**
- **Class imbalance** is significant, especially for `feverF` (~6%). This increases sensitivity of F1-score and recall to even one misclassified test sample.
- **Polynomial Regression** leads to feature explosion (819+ engineered features), making overfitting much more likely for this dataset size.
- **Demographic features** (Gender, Ethnicity) are included but fairness across subgroups is not evaluated.
- **Preprocessing is applied before cross-validation** rather than inside a scikit-learn Pipeline, which is acceptable for exploration but not ideal for production.

### **Future Enhancements**
- Wrap preprocessing and modelling into a **single scikit-learn Pipeline** for cleaner CV and deployment.
- Improve classification calibration (e.g., Platt scaling, isotonic regression) for more reliable probability thresholds.
- Add basic **fairness diagnostics** to check performance consistency across demographic groups.
- Investigate **uncertainty estimation** techniques for medical triage use cases.

---


## 10. Acknowledgements

- **Dataset:**  
  *Infrared Thermography Temperature* dataset from the UCI Machine Learning Repository.

- **Libraries & Tools Used:**  
  - scikit-learn  
  - TensorFlow / Keras  
  - Keras Tuner  
  - NumPy & Pandas  
  - Matplotlib  
  - ucimlrepo for dataset retrieval

This notebook serves as a complete, transparent end-to-end case study showing how to progress from raw infrared measurements to validated regression and classification models, while emphasising reproducibility, correctness of data splitting, and methodical threshold tuning.

---

## Skills Demonstrated

This project showcases a complete **end-to-end machine learning workflow** for both regression and classification in a medically relevant, noisy, and imbalanced dataset, with strong emphasis on preprocessing, model development, evaluation, interpretability, and reproducibility.

### Data Preprocessing & Feature Engineering
- Built reusable preprocessing pipelines using **ColumnTransformer**
- **Median imputation**, **StandardScaler**, and selective scaling for distance-based models
- **One-hot encoding** (Gender, Ethnicity) and **ordinal encoding** (Age groups)
- Cleaned and merged inconsistent categorical age brackets
- Constructed **binary clinical fever labels** using medically accepted thresholds
- **Stratified splits** to preserve rare fever cases

### Model Development & Training
- Implemented and compared multiple model families for regression and classification:
  - **Regression:** Linear, Ridge, Lasso, Elastic Net, Polynomial, k-NN, SGD, MLP  
  - **Classification:** Logistic Regression, SGDClassifier, MLPClassifier, Random Forest
- Applied **GridSearchCV** for linear and distance-based models
- Applied **Bayesian Hyperparameter Optimisation** for neural networks (Keras Tuner)
- Used **early stopping**, **dropout**, and controlled learning rates in MLPs
- Ensured full **reproducibility** using fixed seeds and deterministic TensorFlow operations

### Evaluation & Model Selection
- Regression metrics: **RMSE**, **MAE**, **R²**
- Classification metrics: **F1**, **Precision**, **Recall**, **ROC-AUC**, **PR-AUC**
- Applied **threshold optimisation** to address imbalance and improve clinical recall
- Analysed recall–precision trade-offs, especially for fever detection where false negatives matter

### Interpretability & Insights
- Explored model behaviour using:
  - **Coefficient analysis** (linear models)
  - **Feature importance** (Random Forest)
  - **Permutation importance** (MLP models)
- Identified key predictive thermal regions (e.g., **canthi**, **mouth region**, **T_Max1**)
- Explained why **monitor-mode temperature (aveOralM)** is easier to predict
- Linked model behaviour to physiological temperature variation and sensor noise

### Software Engineering Practices
- Structured a clean, reproducible Jupyter workflow
- Implemented **modular helper functions** for evaluation and plotting
- Ensured strict separation between training, validation, and test sets
- Removed leakage risks and maintained proper ML experimentation hygiene

### Applied Medical ML Reasoning
- Incorporated **clinical fever thresholds** into labelling
- Prioritised **recall** in fever detection to align with triage requirements
- Considered sensor variability, environmental noise, and anatomical heat distribution

## Project Structure

```text
.
├── ML_Fever_Prediction_Infrared_Pipeline.ipynb   # Main notebook (EDA, preprocessing, modelling, evaluation)
├── README.md
└── LICENSE                                       # CC BY 4.0 license
