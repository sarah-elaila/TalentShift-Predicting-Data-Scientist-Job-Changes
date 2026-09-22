# TalentShift: Predicting Data Scientist Job Changes

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Best%20Model-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

A machine learning project that predicts whether a candidate who completes a company's data science training course is likely to **look for a new job** or **stay with the company**. The goal is to help HR teams plan hiring, reduce training costs, and focus on candidates who are more likely to stay.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Key Insights](#key-insights)
- [Saved Artifacts](#saved-artifacts)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Problem Statement

Companies that run data science training programs want to know which enrollees are genuinely interested in working for them after training. This is a **binary classification** problem:

- `0` → Not looking for a job change
- `1` → Looking for a job change

The dataset is **imbalanced**: most candidates belong to class 0. The project handles this with class weighting (`class_weight='balanced'` and `scale_pos_weight`).

---

## Dataset

The dataset is the **HR Analytics: Job Change of Data Scientists** dataset (available on Kaggle).

| Feature | Description |
|---|---|
| `enrollee_id` | Unique candidate ID (dropped before training) |
| `city` | City code |
| `city_development_index` | Development index of the city (scaled 0–1) |
| `gender` | Candidate gender |
| `relevent_experience` | Whether the candidate has relevant experience |
| `enrolled_university` | Type of university course enrolled in, if any |
| `education_level` | Education level |
| `major_discipline` | Education major |
| `experience` | Total years of experience |
| `company_size` | Number of employees at current employer |
| `company_type` | Type of current employer |
| `last_new_job` | Years between previous job and current job |
| `training_hours` | Training hours completed |
| `target` | 0 = not looking for a change, 1 = looking for a change |

---

## Project Workflow

### 1. Exploratory Data Analysis
- Checked shape, data types, missing values, and duplicates
- Examined the target class distribution
- Visualized categorical features, including missing values
- Plotted histograms and boxplots for numeric features to detect outliers
- Compared feature distributions across target classes
- Built a correlation heatmap for numeric features

### 2. Data Cleaning
- `gender`, `major_discipline`, `company_size`, `company_type`: missing values filled with `"Unknown"`
- `enrolled_university`, `education_level`, `last_new_job`: filled with the mode
- `experience`: converted `">20"` → `21` and `"<1"` → `0`, cast to numeric, filled with the median
- Before/after visualizations were used to confirm the cleaning

### 3. Preprocessing
- Dropped `enrollee_id` because it carries no predictive signal
- 80/20 train-test split (`random_state=42`)
- **Ordinal Encoding** for categorical features, with unknown categories mapped to `-1`
- **Standard Scaling** for numeric features (`city_development_index`, `training_hours`, `experience`)
- The encoder and scaler were fit on training data only, to avoid data leakage

### 4. Models Trained

| Model | Key Parameters | Imbalance Handling |
|---|---|---|
| Decision Tree | `max_depth=6` | `class_weight='balanced'` |
| Random Forest | 300 trees, `max_depth=10`, `min_samples_leaf=5` | `class_weight='balanced'` |
| Logistic Regression | `max_iter=1000` | none |
| SVM | RBF kernel, `C=10`, `gamma='scale'` | `class_weight='balanced'` |
| XGBoost | 100 trees, `max_depth=4`, `learning_rate=0.05`, `reg_alpha=1` | `scale_pos_weight` |

### 5. Evaluation
Each model was evaluated with:
- Accuracy
- Precision, Recall, and F1-score (classification report)
- Confusion matrix
- ROC curve and ROC-AUC score

Train and test reports were compared to check for overfitting.

---

## Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Decision Tree | 0.779 | 0.782 |
| Random Forest | **0.789** | 0.800 |
| Logistic Regression | 0.772 | 0.774 |
| SVM | 0.768 | 0.761 |
| **XGBoost** | 0.782 | **0.804** |

### Why XGBoost?

Random Forest achieved slightly higher accuracy, but **XGBoost achieved the highest ROC-AUC (0.804)**. Because the dataset is imbalanced, accuracy can be misleading: a model can score well just by predicting the majority class. ROC-AUC measures how well the model separates the two classes across all thresholds, so it is the more reliable metric here. For this reason, **XGBoost was selected as the final model** and saved for deployment.

---

## Key Insights

- **`city_development_index` is the strongest predictor.** Candidates from less-developed cities are much more likely to be looking for a job change.
- Tree-based ensemble models (XGBoost and Random Forest) outperformed linear and kernel-based models.
- All models reached similar performance (ROC-AUC 0.76–0.80), which suggests the features carry a moderate amount of signal and that feature engineering may matter more than further model tuning.

---

## Saved Artifacts

```
models/
├── xgb_model.pkl   # trained XGBoost classifier
├── encoder.pkl     # fitted OrdinalEncoder
└── scaler.pkl      # fitted StandardScaler
```

Load them with:

```python
import joblib

model   = joblib.load('models/xgb_model.pkl')
encoder = joblib.load('models/encoder.pkl')
scaler  = joblib.load('models/scaler.pkl')
```

> **Note:** New data must go through the same cleaning steps (especially the `experience` conversion) before encoding and scaling.

---

## Tech Stack

- **Language:** Python
- **Data handling:** pandas, NumPy
- **Machine learning:** scikit-learn, XGBoost
- **Visualization:** Matplotlib, Seaborn
- **Model saving:** joblib
- **Environment:** Google Colab

---

## How to Run

1. Clone the repository:
```bash
   git clone https://github.com/sarah-elaila/TalentShift-Predicting-Data-Scientist-Job-Changes.git
   cd TalentShift-Predicting-Data-Scientist-Job-Changes
```

2. Install the dependencies:
```bash
   pip install -r requirements.txt
```

3. Open the notebook in Jupyter Notebook or Google Colab and run all cells.

**requirements.txt**
```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
joblib
```

---

## Future Improvements

- Use One-Hot or Target Encoding for nominal features like `city` and `company_type`
- Tune hyperparameters with GridSearchCV or Optuna
- Try SMOTE or decision-threshold tuning to improve minority-class recall
- Add SHAP values for model explainability
- Deploy the model as a web app with Streamlit or Flask

---
