# TalentShift: Predicting Data Scientist Job Changes

A machine learning project that predicts whether a candidate who completes a company's training course is likely to **look for a new job** or **stay with the company**. The goal is to help HR teams plan hiring, reduce training costs, and focus on candidates who are more likely to stay.

---

## Problem Statement

Companies that run data science training programs want to know which enrollees are genuinely interested in working for them after training. This is a **binary classification** problem:

- `0` → Not looking for a job change
- `1` → Looking for a job change

The dataset is **imbalanced**: most candidates belong to class 0. The project handles this with class weighting.

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
- Examined target class distribution
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
- **Ordinal Encoding** for categorical features, with unknown categories handled
- **Standard Scaling** for numeric features (`city_development_index`, `training_hours`, `experience`)
- The encoder and scaler were fit on training data only, to avoid data leakage

### 4. Models Trained

| Model | Imbalance Handling |
|---|---|
| Decision Tree (`max_depth=6`) | `class_weight='balanced'` |
| Random Forest (300 trees, `max_depth=10`) | `class_weight='balanced'` |
| Logistic Regression | none |
| SVM (RBF kernel, `C=10`) | `class_weight='balanced'` |
| XGBoost (100 trees, `max_depth=4`, `lr=0.05`) | `scale_pos_weight` |

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
| Decision Tree | x.xxx | x.xxx |
| Random Forest | x.xxx | x.xxx |
| Logistic Regression | x.xxx | x.xxx |
| SVM | x.xxx | x.xxx |
| **XGBoost** | **x.xxx** | **x.xxx** |

**XGBoost** was selected as the final model and saved for deployment.

Key insight: `city_development_index` is one of the strongest predictors. Candidates from less-developed cities are more likely to be looking for a job change.

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

> New data must go through the same cleaning steps (especially the `experience` conversion) before encoding and scaling.

---

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- XGBoost
- Matplotlib, Seaborn
- joblib
- Google Colab

---

## How to Run

```bash
git clone https://github.com/<your-username>/talentshift-job-change-prediction.git
cd talentshift-job-change-prediction
pip install -r requirements.txt
```

Then open the notebook in Jupyter or Google Colab and run all cells.

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
- Try SMOTE or threshold tuning for better minority-class recall
- Add SHAP values for model explainability
- Deploy as a web app with Streamlit or Flask

---

## Author

**Your Name**
[LinkedIn](https://linkedin.com/in/your-profile) • [GitHub](https://github.com/your-username)
