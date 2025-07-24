# Employee Performance Prediction 🚀

Predicting employee performance ratings using machine learning models.

---

## 📖 Overview

This project analyzes HR data to predict employees’ performance ratings. It includes:

- Exploratory Data Analysis (EDA): detecting distribution issues, outliers, skewness, and categorical relationships.
- Data preprocessing: missing value handling, transformations (e.g., square‑root), scaling, and encoding.
- Machine learning modeling: training classifiers (e.g., Logistic Regression, Decision Tree), evaluating performance with accuracy, ROC-AUC, precision, recall, and F1 scores.
- Model saving for future inference.

---

## 🗂️ File Structure

├── data/
│ └── employee_performance_analysis_preprocessed_data.csv
├── notebooks/
│ └── Employee Performance Prediction.ipynb # full analysis pipeline
└── models/
└── trained_model.pkl # serialized final model


---

## 📊 Data Pipeline

1. **Load Data**  
   Read preprocessed CSV, drop unnecessary columns (e.g., `Unnamed: 0`).

2. **EDA**  
   - Checked basic stats: shape, missing values.  
   - Visualized distributions, boxplots for numerical features.  
   - Explored categorical relationships (e.g., education level vs. performance rating).

3. **Feature Engineering**  
   - Performed square-root transform on skewed features (e.g., `YearsSinceLastPromotion`).  
   - Standard-scaled numerical variables.

4. **Train–Test Split**  
   Divided data into training and testing sets (e.g., 70/30 split with `random_state=42`).

5. **Model Training & Evaluation**  
   - Trained classification models.  
   - Computed evaluation metrics: accuracy, precision, recall, F1 score, ROC-AUC.  
   - Selected the best model based on balanced performance.

---

## 🧠 Modeling Example (Logistic Regression)

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Setup X, y
X = data.drop(columns=['PerformanceRating'])
y = data['PerformanceRating']

# Split & scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train
clf = LogisticRegression(max_iter=1000).fit(X_train_scaled, y_train)

# Evaluate
y_pred = clf.predict(X_test_scaled)
y_proba = clf.predict_proba(X_test_scaled)[:, 1]
print(classification_report(y_test, y_pred))
print('ROC‑AUC:', roc_auc_score(y_test, y_proba))


🔧 Requirements & Setup
Clone the repo:
git clone https://github.com/youruser/Employee-Performance-Prediction.git
cd Employee-Performance-Prediction
Install dependencies:
manually install: pandas, numpy, scikit‑learn, matplotlib, seaborn, scipy.

Launch the notebook:
🔍 Insights & Findings
Key features positively correlated with higher performance ratings:

Environment satisfaction

Last salary hike percentage

Work–life balance

Square-root transformation helped reduce skewness in YearsSinceLastPromotion.

Standard scaling improved model convergence and evaluation consistency.

🆕 Future Work
Try advanced models (Random Forest, XGBoost, Neural Networks).

Apply feature selection or dimensionality reduction (e.g., PCA).

Connect model to web app or API for real-time prediction.

Tune hyperparameters and implement cross-validation for robustness.

Explore model explainability (SHAP, LIME for feature importance).

📝 License
This project is available under the MIT License. Feel free to use, modify, and distribute!

✨ Credits to Narendra6305
Based on  Jupyter notebook analysis. Built with ❤️ using pandas, scikit-learn, and seaborn.

