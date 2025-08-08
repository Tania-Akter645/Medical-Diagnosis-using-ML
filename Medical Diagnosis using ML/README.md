# 🩺 Medical Diagnosis using Machine Learning

This project focuses on building machine learning models to classify breast cancer as malignant or benign using clinical features. It uses **SVM** and **XGBoost**, along with data preprocessing, model evaluation, and explainability techniques (like SHAP values and Confusion Matrix).

---

## 📂 Project Structure
- Medical Diagnosis using ML/
```
├── data/
│ ├── breast cancer..csv # Original dataset
│ ├── X_train.csv
│ ├── X_test.csv
│ ├── y_train.csv
│ └── y_test.csv
│
├── report/figures/ # Evaluation and explainability visuals
│ ├── Confusion Matrix.png
│ └── SHAP Summary Plot.png
│
├── prepocessing.py # Data cleaning and splitting
├── svm model.py # SVM model training and evaluation
├── xgboost tuned model with evalution explainability.py # Tuned XGBoost with SHAP
├── requirements.txt
├── .gitignore
└── LICENSE
```


---

## 📊 Dataset

- Breast cancer data from the UCI Machine Learning Repository.
- Features include measurements such as radius, texture, perimeter, area, etc.
- Target: `diagnosis` (M = Malignant, B = Benign)

---


## ⚙️ Features

- ✅ Data cleaning and preprocessing
- ✅ Feature scaling using `StandardScaler`
- ✅ Train-test split (80/20)
- ✅ SVM model training and evaluation
- ✅ XGBoost model with hyperparameter tuning
- ✅ Confusion Matrix and SHAP explainability

---

## 🚀 How to Run

### 1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/Medical-Diagnosis-using-ML.git
   cd Medical-Diagnosis-using-ML
   ```

#### 2. Install the dependencies:

```bash
pip install -r requirements.txt
```

### 3. Run the preprocessing script:

```bash
python prepocessing.py
```

### - For SVM:

```bash
python "svm model.py"
```

### - For XGBoost:

```bash
python "xgboost tuned model with evalution explainability.py"
```

📈 Outputs
📌 Confusion Matrix (see ```report/figures/Confusion Matrix.png```)

📌 SHAP Summary Plot (see ```report/figures/SHAP Summary Plot.png```)

🧪 Dependencies
See requirements.txt for full list.

👩‍💻 Author
Tania Akter Rahima
- M.Sc. in Mathematics, Jahangirnagar University

📜 License
- This project is licensed under the MIT License – see the LICENSE file for details.
