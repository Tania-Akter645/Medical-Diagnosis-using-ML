# Author Tania Akter Rahima
# XGBoost
#Import Required Libraries


from sklearn.model_selection import RandomizedSearchCV  # for hyperparameter tuning of XGBoost
from xgboost import XGBClassifier # main classification model
from sklearn.metrics import accuracy_score # For Accuracy
from sklearn.model_selection import StratifiedKFold  # Stratified Cross-Validation — ensures class balance in each fold
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import LabelEncoder
import shap




# for Defining Hyperparameter Grid
param_grid = {
    'n_estimators': [10, 20, 30], # 	Number of boosting rounds (trees)
    'max_depth': [3, 4, 5, 6], # Maximum depth of each tree
    'learning_rate': [0.01, 0.05, 0.1],  #Step size shrinkage
    'subsample': [0.6, 0.8, 1.0],  # Fraction of training samples used for each tree
    'colsample_bytree': [0.6, 0.8, 1.0], # Fraction of features (columns) used for each tree
    'gamma': [0, 0.1, 0.2],  # Minimum loss reduction required to make a split
    'reg_alpha': [0, 0.1, 1],  # L1 regularization term (Lasso)
    'reg_lambda': [1, 1.5, 2],  # L2 regularization term
}

# note: L1 regularization  is to prevent overfitting and perform feature selection by penalizing the absolute values of model weights.
# L2 regularization is a technique to prevent overfitting by penalizing large weights in a model.

#=======XGBoost Model Setup=======
xgb = XGBClassifier(
    eval_metric='logloss', # logarithmic loss  Uses  to evaluate performance while training.
    random_state=10 # Ensures reproducibility (give the same results each time.)
)

# ======Cross-Validation Strategy======
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)

#========RandomizedSearchCV Setup===========
grid = RandomizedSearchCV(
    estimator=xgb, # The model to train is XGBClassifier
    param_distributions=param_grid,
    n_iter=20,  #  try 20 random combinations of hyperparameters.
    scoring='accuracy',
    n_jobs=-1, # Uses all CPU cores to speed up computation.
    cv=cv,  # Uses for cross-validation
    verbose=2,  # Displays progress messages during training.
    random_state=10 # For reproducibility.
)

#==========Load & Encode Data===========

#X: All features (everything except "diagnosis" column).
#y: Target variable (whether the tumor is benign or malignant).

df = pd.read_csv("data/breast cancer..csv")
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

#======== Label Encoding ========
le = LabelEncoder()
y_encoded = le.fit_transform(y)

#========== Train-Test Split ============
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=10, stratify=y_encoded
)



os.makedirs("data", exist_ok=True)

#============== Save Train/Test Sets as CSV===============
pd.DataFrame(X_train, columns=X.columns).to_csv("data/X_train.csv", index=False)

pd.DataFrame(X_test, columns=X.columns).to_csv("data/X_test.csv", index=False)

pd.DataFrame(y_train, columns=["y_train"]).to_csv("data/y_train.csv", index=False)

pd.DataFrame(y_test, columns=["y_test"]).to_csv("data/y_test.csv", index=False)


#==========Train Model with Best Params==============
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

#========Evaluation - Accuracy & Classification Report=========
print("🎯 Best Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#==============Confusion Matrix + Save Plot==============

#confusion_matrix(y_test, y_pred): Compares the true labels (y_test) vs predicted labels (y_pred)
# fmt='d': formats annotations as integers.
# cmap='Blues': uses a blue color palette to shade the heatmap.
# sns.heatmap(...): Plots this matrix as a colored grid using the Seaborn library.

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")


os.makedirs('report/figures', exist_ok=True)

plt.savefig('report/figures/Confusion Matrix.png')  #save figure to the report file
plt.show() # display the plot on the screen


#=============SHAP Explainability===========

explainer = shap.Explainer(best_model)
shap_values = explainer(X_test)

os.makedirs("report/figures", exist_ok=True)  # This ensures the SHAP plot image can be saved in that folder later if needed.
shap.summary_plot(shap_values.values, X_test, show=False)


# note: show=False means the plot will not automatically display
# SHAP summary plot shows Feature importance, Impact direction, Feature value color

#======= save figures to the report file=========
plt.savefig("report/figures/SHAP Summary Plot.png", bbox_inches='tight')

plt.show() # display the plot on the screen

shap.plots.beeswarm(shap_values, max_display=10)


#for saving trained XGBoost model to reuse it later for prediction without training again, we can use this
#os.makedirs('models', exist_ok=True)
#import joblib
#joblib.dump(best_model, "models/xgboost_best_model.pkl")

# note: .pkl is a pickle file format  to save trained models.