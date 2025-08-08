# Author Tania Akter Rahima
# Importing Libraries

import pandas as pd  # for data handling.
from sklearn.model_selection import train_test_split # split data into training and testing sets.
from sklearn.svm import SVC  # SVC → Support Vector Classifier
from sklearn.metrics import accuracy_score  # evaluate model performance.
from sklearn.preprocessing import StandardScaler  #StandardScaler → for feature scaling

# Load the data
df = pd.read_csv("data/breast cancer..csv")

# Drop rows with missing values
df.dropna(inplace=True)

# Split into features and target
X = df.drop('diagnosis', axis=1)  # X = all columns except diagnosis (the input features).
y = df['diagnosis']  # y = diagnosis column (the target variable)
# M =  Malignant
# B = Benign


#  Class Distribution
#note: it helps check if the dataset is imbalanced

print("Class counts in full dataset:")
print(y.value_counts())

# Train-test split with stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10, stratify=y  # stratify=y ensures both training and test sets have same class proportions as the original dataset.
)

# Check for NaNs
print("🔎 Checking NaN values...")
print("X_train has NaN:", X_train.isnull().values.any())
print("y_train has NaN:", y_train.isnull().values.any())

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # apply to training data to compute and apply the scaling
X_test_scaled = scaler.transform(X_test)  # transform() is used on test data using the same scaling learned from training.

# Train the SVM model

model = SVC(kernel='linear')
model.fit(X_train_scaled, y_train)

#note: fit() trains the model using the scaled training data and corresponding labels.

# Predict and evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)  # compares the predicted labels (y_pred) with the actual test labels (y_test)
print(f"✅ SVM Accuracy: {accuracy:.2f}")  # prints the accuracy score formatted to 2 decimal places.

