# Author Tania Akter Rahima
# Importing Libraries


import pandas as pd
import os
from sklearn.model_selection import train_test_split  #  # split data into training and testing sets.
from sklearn.preprocessing import StandardScaler

#===== Load and preprocess data=======
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    if 'id' in df.columns:
        df = df.drop(['id'], axis=1)
        df = df.dropna()
        df.dropna(inplace=True)  #Drop rows with missing values
        df.to_csv("data/breast cancer..csv", index=False) # data collected from open source

# M for Malignant (cancerous)
# B for Benign (non-cancerous)

    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0}) #  line converts categorical labels into numeric format
    return df

# Outside the function: Data cleaning again
df = pd.read_csv("data/breast cancer..csv")



if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True) #  double check to remove id again, in case it’s still there

df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0}) # Again, mapping labels (M→1, B→0) just to make sure it's in numeric form.

#======== Split features and target========
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']


#======= Scale features======
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # apply to training data to compute and apply  the scaling


#========== Train-test split=========
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=10
)

#os.makedirs("data/breast cancer..csv")

#========== Saving Preprocessed Data to CSV =============

#note:index=False means the DataFrame index won’t be saved as a separate column in the CSV file.
# X_train.csv = Features for training
# X_test.csv = 	Features for testing
# y_train.csv = Labels for training
# y_test.csv = Labels for testing
pd.DataFrame(X_train, columns=X.columns).to_csv("data/X_train.csv", index=False)
pd.DataFrame(X_test, columns=X.columns).to_csv("data/X_test.csv", index=False)
pd.DataFrame(y_train, columns=["diagnosis"]).to_csv("data/y_train.csv", index=False)
pd.DataFrame(y_test, columns=["diagnosis"]).to_csv("data/y_test.csv", index=False)

# =====Print Preprocessing Completion======
print("✅ Preprocessing complete. CSV files saved in /data folder.")

#====Main Guard Block=======
if __name__ == "__main__":   # ensures that this block only runs when the script is executed directly
    df = load_and_clean_data('data/breast cancer..csv')


    print(df)  # displays the cleaned DataFrame