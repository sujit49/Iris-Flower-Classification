# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset from the CSV file
df = pd.read_csv('Iris.csv')

# Print the first few rows of the dataframe to ensure it's loaded correctly
print("First five rows of the dataset:")
print(df.head())

# Drop the 'Id' column if it exists
if 'Id' in df.columns:
    df.drop(columns=['Id'], inplace=True)

# Rename columns for convenience
df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]

# Print column names to ensure they are correctly renamed
print("Column names:")
print(df.columns)

# Explore the data
print("Dataset statistics:")
print(df.describe())
print("Species distribution:")
print(df['species'].value_counts())

# Visualize the data
print("Pairplot of the dataset:")
sns.pairplot(df, hue='species')
plt.show()
print("Heatmap of the dataset correlations:")
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Preprocess the data
X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Visualize the confusion matrix
print("Confusion Matrix Heatmap:")
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# Save the model
joblib.dump(knn, 'iris_knn_model.pkl')

# Load the model (optional)
knn_loaded = joblib.load('iris_knn_model.pkl')
print("Predictions using the loaded model:")
print(knn_loaded.predict(X_test[:5]))
# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np

# Ensure the previous imports are still available

# Heatmap of the dataset correlations
print("Heatmap of the dataset correlations:")
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Preprocess the data
X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Visualize the confusion matrix
print("Confusion Matrix Heatmap:")
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save the model
joblib.dump(knn, 'iris_knn_model.pkl')

# Load the model (optional)
knn_loaded = joblib.load('iris_knn_model.pkl')
print("Predictions using the loaded model:")
print(knn_loaded.predict(X_test[:5]))
