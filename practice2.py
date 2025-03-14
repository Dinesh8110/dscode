import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load and Explore the Dataset
df = pd.read_csv("DataSet1/Iris.csv")
print("Dataset Head:\n", df.head())
print("\nSummary Statistics:\n", df.describe())
print("\nMissing Values:\n", df.isnull().sum())
print("\nClass Distribution:\n", df["Species"].value_counts())

# Step 2: Data Preprocessing
# Drop 'Id' column since it's not needed
df.drop(columns=["Id"], inplace=True)

# Handle missing values (if any)
df.fillna(df.median(numeric_only=True), inplace=True)

# Split features and labels
X = df.drop(columns=["Species"])
y = df["Species"]

# Encode target labels
y = y.astype("category").cat.codes  # Convert species names to numerical values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Model Selection and Training
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 4: Model Evaluation
predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("\nModel Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, predictions))

# Step 5: Data Visualization
plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Feature Importance Plot
feature_importance = pd.Series(clf.feature_importances_, index=X.columns)
feature_importance.nlargest(4).plot(kind='barh', color='skyblue')
plt.title("Feature Importance")
plt.show()
