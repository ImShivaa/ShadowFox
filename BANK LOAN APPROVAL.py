import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\msiva\OneDrive\Desktop\SHIVA\INTERNSHIPS\SHADOW FOX\PROJECTS\loan_data.csv")

# Drop rows with missing values (or you can choose to fill them)
df.dropna(inplace=True)

# Drop Loan_ID because it's not a predictive feature
df.drop('Loan_ID', axis=1, inplace=True)

# Encode categorical variables
le = LabelEncoder()
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Separate features and target variable
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on test data
predictions = model.predict(X_test)

# Print accuracy and classification report
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, predictions))

# Plot confusion matrix
plt.figure(figsize=(6,4))
cm = pd.crosstab(y_test, predictions, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Random Forest Confusion Matrix\nAccuracy: {accuracy:.2f}")
plt.tight_layout()
plt.show()
