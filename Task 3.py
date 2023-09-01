'''
ARTIFICIAL INTELLIGENCE INTERNSHIP

TASK-3

Heart Disease Prediction

Here are the simplified steps to perform Heart Disease Prediction:

1. Gather a dataset of labeled heart disease-related features and target variables.

2. Preprocess the data by handling missing values, normalizing features, and splitting it into training and testing sets.

3. Choose and implement a suitable machine learning algorithm, such as logistic regression or random forest, for heart disease prediction. 4.Train the model using the prepared dataset and evaluate its performance using appropriate metrics.

5. Fine-tune the model's hyperparameters to optimize its predictive capabilities.

6. Test the trained model on new, unseen data to assess its accuracy and potential for heart disease prediction.'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('heart.csv')

X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

report = classification_report(y_test, y_pred)
print(report)