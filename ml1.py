# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Step 2: Prepare the data
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'score': [12, 25, 32, 40, 50, 55, 65, 72, 80, 90]
}

df =pd.DataFrame(data)
 
# Step 3: Create binary target: Pass or Fail
df['pass'] = (df['score'] >= 50).astype(int)

X = df[['Hours_Studied']] # Features
y = df['pass']            # Target variable {0: Fail, 1: Pass}
# Step 4: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)
# Step 6: Make predictions
y_pred = model.predict(X_test)
# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("Predictions:", y_pred)
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
# Step 8: Visualize the results
X_sorted = np.linspace(0, 12, 100).reshape(-1, 1)
y_prob = model.predict_proba(X_sorted)[:, 1]

plt.scatter(X, y, color='blue', label='Actual Pass/Fail')
plt.plot(X_sorted, y_prob * 100, color='red', label='Predicted Probability (Pass)')
plt.xlabel('Hours Studied')
plt.title('Logistic Regression: Pass vs Fail')
plt.ylabel('Probability of Passing')
plt.legend()
plt.show()


