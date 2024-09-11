# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('emails.csv')

# Drop the first column (Email No.)
df = df.drop('Email No.', axis=1)

# Separate features and target
X = df.drop('Prediction', axis=1)  # Features
y = df['Prediction']  # Target (spam or not)

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data (optional but recommended for models sensitive to feature scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Random Forest model
rf = RandomForestClassifier(random_state=42)
# Define a simple logistic regression
log_reg = LogisticRegression(solver='liblinear')

# Define the hyperparameter space for Grid Search
param_grid = {
    'C': [0.01, 0.1, 1, 10],  # Regularization strength
    'penalty': ['l1', 'l2']   # L1 or L2 regularization
}

# Define the hyperparameter space for Random Search
param_dist = {
    'n_estimators': [50, 100],  
    'max_depth': [10, 20],      # Limited depth to reduce computation
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

# Grid Search (Systematic search)
grid_search = GridSearchCV(log_reg, param_grid, cv=3, verbose=1)
grid_search.fit(X_train, y_train)

# Random Search
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=5, cv=2, verbose=1, n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

# Evaluate the best model from Grid Search
best_grid_model = grid_search.best_estimator_
y_pred_grid = best_grid_model.predict(X_test)
print("Grid Search Best Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_grid))
print(classification_report(y_test, y_pred_grid))

# Evaluate the best model from Random Search
best_random_model = random_search.best_estimator_
y_pred_random = best_random_model.predict(X_test)
print("Random Search Best Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_random))
print(classification_report(y_test, y_pred_random))


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Predictions on the test data
y_pred = random_search.predict(X_test)

# Accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Display Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

print("=== Model Performance Evaluation ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nBest Hyperparameters:", random_search.best_params_)
print("\nClassification Report:\n", classification_report(y_test, y_pred))