import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('emails.csv')

# Remove the first column (index)
df = df.iloc[:, 1:]

# Split the data into features and target
X = df.drop('Prediction', axis=1)
y = df['Prediction']

# Convert text data to numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Naive Bayes classifier
nb = MultinomialNB()

# Define hyperparameter grid for tuning
param_grid = {
    'alpha': [0.1, 1, 10]
}

# Create a GridSearchCV object
grid_search = GridSearchCV(nb, param_grid, cv=5)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Evaluate the model on the testing data
y_pred = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)