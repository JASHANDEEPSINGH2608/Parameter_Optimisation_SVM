import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import NuSVC
from sklearn.metrics import accuracy_score

# Load the digits dataset
dataset = datasets.load_digits()
X = dataset.data
y = dataset.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'nu': [0.1, 0.5, 0.9],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4],
}

# Initialize NuSVC
nusvc = NuSVC()

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=nusvc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Get the best parameters and accuracies
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_

# Predict on the test set
y_pred = grid_search.best_estimator_.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

# Prepare the results for saving
results_df = pd.DataFrame({
    'Best Parameters': [best_params],
    'Best Cross-Validation Accuracy': [best_accuracy],
    'Test Set Accuracy': [test_accuracy]
})

# Save the results to a CSV file
csv_filename = 'svm_optimization_results.csv'
results_df.to_csv(csv_filename, index=False)
print(f"Results saved to {csv_filename}")

# Get the mean test scores for convergence graph
results = grid_search.cv_results_
mean_test_scores = results['mean_test_score']
iterations = range(1, len(mean_test_scores) + 1)

# Plot the convergence graph
plt.plot(iterations, mean_test_scores, marker='o')
plt.title("Convergence Graph of NuSVC Optimization")
plt.xlabel("Iterations")
plt.ylabel("Cross-Validation Accuracy")
plt.grid(True)

# Save the convergence graph as an image
image_filename = 'convergence_graph.png'
plt.savefig(image_filename)
print(f"Convergence graph saved to {image_filename}")

plt.show()
