# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define the confusion matrix plotting function
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap="Blues", fmt="d")
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['Did Not Land', 'Landed'])
    ax.yaxis.set_ticklabels(['Did Not Land', 'Landed'])
    plt.show()

# Load the datasets
data_part_2 = pd.read_csv("dataset_part_2.csv")
data_part_3 = pd.read_csv("dataset_part_3.csv")

# Data preparation
numerical_features = ['PayloadMass', 'Flights', 'Block', 'ReusedCount']
scaler = StandardScaler()
X = scaler.fit_transform(data_part_3[numerical_features])
Y = data_part_2['Class'].to_numpy()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Task 6: Hyperparameter tuning for SVM
print("\nTask 6: Hyperparameter Tuning for SVM")
parameters_svm = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10],
    'gamma': [0.1, 1, 10]
}
svm = SVC()
svm_cv = GridSearchCV(estimator=svm, param_grid=parameters_svm, cv=5, scoring='accuracy', verbose=3, n_jobs=-1)
svm_cv.fit(X_train, Y_train)
print("\nTuned hyperparameters (best parameters):", svm_cv.best_params_)
print("Best cross-validation accuracy:", svm_cv.best_score_)

# Task 7: Test Accuracy and Confusion Matrix for SVM
try:
    print("\nTask 7: Test Accuracy and Confusion Matrix for SVM")
    yhat_svm = svm_cv.predict(X_test)
    svm_test_accuracy = svm_cv.score(X_test, Y_test)
    print("\nTest Accuracy of SVM Model:", svm_test_accuracy)
    plot_confusion_matrix(Y_test, yhat_svm)
except Exception as e:
    print("An error occurred in Task 7:", str(e))

# Task 8: Decision Tree Hyperparameter Tuning
try:
    print("\nTask 8: Decision Tree Hyperparameter Tuning")
    parameters_tree = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [2, 4, 6, 8, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    tree = DecisionTreeClassifier()
    tree_cv = GridSearchCV(estimator=tree, param_grid=parameters_tree, cv=10, scoring='accuracy', verbose=3, n_jobs=-1)
    tree_cv.fit(X_train, Y_train)
    print("\nTuned hyperparameters (best parameters):", tree_cv.best_params_)
    print("Best cross-validation accuracy:", tree_cv.best_score_)
    yhat_tree = tree_cv.predict(X_test)
    tree_test_accuracy = tree_cv.score(X_test, Y_test)
    print("\nTest Accuracy of Decision Tree Model:", tree_test_accuracy)
    plot_confusion_matrix(Y_test, yhat_tree)
except Exception as e:
    print("An error occurred in Task 8:", str(e))
# Task 9: Evaluate Decision Tree on Test Data
print("\nTask 9: Evaluate Decision Tree on Test Data")

try:
    # Predict the test data using the best Decision Tree model
    yhat_tree = tree_cv.predict(X_test)

    # Calculate the accuracy on test data
    tree_test_accuracy = tree_cv.score(X_test, Y_test)

    # Print the test accuracy
    print("\nTest Accuracy of Decision Tree Model:", tree_test_accuracy)

    # Plot the confusion matrix
    plot_confusion_matrix(Y_test, yhat_tree)

except Exception as e:
    print("An error occurred in Task 9:", str(e))
from sklearn.neighbors import KNeighborsClassifier

# Task 10: K-Nearest Neighbors Hyperparameter Tuning
print("\nTask 10: K-Nearest Neighbors Hyperparameter Tuning")

try:
    # Define the parameter grid for KNN
    parameters_knn = {
        'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Number of neighbors
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm used for nearest neighbors
        'p': [1, 2]  # Power parameter for Minkowski metric (1=Manhattan, 2=Euclidean)
    }

    # Create a K-Nearest Neighbors object
    knn = KNeighborsClassifier()

    # Create a GridSearchCV object with cv=10
    knn_cv = GridSearchCV(
        estimator=knn, 
        param_grid=parameters_knn, 
        cv=10, 
        scoring='accuracy', 
        verbose=3, 
        n_jobs=-1
    )

    # Fit the GridSearchCV object to the training data
    knn_cv.fit(X_train, Y_train)

    # Output the best parameters and accuracy
    print("\nTuned hyperparameters (best parameters):", knn_cv.best_params_)
    print("Best cross-validation accuracy:", knn_cv.best_score_)

except Exception as e:
    print("An error occurred in Task 10:", str(e))
# Task 11: Evaluate K-Nearest Neighbors on Test Data
print("\nTask 11: Evaluate K-Nearest Neighbors on Test Data")

try:
    # Predict the test data using the best KNN model
    yhat_knn = knn_cv.predict(X_test)

    # Calculate the accuracy on test data
    knn_test_accuracy = knn_cv.score(X_test, Y_test)

    # Print the test accuracy
    print("\nTest Accuracy of KNN Model:", knn_test_accuracy)

    # Plot the confusion matrix
    plot_confusion_matrix(Y_test, yhat_knn)

except Exception as e:
    print("An error occurred in Task 11:", str(e))
# Final Task: Compare Models to Find the Best Performing Method
print("\nFinal Task: Compare Models to Find the Best Performing Method")

# Store test accuracies in a dictionary
model_accuracies = {
    "Support Vector Machine": svm_test_accuracy,
    "Decision Tree": tree_test_accuracy,
    "K-Nearest Neighbors": knn_test_accuracy
}

# Find the best model
best_model = max(model_accuracies, key=model_accuracies.get)
best_accuracy = model_accuracies[best_model]

# Print the results
print("\nModel Test Accuracies:")
for model, accuracy in model_accuracies.items():
    print(f"{model}: {accuracy:.2f}")

print(f"\nBest Performing Model: {best_model} with an accuracy of {best_accuracy:.2f}")
