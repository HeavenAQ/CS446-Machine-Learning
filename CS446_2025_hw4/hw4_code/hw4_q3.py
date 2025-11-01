"""
CS 446/ECE 449 - Model Selection Homework
==========================================
In this assignment, you will implement model selection using k-fold cross-validation
to find the best hyperparameters for polynomial regression with regularization.

Instructions:
- Complete all TODO sections
- Do not modify the function signatures
- You may add helper functions if needed
"""

from hw4_utils import ModelPipeline, create_polynomial_features

import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
import warnings

warnings.filterwarnings("ignore")


def cross_validate_model(X, y, model, k_folds=5):
    """
    Perform k-fold cross-validation and return average validation error.

    Args:
        X: Training features (n_samples, n_features)
        y: Training labels (n_samples,)
        model: Sklearn model object
        k_folds: Number of folds for cross-validation

    Returns:
        avg_val_error: Average validation MSE across all folds
        std_val_error: Standard deviation of validation MSE across folds
    """
    # TODO: Implement k-fold cross-validation
    # 1. Create KFold() object with k_folds splits (use shuffle=True, random_state=42)
    # 2. For each fold:
    #    - Split data into train and validation sets
    #    - Fit model on training data
    #    - Calculate MSE on validation data
    # 3. Return average and standard deviation of validation errors

    # Remark 1: for `model`, you can safely assume that you can call model.fit(X, y) to
    #   train the model on data X, y; in addition, you can call model.predict(X)
    #   to obtain predictions from the model.
    # Remark 2: for each iteration during k fold validation, please do
    #   `model_clone = deepcopy(model)` and call `model_clone.fit()` and `model_clone.predict()`.
    #   Otherwise, you will be training a model that is from the previous iteration.

    val_errors = []
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for train_idx, validate_idx in kf.split(X):
        # clone the model
        model_clone = deepcopy(model)

        # split
        cur_train_x, cur_validate_x = X[train_idx], X[validate_idx]
        cur_train_y, cur_validate_y = y[train_idx], y[validate_idx]

        # train the model
        model_clone.fit(cur_train_x, cur_train_y)
        y_pred = model_clone.predict(cur_validate_x)
        val_errors.append(np.mean((cur_validate_y - y_pred) ** 2))

    # calculate the error and standard deviation
    avg_val_error = np.mean(val_errors)
    std_val_error = np.std(val_errors)
    return avg_val_error, std_val_error


def evaluate_model(X, y, model):
    avg_err, std_err = cross_validate_model(X, y, model)
    return avg_err, model


def select_best_model(X_train, y_train):
    """
    Select the best model and hyperparameters using cross-validation.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        returned_best_model: Trained best model
    """
    # TODO Implement model selection
    # 1. For each polynomial degree:
    #    a. Create polynomial features for training data (already implemented)
    #    b. Standardize features using StandardScaler (fit on train, transform both) (already implemented)
    #    c. For LinearRegression:
    #       - Perform cross-validation with k = 5
    #    d. For Ridge regression:
    #       - Try different alpha values
    #       - Perform cross-validation for each alpha with k = 5
    #    e. For Lasso regression:
    #       - Try different alpha values
    #       - Perform cross-validation for each alpha with k = 5
    # 2. Select the best model based on lowest cross-validation error

    # Remark 1: you can use `LinearRegression()` to initialize the Linear Regression model.
    # Remark 2: you can use `Ridge(alpha=alpha, random_state=42)` to initialize the Ridge
    #   Regression model.
    # Remark 3: you can use `Lasso(alpha=alpha, random_state=42, max_iter=2000)` to
    #   initialize the Lasso Regression model.

    # Hyperparameter search space (Do not modify these!)
    degrees = [1, 2, 3, 4, 5, 6, 7, 8]
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    best_model = None
    best_degree = None

    cur_best_model = {
        "error": np.inf,
        "model": None,
    }
    for degree in degrees:
        # Create polynomial features
        X_poly = create_polynomial_features(X_train, degree)
        scaler = StandardScaler()
        X_poly_scaled = scaler.fit_transform(X_poly)

        # Models to be evaluated
        all_models = {
            "Linear": [LinearRegression()],
            "Ridge": [Ridge(alpha=a, random_state=42) for a in alphas],
            "Lasso": [Lasso(alpha=a, random_state=42, max_iter=200) for a in alphas],
        }

        # Evaluate all the models
        for name, models in all_models.items():
            print(f"Evaluating {name} Regression")
            results = [
                evaluate_model(X_poly_scaled, y_train, model) for model in models
            ]
            errors = [result[0] for result in results]
            min_idx = np.argmin(errors)

            # update the current best
            if errors[min_idx] < cur_best_model["error"]:
                cur_best_model["error"] = errors[min_idx]
                cur_best_model["model"] = results[min_idx][1]
                best_degree = degree

    best_model = cur_best_model["model"]
    returned_best_model = ModelPipeline(best_degree, best_model, StandardScaler())

    return returned_best_model
