# My EE2211 Library

A sample Python library for linear algebra operations, regression analysis, and optimization algorithms designed for EE2211 coursework.

## Table of Contents
- [Installation](#installation)
- [Features](#features)
- [Function Reference](#function-reference)
  - [Linear Regression](#linear-regression)
  - [Polynomial Regression](#polynomial-regression)
  - [One-Hot Encoding with Classification](#one-hot-encoding-with-classification)
  - [Matrix Analysis](#matrix-analysis)
  - [Gradient Descent](#gradient-descent)
  - [Correlation Analysis](#correlation-analysis)
  - [Regression Trees](#regression-trees)
- [Common Patterns](#common-patterns)
- [Tips and Best Practices](#tips-and-best-practices)
- [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites
```bash
pip install numpy matplotlib pandas scikit-learn sympy torch
```

### Setup
1. Clone or download the repository
2. Place `ee2211_lib.py` in your project directory
3. Import the library:
```python
from ee2211_lib import *
```

---

## Features

- **Linear & Polynomial Regression** with automatic system determination (under/over/well-determined)
- **Classification** using one-hot encoding
- **Matrix Operations** including determinants, inverses, and RREF
- **Gradient Descent** optimization with automatic differentiation (PyTorch)
- **Pearson Correlation** analysis
- **Regression Trees** with MSE tracking

---

## Function Reference

### Linear Regression

Performs linear regression **without bias term**. For bias, use `polynomial_regression` with `degree=1`.

```python
def linear_regression(x, y, x_pred=None)
```

**Returns:** `(weights, predictions, MSE)`

**Use Case 1: Overdetermined System (Least Squares)**
```python
X = np.array([[1, 2], [2, 4], [3, 6], [4, 8]])
Y = np.array([[3], [5], [7], [9]])
w, y_pred, mse = linear_regression(X, Y)
# Output: Weights via left inverse, MSE, y_pred = None
```

**Use Case 2: Underdetermined System (Minimum Norm)**
```python
X = np.array([[1, 2, 3, 4]])
Y = np.array([[10]])
w, y_pred, mse = linear_regression(X, Y)
# Output: Weights via right inverse, MSE = 0.0, y_pred = None
```

**Use Case 3: Well-Determined System with Predictions**
```python
X = np.array([[1, 2], [3, 4]])
Y = np.array([[5], [11]])
X_pred = np.array([[5, 6]])
w, y_pred, mse = linear_regression(X, Y, x_pred=X_pred)
# Output: Exact inverse weights, MSE = 0.0, y_pred = [[17.]]
```

**Use Case 4: Singular Matrix Handling**
```python
X = np.array([[1, 2], [2, 4]]) # Linearly dependent columns
Y = np.array([[3], [6]])
w, y_pred, mse = linear_regression(X, Y)
# Output: Prints "Matrix is singular, cannot compute inverse.", returns (None, None, None)
```

---

### Polynomial Regression

Performs polynomial regression with optional regularization (Ridge regression).

```python
def polynomial_regression(x, y, degree, x_pred=None, lmbda=0.0, pearson=False)
```

**Returns:** `(weights, predictions, MSE)`

**Use Case 1: Overdetermined (Primal Mode)**
```python
X = np.array([[1], [2], [3], [4], [5]])
Y = np.array([[1], [4], [9], [16], [25]])
w, y_pred, mse = polynomial_regression(X, Y, degree=2)
# Output: Solves using X_poly.T @ X_poly (primal mode), exact fit MSE ~ 0
```

**Use Case 2: Underdetermined (Dual Mode with Regularization)**
```python
X = np.array([[1], [2]])
Y = np.array([[1], [0]])
w, y_pred, mse = polynomial_regression(X, Y, degree=5, lmbda=0.1)
# Output: System has more features (6) than samples (2). Solves using dual mode with Ridge penalty.
```

**Use Case 3: Pearson Correlation Output**
```python
X = np.array([[50, 10], [40, 7], [65, 12], [70, 5], [75, 4]])
Y = np.array([[9, 3], [6, 7], [5, 6], [3, 1], [2, 9]])
w, y_pred, mse = polynomial_regression(X, Y, degree=2, pearson=True)
# Output: Prints Pearson correlation coefficient for output dimension 0 and 1, returns standard tuple.
```

---

### One-Hot Encoding with Classification

Performs classification using one-hot encoding for multi-class problems.

```python
def one_hot_encoder(x, y, degree=1, x_pred=None, lmbda=0)
```

**Returns:** `(weights, error_count, predicted_labels)`

**Use Case 1: Well-Determined System with Exact Classification**
```python
X = np.array([[1, 0], [0, 1], [-1, 0]])
Y = np.array([['A'], ['B'], ['C']]) # 3 samples, transforms to 3 classes (3 columns)
w, errors, pred = one_hot_encoder(X, Y, degree=1)
# Output: Solves exactly, error_count = 0
```

**Use Case 2: Overdetermined System with Misclassifications**
```python
X = np.array([[1], [2], [1.5], [8], [9], [8.5]])
Y = np.array([[1], [1], [2], [3], [3], [1]]) # Deliberate overlap/noise
w, errors, pred = one_hot_encoder(X, Y, degree=1)
# Output: Solves via least squares, prints "Number of misclassifications: X out of 6"
```

**Use Case 3: Underdetermined System with Predictions**
```python
X = np.array([[1], [2]])
Y = np.array([['cat'], ['dog']])
w, errors, pred = one_hot_encoder(X, Y, degree=4, x_pred=np.array([[1.5]]))
# Output: Solves via dual mode, pred = array(['cat'] or ['dog']) based on closest fit
```

---

### Matrix Analysis

#### Determinant
```python
def det(x)
```

**Use Case 1: Invertible Matrix**
```python
X = np.array([[1, 2], [3, 4]])
determinant = det(X)
# Output: -2.0
```

**Use Case 2: Singular Matrix**
```python
X = np.array([[1, 1], [1, 1]])
determinant = det(X)
# Output: 0.0
```

**Use Case 3: Non-Square Matrix**
```python
X = np.array([[1, 2, 3], [4, 5, 6]])
determinant = det(X)
# Output: Prints "Matrix is not square, determinant not defined.", returns None
```

---

#### System Determination

Analyzes matrix systems and finds solutions when possible.

```python
def determine(x, y=None)
```

**Use Case 1: Overdetermined System (Least Squares)**
```python
X = np.array([[1, 4], [2, 7], [-3, 11]])
Y = np.array([[1], [-2.5], [4]])
w = determine(X, Y)
# Output: Prints left inverse existence, returns least squares w, prints residual
```

**Use Case 2: Underdetermined System (Minimum Norm)**
```python
X = np.array([[1, 2, 3], [4, 5, 6]])
Y = np.array([[7], [8]])
w = determine(X, Y)
# Output: Prints right inverse existence, returns w
```

**Use Case 3: Singular Square Matrix with One-Sided Check**
```python
X = np.array([[1, 2], [2, 4]])
Y = np.array([[3], [6]])
w = determine(X, Y)
# Output: Fails direct inverse, falls back to checking left/right inverses.
```

**Use Case 4: Pure Structure Analysis (No Y)**
```python
X = np.array([[1, 4, 3], [2, -1, 3]])
determine(X) 
# Output: Prints "Matrix is UNDERDETERMINED", checks right inverse, returns None
```

---

#### Reduced Row Echelon Form

```python
def rref(x, y=None)
```

**Use Case 1: Standard Matrix RREF**
```python
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
rref_matrix, pivots = rref(X)
# Output: rref_matrix of X, pivots = (0, 1)
```

**Use Case 2: Augmented Matrix Solving**
```python
X = np.array([[1, 2], [3, 4]])
Y = np.array([[5], [11]])
rref_matrix, pivots = rref(X, Y)
# Output: Appends Y to X. rref_matrix shows identity on left, solution on right column.
```

---

### Gradient Descent

Performs gradient descent using PyTorch's automatic differentiation.

```python
def gradient_descent(initial, function, learning_rate=0.01, trials=10)
```

**Important:** Use PyTorch functions (`torch.sin`, `torch.cos`, etc.) not NumPy!

**Use Case 1: Multivariable Quadratic (Convergence)**
```python
def f(x):
    return x[0]**2 + x[1]**2

result = gradient_descent(initial=[3.0, 4.0], function=f, learning_rate=0.1, trials=50)
# Output: Iterates 50 times, finds minimum near array([0., 0.])
```

**Use Case 2: Rosenbrock Function (Challenging Landscape)**
```python
def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

result = gradient_descent(initial=[-1.0, 1.0], function=rosenbrock, learning_rate=0.001, trials=1000)
# Output: Carefully navigates the valley, converges toward array([1., 1.])
```

**Use Case 3: Initial Gradient Inspection Only**
```python
def f(x):
    return x**3 - 2*x**2 + x

result = gradient_descent(initial=2.0, function=f, learning_rate=0.1, trials=0)
# Output: Prints initial gradient at x=2, performs 0 trials, returns 2.0
```

---

### Correlation Analysis

#### Pearson Correlation (Pairwise)

```python
def pearson_correlation(x, y)
```

**Use Case 1: Strong Correlation**
```python
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])
r = pearson_correlation(x, y)
# Output: 1.0
```

**Use Case 2: Zero Variance Error Protection**
```python
x = np.array([5, 5, 5, 5])
y = np.array([1, 2, 3, 4])
r = pearson_correlation(x, y)
# Output: Prints "Error: Standard deviation is zero", returns None
```

---

#### Pearson Correlation (Row-wise)

Computes correlation for each row (feature) of X against Y.

```python
def pearson_correlation_rows(X, Y)
```

**Returns:** `(correlations, best_feature_row, best_feature_number, best_correlation)`

**Use Case 1: Finding the Best Feature**
```python
X = np.array([[3.3, 1.0, 3.2], [2.7, 2.9, 1.4], [-1.7, -0.7, -0.9]])
Y = np.array([[3.0, 1.1, 2.2]])
corrs, best_feat, feat_num, best_corr = pearson_correlation_rows(X, Y)
# Output: Evaluates 3 features, returns array of 3 correlations, isolates the highest absolute correlation.
```

**Use Case 2: Handling Dead Features (Zero Variance Row)**
```python
X = np.array([[1.0, 1.0, 1.0], [2.0, 4.0, 6.0]])
Y = np.array([[1.0, 2.0, 3.0]])
corrs, best_feat, feat_num, best_corr = pearson_correlation_rows(X, Y)
# Output: Row 0 gets np.nan. Identifies Row 1 as best feature.
```

---

### Regression Trees

Builds a simple regression tree with MSE tracking at each depth.

```python
def regression_tree(x, y, initial_threshold=None, max_depth=3)
```

**Returns:** `(tree_structure, mse_at_depth)`

**Use Case 1: Standard Tree Generation**
```python
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
Y = np.array([[2], [3], [5], [7], [11], [13], [17], [19]])
tree, mse_list = regression_tree(X, Y, initial_threshold=4.5, max_depth=2)
# Output: Returns nested dictionary tree, and mse_list [mse_depth0, mse_depth1, mse_depth2]
```

**Use Case 2: Unequal Array Length Safeguard**
```python
X = np.array([1, 2, 3])
Y = np.array([1, 2, 3, 4])
tree, mse_list = regression_tree(X, Y)
# Output: Prints "Error: x and y must have same length", returns (None, None)
```

---

## Common Patterns

### Pattern 1: Full Pipeline with Predictions
```python
# Load data
X_train = np.array([[1, 2], [3, 4], [5, 6]])
Y_train = np.array([[2], [4], [6]])
X_test = np.array([[7, 8]])

# Train model
w, _, mse = polynomial_regression(X_train, Y_train, degree=2)

# Make predictions
_, y_pred, _ = polynomial_regression(X_train, Y_train, degree=2, 
                                     x_pred=X_test)
```

### Pattern 2: System Analysis Before Solving
```python
X = np.array([[1, 2], [3, 4], [5, 6]])
Y = np.array([[1], [2], [3]])

# Analyze system
determine(X)  # Check if over/under/well-determined

# Get RREF
rref_matrix, pivots = rref(X, Y)

# Solve system
w = determine(X, Y)
```

### Pattern 3: Feature Selection with Correlation
```python
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
Y = np.array([[2, 3, 4]])

# Find best feature
corrs, best_feat, feat_num, _ = pearson_correlation_rows(X, Y)

# Use only best feature for regression
X_best = best_feat.reshape(-1, 1)
w, _, mse = linear_regression(X_best, Y.T)
```

---

## Tips and Best Practices

1. **Regularization**: Use `lmbda > 0` when you have multicollinearity or overfitting
2. **Gradient Descent**: Start with small learning rates (0.001-0.1) and increase if convergence is slow
3. **PyTorch Functions**: Always use `torch.*` functions in gradient descent, not `np.*`
4. **System Determination**: Check system type with `determine()` before solving
5. **Classification**: Use higher polynomial degrees cautiously to avoid overfitting

---

## Troubleshooting

**Problem:** "Matrix is singular, cannot compute inverse"
- **Solution:** Try adding regularization (`lmbda > 0`) or check if your data is linearly dependent

**Problem:** Gradient descent diverges (values become very large)
- **Solution:** Reduce learning rate or check your function definition

**Problem:** "Operator @ not supported for types spmatrix"
- **Solution:** The library handles this automatically, but ensure you're using the latest version

**Problem:** Poor classification accuracy
- **Solution:** Try increasing polynomial degree or adding more training data