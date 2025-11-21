# EE2211 Machine Learning Library

A comprehensive Python library for linear algebra operations, regression analysis, and optimization algorithms designed for EE2211 coursework.

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

**Parameters:**
- `x`: Input features (n_samples × n_features)
- `y`: Target values (n_samples × n_outputs)
- `x_pred`: Optional prediction points

**Returns:** `(weights, predictions, MSE)`

**Example 1: Simple Linear Regression**
```python
X = np.array([[1, 2], [2, 4], [3, 6], [4, 8]])
Y = np.array([[3], [5], [7], [9]])
w, y_pred, mse = linear_regression(X, Y)
```

**Example 2: With Predictions**
```python
X = np.array([[50, 10], [40, 7], [65, 12], [70, 5], [75, 4]])
Y = np.array([[9, 3], [6, 7], [5, 6], [3, 1], [2, 9]])
w, y_pred, mse = linear_regression(X, Y, x_pred=np.array([[42, 8]]))
```

---

### Polynomial Regression

Performs polynomial regression with optional regularization (Ridge regression).

```python
def polynomial_regression(x, y, degree, x_pred=None, lmbda=0.0, pearson=False)
```

**Parameters:**
- `degree`: Polynomial degree (1 = linear with bias, 2 = quadratic, etc.)
- `lmbda`: Regularization parameter (0 = no regularization)
- `pearson`: Whether to compute Pearson correlation

**Example 1: Quadratic Regression**
```python
X = np.array([[1], [2], [3], [4], [5]])
Y = np.array([[1], [4], [9], [16], [25]])
w, y_pred, mse = polynomial_regression(X, Y, degree=2)
```

**Example 2: With Regularization**
```python
X = np.array([[50, 10], [40, 7], [65, 12], [70, 5], [75, 4]])
Y = np.array([[9, 3], [6, 7], [5, 6], [3, 1], [2, 9]])
w, y_pred, mse = polynomial_regression(X, Y, degree=3, 
                                       x_pred=np.array([[42, 8]]), 
                                       lmbda=0.1)
```

**Example 3: Cubic Polynomial**
```python
X = np.array([[1, 4], [5, -1], [2, 3]])
Y = np.array([[1], [3], [1]])
w, y_pred, mse = polynomial_regression(X, Y, degree=2, lmbda=0)
```

---

### One-Hot Encoding with Classification

Performs classification using one-hot encoding for multi-class problems.

```python
def one_hot_encoder(x, y, degree=1, x_pred=None, lmbda=0)
```

**Parameters:**
- `degree`: Polynomial degree for feature expansion
- `lmbda`: Regularization parameter

**Returns:** `(weights, error_count, predicted_labels)`

**Example 1: Linear Classification**
```python
X = np.array([[1, 3, -2], [-4, 0, -1], [3, 1, 8], [2, 1, 6], [8, 4, 6]])
Y = np.array([[1], [1], [2], [3], [3]])
w, errors, pred = one_hot_encoder(X, Y, degree=1, 
                                   x_pred=np.array([[1, -2, 4]]))
```

**Example 2: Polynomial Classification**
```python
X = np.array([[4], [7], [10], [2], [3], [9]])
Y = np.array([[1], [1], [1], [2], [2], [2]])
w, errors, pred = one_hot_encoder(X, Y, degree=4, 
                                   x_pred=np.array([[6]]))
```

**Example 3: Multi-Feature Classification**
```python
X = np.array([[1, 3, -2], [-4, 0, -1], [3, 1, 8], [2, 1, 6], [8, 4, 6]])
Y = np.array([[1], [1], [2], [3], [3]])
w, errors, pred = one_hot_encoder(X, Y, degree=3, 
                                   x_pred=np.array([[1, -2, 4]]), 
                                   lmbda=0)
```

---

### Matrix Analysis

#### Determinant
```python
def det(x)
```

**Example:**
```python
X = np.array([[1, 2], [3, 4]])
determinant = det(X)
```

---

#### System Determination

Analyzes matrix systems and finds solutions when possible.

```python
def determine(x, y=None)
```

**Example 1: Overdetermined System**
```python
X = np.array([[1, 4], [2, 7], [-3, 11]])
Y = np.array([[1], [-2.5], [4]])
w = determine(X, Y)
```

**Example 2: Matrix Structure Analysis**
```python
X = np.array([[1, 4, 3], [2, -1, 3]])
determine(X)  # Analyzes structure without solving
```

**Example 3: Underdetermined System**
```python
X = np.array([[1, 2, 3], [4, 5, 6]])
Y = np.array([[7], [8]])
w = determine(X, Y)
```

**Example 4: Singular Square Matrix**
```python
X = np.array([[1, 2], [2, 4]])
Y = np.array([[3], [6]])
w = determine(X, Y)  # Checks for one-sided inverses
```

---

#### Reduced Row Echelon Form

```python
def rref(x, y=None)
```

**Example:**
```python
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
rref_matrix, pivots = rref(X)
```

---

### Gradient Descent

Performs gradient descent using PyTorch's automatic differentiation.

```python
def gradient_descent(initial, function, learning_rate=0.01, trials=10)
```

**Parameters:**
- `initial`: Starting point (scalar or array for multivariable)
- `function`: Function to minimize (must use PyTorch operations)
- `learning_rate`: Step size
- `trials`: Number of iterations (0 = show initial gradient only)

**Returns:** Optimized parameter values

**Important:** Use PyTorch functions (`torch.sin`, `torch.cos`, etc.) not NumPy!

---

#### Example 1: Simple Quadratic
```python
def f(x):
    return x**2

result = gradient_descent(initial=5.0, function=f, 
                         learning_rate=0.1, trials=20)
# Finds minimum at x=0
```

---

#### Example 2: Polynomial Function
```python
def f(x):
    return x**4 - 3*x**3 + 2*x

result = gradient_descent(initial=2.0, function=f, 
                         learning_rate=0.01, trials=50)
```

---

#### Example 3: Trigonometric Function
```python
def f(x):
    return torch.sin(x)**2

result = gradient_descent(initial=3.0, function=f, 
                         learning_rate=0.1, trials=30)
```

---

#### Example 4: Exponential Function
```python
def f(x):
    return torch.exp(x**2)

result = gradient_descent(initial=2.0, function=f, 
                         learning_rate=0.01, trials=100)
```

---

#### Example 5: Multivariable Quadratic
```python
def f(x):
    return x[0]**2 + x[1]**2

result = gradient_descent(initial=[3.0, 4.0], function=f, 
                         learning_rate=0.1, trials=50)
# Finds minimum at (0, 0)
```

---

#### Example 6: Multivariable Mixed Terms
```python
def f(x):
    return x[0]**2 + x[0] * x[1]**2

result = gradient_descent(initial=[3.0, 2.0], function=f, 
                         learning_rate=0.05, trials=100)
```

---

#### Example 7: Rosenbrock Function (Challenging)
```python
def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

result = gradient_descent(initial=[-1.0, 1.0], function=rosenbrock, 
                         learning_rate=0.001, trials=1000)
# Finds minimum at (1, 1)
```

---

#### Example 8: Three Variables
```python
def f(x):
    return x[0]**2 + 2*x[1]**2 + 3*x[2]**2

result = gradient_descent(initial=[1.0, 2.0, 3.0], function=f, 
                         learning_rate=0.1, trials=50)
# Finds minimum at (0, 0, 0)
```

---

#### Example 9: Check Initial Gradient Only
```python
def f(x):
    return x**3 - 2*x**2 + x

result = gradient_descent(initial=2.0, function=f, 
                         learning_rate=0.1, trials=0)
# Shows gradient at x=2 without updating
```

---

#### Example 10: Saddle Point
```python
def f(x):
    return x[0]**2 - x[1]**2

result = gradient_descent(initial=[1.0, 1.0], function=f, 
                         learning_rate=0.1, trials=50)
# Converges to saddle point at (0, 0)
```

---

### Correlation Analysis

#### Pearson Correlation (Pairwise)

```python
def pearson_correlation(x, y)
```

**Example:**
```python
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])
r = pearson_correlation(x, y)
```

---

#### Pearson Correlation (Row-wise)

Computes correlation for each row (feature) of X against Y.

```python
def pearson_correlation_rows(X, Y)
```

**Returns:** `(correlations, best_feature_row, best_feature_number, best_correlation)`

**Example:**
```python
X = np.array([[3.3459, 1.0893, 3.2103, 1.744, 1.6762],
              [2.7435, 2.9113, 1.4706, 1.2895, 2.1366],
              [-1.7253, -0.7804, -0.9944, 0.5307, -1.0502]])
Y = np.array([[2.9972, 1.1399, 2.229, 0.3387, 2.5042]])
corrs, best_feat, feat_num, best_corr = pearson_correlation_rows(X, Y)
```

---

### Regression Trees

Builds a simple regression tree with MSE tracking at each depth.

```python
def regression_tree(x, y, initial_threshold=None, max_depth=3)
```

**Parameters:**
- `initial_threshold`: First split point (default: median)
- `max_depth`: Maximum tree depth

**Returns:** `(tree_structure, mse_at_depth)`

**Example 1: Simple Tree**
```python
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
Y = np.array([[2], [3], [5], [7], [11], [13], [17], [19]])
tree, mse_list = regression_tree(X, Y, initial_threshold=4.5, max_depth=2)
```

**Example 2: Real Data**
```python
X = np.array([[0.1], [0.7], [1.6], [2.2], [3.6], [4.1], 
              [4.4], [5.2], [6.2], [7.3]])
Y = np.array([[1.9], [1.5], [5.4], [6.1], [8.9], [9.5], 
              [9.6], [12.9], [13.6], [15.7]])
tree, mse_list = regression_tree(X, Y, initial_threshold=4, max_depth=3)
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

---

## License

For educational use in EE2211 course.

---

## Contact

For questions or issues, refer to course materials or consult with teaching staff.