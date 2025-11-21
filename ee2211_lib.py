import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sympy import Matrix
import torch

def linear_regression(x, y, x_pred = None):
    """
    Performs linear regression on the given data but doesn't add a bias term. If bias is needed, use polynomial_regression with degree 1.
    """
    # Check if over, under or well determined
    
    # More columns than rows, hence underdetermined
    if x.shape[0] < x.shape[1]:
        try:
            w = x.T @ np.linalg.inv(x @ x.T) @ y
            print("Weights: \n", w)
        except np.linalg.LinAlgError:
            print("Matrix is singular, cannot compute inverse.")
            return None, None, None
    
    # More rows than columns, hence overdetermined
    elif x.shape[0] > x.shape[1]:
        try:
            w = np.linalg.inv(x.T @ x) @ x.T @ y
            print("Weights: \n", w)
        except np.linalg.LinAlgError:
            print("Matrix is singular, cannot compute inverse.")
            return None, None, None
    else:
        try:
            w = np.linalg.lstsq(x, y, rcond=None)[0]
            print("Weights: \n", w)
        except np.linalg.LinAlgError:
            print("Matrix is singular, cannot compute inverse.")
            return None, None, None
        
    y_test = x @ w
    mse_per_col = np.mean((y - y_test) ** 2, axis=0)
    MSE = mse_per_col if len(mse_per_col) > 1 else mse_per_col.item()
    print(f"MSE: {MSE}\n")
    
    if x_pred is not None:
        y_pred = x_pred @ w
        for i in range(len(x_pred)):
            print(f"Prediction for y when x = {x_pred[i]} : {y_pred[i]}\n")
        return w, y_pred, MSE
    else:
        return w, None, MSE
    
def polynomial_regression(x, y, degree, x_pred=None, lmbda=0.0, pearson=False):    
    """
    Performs polynomial regression on the given data. If degree=1, it is equivalent to linear regression with bias term.
    """
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(x)
    # Check if X_poly is over, under or well determined
    
    # More columns than rows, hence underdetermined
    if X_poly.shape[0] < X_poly.shape[1]:
        try:
            w = X_poly.T @ np.linalg.inv(X_poly @ X_poly.T + lmbda * np.eye(X_poly.shape[0])) @ y
            print(f"System is underdetermined, used dual mode, weights: \n{w}\n")
        except np.linalg.LinAlgError:
            print("Matrix is singular, cannot compute inverse.")
            return None, None, None
    
    # More rows than columns, hence overdetermined
    elif X_poly.shape[0] > X_poly.shape[1]:
        try:
            w = np.linalg.inv(X_poly.T @ X_poly + lmbda * np.eye(X_poly.shape[1])) @ X_poly.T @ y
            print(f"System is overdetermined, used primal mode, weights: \n{w}\n")
        except np.linalg.LinAlgError:
            print("Matrix is singular, cannot compute inverse.")
            return None, None, None
    else:
        try:
            w = np.linalg.lstsq(X_poly, y, rcond=None)[0]
            print(f"System is well-determined, used lstsq, weights: \n{w}\n")
        except np.linalg.LinAlgError:
            print("Matrix is singular, cannot compute inverse.")
            return None, None, None
    
    y_test = X_poly @ w
    mse_per_col = np.mean((y - y_test) ** 2, axis=0)
    MSE = mse_per_col if len(mse_per_col) > 1 else mse_per_col.item()
    if x_pred is not None:
        X_pred_poly = poly.transform(x_pred)
        y_pred = X_pred_poly @ w
        for i in range(len(x_pred)):
            print(f"Prediction for y when x = {x_pred[i]} : {y_pred[i]}\n")
        print(f"MSE: {MSE}\n")
        return w, y_pred, MSE
    
    if pearson:
        # Calculate Pearson correlation coefficient for each output dimension
        for i in range(y.shape[1]):
            correlation_matrix = np.corrcoef(y[:, i], y_test[:, i])
            pearson_corr = correlation_matrix[0, 1]
            print(f"Pearson correlation coefficient for output dimension {i}: {pearson_corr}\n")
    
    return w, None, MSE



def one_hot_encoder(x, y, degree = 1, x_pred = None, lmbda = 0):
    """
    One hot encodes the output labels and performs linear or polynomial regression based on the degree parameter.
    """
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y)
    print(f"One hot encoded labels: \n{y_encoded}\n")
    
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(x)
    
    # Check if overdetermined, underdetermined or well-determined
    # More columns than rows, hence underdetermined
    if x.shape[0] < x.shape[1]:
        try:
            w = X_poly.T @ np.linalg.inv(X_poly @ X_poly.T + lmbda * np.eye(X_poly.shape[0])) @ y_encoded
            print("Underdetermined system")
        except np.linalg.LinAlgError:
            print("Matrix is singular, cannot compute inverse.")
            return None, None, None
    
    # More rows than columns, hence overdetermined
    elif x.shape[0] > x.shape[1]:
        try:
            w = np.linalg.inv(X_poly.T @ X_poly + lmbda * np.eye(X_poly.shape[1])) @ X_poly.T @ y_encoded
            print("Overdetermined system")
        except np.linalg.LinAlgError:
            print("Matrix is singular, cannot compute inverse.")
            return None, None, None
    
    else:
        try:
            w = np.linalg.lstsq(X_poly, y_encoded, rcond=None)[0]
            print("Well-determined system")
        except np.linalg.LinAlgError:
            print("Matrix is singular, cannot compute inverse.")
            return None, None, None
    
    # Calculate classs classification error
    y_test = X_poly @ w
    y_pred_classes = np.argmax(y_test, axis=1)
    y_true_classes = np.argmax(y_encoded, axis=1)
    error_count = np.sum(y_pred_classes != y_true_classes)
    print(f"Number of misclassifications on training data: {error_count} out of {len(y)}\n")
    
    if x_pred is not None:
        X_pred_poly = poly.transform(x_pred)
        # Convert sparse matrix to dense if needed
        if hasattr(X_pred_poly, 'toarray'):
            X_pred_poly = X_pred_poly.toarray()
            
        y_pred = X_pred_poly @ w
        print(f"Predicted one-hot outputs for x_pred: \n{y_pred}\n")
        y_pred_indices = np.argmax(y_pred, axis=1).reshape(-1, 1)
        
        # Create one-hot encoding for predicted indices
        y_pred_onehot = np.zeros((y_pred_indices.shape[0], y_encoded.shape[1]))
        y_pred_onehot[np.arange(y_pred_indices.shape[0]), y_pred_indices.flatten()] = 1
        
        # Convert back to original labels
        y_pred_labels = encoder.inverse_transform(y_pred_onehot)
        
        for i in range(len(x_pred)):
            print(f"Class prediction when x = {x_pred[i]} -> class {y_pred_labels[i][0]}\n")
        return w, error_count, y_pred_labels.flatten()

    return w, error_count, None


def det(x):
    try:
        det = np.linalg.det(x)
        print(f"Determinant: {det}\n")
        return det
    except np.linalg.LinAlgError:
        if x.shape[0] != x.shape[1]:
            print("Matrix is not square, determinant not defined.")
        print("Matrix is singular, cannot compute determinant.")
        return None

def old_determine(x, y=None):
    # Determines if a matrix is under, over or well determined
    # Tells you left, right inverse
    m, n = x.shape

    if y is not None:
        if m < n:
            # Check if system can actually be solved
            # Find right inverse
            try:
                w = x.T @ np.linalg.inv(x @ x.T) @ y
                y_approx = x @ w
                print(f"Underdetermined, more columns than rows, right inverse exists: \n{w}\n")
                print(f"y approximated: \n{y_approx}\n")
            except np.linalg.LinAlgError:
                print("Underdetermined, no right inverse")
        elif m > n:
            try:
                w = np.linalg.inv(x.T @ x) @ x.T @ y
                y_approx = x @ w
                print(f"Overdetermined, more rows than columns, left inverse exists: \n{w}\n")
                print(f"y approximated: \n{y_approx}\n")
            except np.linalg.LinAlgError:
                print("Overdetermined, no left inverse")
        else:
            try:
                w = np.linalg.inv(x) @ y
                print(f"Well Determined, use inverse: w = A^-1 * b: \n{w}\n")
            except np.linalg.LinAlgError:
                print("Well Determined, no inverse exists")
                
    else:
        if m > n:
            try:
                np.linalg.inv(x.T @ x)
                print("Left inverse exists\n")
            except np.linalg.LinAlgError:
                print("Left inverse does not exist\n")
        elif m < n:
            try:
                np.linalg.inv(x @ x.T)
                print("Right inverse exists\n")
            except np.linalg.LinAlgError:
                print("Right inverse does not exist\n")
        else:
            try:
                np.linalg.inv(x)
                print("Inverse exists\n")
            except np.linalg.LinAlgError:
                print("Inverse does not exist\n")
                
def determine(x, y=None):
    """
    Determines if a matrix is under, over, or well determined.
    Checks for left/right inverses and solves the system if y is provided.
    
    For square matrices that are non-invertible, also checks for one-sided inverses.
    
    Parameters:
    - x: Input matrix (numpy array)
    - y: Optional target vector for solving Ax = y
    
    Returns:
    - w: Solution vector if y is provided, None otherwise
    """
    m, n = x.shape
    
    # Helper function to check matrix invertibility
    def check_inverse(matrix, name):
        try:
            np.linalg.inv(matrix)
            print(f"{name} inverse exists")
            return True
        except np.linalg.LinAlgError:
            print(f"{name} inverse does not exist")
            return False
    
    # Case 1: System with target vector y
    if y is not None:
        print(f"Matrix shape: {m} rows × {n} columns\n")
        
        if m < n:
            # Underdetermined: more unknowns than equations
            print("System is UNDERDETERMINED (more columns than rows)")
            try:
                w = x.T @ np.linalg.inv(x @ x.T) @ y
                y_approx = x @ w
                print(f"Right inverse exists, solution found:\nw = \n{w}\n")
                print(f"Verification: x @ w = \n{y_approx}\n")
                return w
            except np.linalg.LinAlgError:
                print("Right inverse does not exist, no exact solution\n")
                return None
                
        elif m > n:
            # Overdetermined: more equations than unknowns
            print("System is OVERDETERMINED (more rows than columns)")
            try:
                w = np.linalg.inv(x.T @ x) @ x.T @ y
                y_approx = x @ w
                residual = np.linalg.norm(y - y_approx)
                print(f"Left inverse exists, least squares solution found:\nw = \n{w}\n")
                print(f"Approximation: x @ w = \n{y_approx}\n")
                print(f"Residual ||y - x@w||: {residual:.6f}\n")
                return w
            except np.linalg.LinAlgError:
                print("Left inverse does not exist, no least squares solution\n")
                return None
                
        else:
            # Square matrix: well-determined
            print("System is WELL-DETERMINED (square matrix)")
            try:
                w = np.linalg.inv(x) @ y
                print(f"Matrix is invertible, exact solution:\nw = \n{w}\n")
                print(f"Verification: x @ w = \n{x @ w}\n")
                return w
            except np.linalg.LinAlgError:
                print("Matrix is singular (not invertible)")
                print("Checking for one-sided inverses...\n")
                
                # Check for left inverse
                left_exists = check_inverse(x.T @ x, "Left")
                if left_exists:
                    try:
                        w = np.linalg.inv(x.T @ x) @ x.T @ y
                        print(f"Least squares solution using left inverse:\nw = \n{w}\n")
                        return w
                    except:
                        pass
                
                # Check for right inverse
                right_exists = check_inverse(x @ x.T, "Right")
                if right_exists:
                    try:
                        w = x.T @ np.linalg.inv(x @ x.T) @ y
                        print(f"Minimum norm solution using right inverse:\nw = \n{w}\n")
                        return w
                    except:
                        pass
                
                print("No solution found via standard methods\n")
                return None
    
    # Case 2: No target vector, just analyze the matrix structure
    else:
        print(f"Analyzing matrix structure: {m} rows × {n} columns\n")
        
        if m < n:
            # Underdetermined
            print("Matrix is UNDERDETERMINED (more columns than rows)")
            check_inverse(x @ x.T, "Right")
            print()
            
        elif m > n:
            # Overdetermined
            print("Matrix is OVERDETERMINED (more rows than columns)")
            check_inverse(x.T @ x, "Left")
            print()
            
        else:
            # Square matrix
            print("Matrix is SQUARE (well-determined)")
            is_invertible = check_inverse(x, "Matrix")
            
            if not is_invertible:
                print("\nMatrix is singular, checking one-sided inverses:")
                check_inverse(x.T @ x, "Left")
                check_inverse(x @ x.T, "Right")
            print()
        
        return None

def rref(x, y = None):
    if y is not None:
        A = np.hstack([x, y.reshape(-1, 1)])
    else:
        A = x
    M = Matrix(A)
    rref_matrix, pivot_cols = M.rref()
    print(f"RREF:\n{rref_matrix}\n")
    print(f"Pivot columns: {pivot_cols}\n")
    return np.array(rref_matrix).astype(np.float64), pivot_cols

def gradient_descent(initial, function, learning_rate=0.01, trials=10):
    """
    Performs gradient descent to minimize the mean squared error between predictions and actual values.
    
    Parameters:
    - initial: Initial guess for the variable to be optimized. Can be a float or a list/array for multivariable optimization.
    - function: A callable that takes the variable as input and returns the value to be minimized
    - learning_rate: Step size for each iteration of gradient descent.
    - trials: Number of iterations to perform. If trials=0, returns initial point with its gradient computed.
    
    Returns:
    - The optimized function value after the specified number of trials.
    
    Example usage:
    # Example 1: Simple quadratic function f(x) = x^2
    def f(x):
        return x**2

    result = gradient_descent(initial=5.0, function=f, learning_rate=0.1, trials=20)

    # Example 2: Multivariable function f(x,y) = x^2 + y^2
    def g(x):
        return torch.sum(x**2)

    result = gradient_descent(initial=[3.0, 4.0], function=g, learning_rate=0.1, trials=20)
    """
    x = torch.tensor(initial, dtype=torch.float32, requires_grad=True)
    
    # Calculate initial function value and gradient
    y = function(x)
    y.backward()
    
    print(f"Initial value: x = {x.detach().numpy()}, f(x) = {y.item():.6f}")
    if x.grad is not None:
        print(f"Initial gradient: {x.grad.detach().numpy()}\n")
    
    # If trials=0, return initial value
    if trials == 0:
        result = x.detach().numpy()
        return result.item() if result.shape == () else result
    
    # Perform gradient descent for the specified number of trials
    for i in range(trials):
        if x.grad is not None:
            with torch.no_grad():
                x -= learning_rate * x.grad
            
            # Zero gradient before next computation
            x.grad.zero_()
            
            # Compute new function value and gradient
            y = function(x)
            y.backward()
            
            print(f"Trial {i+1}: x = {x.detach().numpy()}, f(x) = {y.item():.6f}, gradient = {x.grad.detach().numpy()}")
        else:
            print(f"Trial {i+1}: No gradient computed.")
            break
    
    print(f"\nFinal value: {x.detach().numpy()}\n")
    
    result = x.detach().numpy()
    return result.item() if result.shape == () else result

def fallback_gradient_descent(learning_rate=0.01, trials=10):
    """
    A fallback gradient descent implementation where you manually define the function and gradient.
    """
    a = 1.0 # Replace with initial value
    
    a_out = np.zeros(trials)
    f1_out = np.zeros(trials)
    
    for i in range(trials):
        f1 = a**2 + 4*a + 4  # Example function: f(a) = a^2 + 4a + 4
        grad = 2*a + 4       # Gradient: f'(a) = 2a + 4
        
        a = a - learning_rate * grad
        
        a_out[i] = a
        f1_out[i] = f1
        
        print(f"Trial {i+1}: a = {a}, f(a) = {f1}")
    
    return a_out, f1_out

def pearson_correlation(x, y):
    """
    Calculates the Pearson correlation coefficient between two variables.
    
    Parameters:
    - x: First variable (1D numpy array or list)
    - y: Second variable (1D numpy array or list)
    
    Returns:
    - r: Pearson correlation coefficient (between -1 and 1)
    
    Interpretation:
    - r = 1: Perfect positive correlation
    - r = 0: No correlation
    - r = -1: Perfect negative correlation
    
    Example usage:
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    r = pearson_correlation(x, y)  # Should be close to 1.0
    """
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    
    if len(x) != len(y):
        print("Error: x and y must have the same length")
        return None
    
    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate numerator: sum of (x - x_mean) * (y - y_mean)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    
    # Calculate denominator: sqrt(sum of (x - x_mean)^2) * sqrt(sum of (y - y_mean)^2)
    denominator = np.sqrt(np.sum((x - x_mean)**2)) * np.sqrt(np.sum((y - y_mean)**2))
    
    if denominator == 0:
        print("Error: Standard deviation is zero, correlation undefined")
        return None
    
    r = numerator / denominator
    
    print(f"Pearson correlation coefficient: {r:.6f}\n")
    
    return r

def pearson_correlation_rows(X, Y):
    """
    Computes Pearson correlation between each row of X and a 1D array Y.
    Returns:
      correlations: 1D array of Pearson coefficients per row
      best_feature_row: the row (feature vector) with strongest absolute correlation
      best_feature_number: 1-based feature number
      best_correlation: correlation value
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).squeeze()
    if X.ndim != 2 or Y.ndim != 1 or X.shape[1] != Y.shape[0]:
        print("Shape mismatch.")
        return None, None, None, None

    y_mean = Y.mean()
    y_var = np.sum((Y - y_mean) ** 2)
    if y_var == 0:
        print("Y variance zero.")
        return None, None, None, None

    corrs = []
    for row in X:
        x_mean = row.mean()
        x_var = np.sum((row - x_mean) ** 2)
        if x_var == 0:
            corrs.append(np.nan)
            continue
        num = np.sum((row - x_mean) * (Y - y_mean))
        corrs.append(num / np.sqrt(x_var * y_var))

    corrs = np.array(corrs)
    print(f"Row-wise Pearson correlations:\n{corrs}\n")

    best_idx = int(np.nanargmax(np.abs(corrs)))
    best_feature_number = best_idx + 1  # 1-based
    best_feature_row = X[best_idx]
    best_corr = corrs[best_idx]

    print(f"Best feature: Feature {best_feature_number} (values: {best_feature_row}), correlation = {best_corr:.6f}\n")

    return corrs, best_feature_row, best_feature_number, best_corr

def regression_tree(x, y, initial_threshold=None, max_depth=3):
    """
    Performs simple regression tree by recursively splitting data based on thresholds.
    
    Parameters:
    - x: Input features (1D or 2D numpy array). If 2D, uses first column for splitting.
    - y: Target values (1D numpy array)
    - initial_threshold: Starting threshold for splitting. If None, uses median of x.
    - max_depth: Maximum depth of the tree
    
    Returns:
    - tree: Dictionary representing the tree structure
    - mse_at_depth: List of MSE values at each depth [depth_0, depth_1, ..., depth_max]
    
    Example usage:
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y = np.array([2, 3, 5, 7, 11, 13, 17, 19])
    tree, mse_list = regression_tree(x, y, initial_threshold=4.5, max_depth=2)
    """
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    
    if len(x) != len(y):
        print("Error: x and y must have same length")
        return None, None
    
    if initial_threshold is None:
        initial_threshold = np.median(x)
    
    print(f"Building regression tree with max depth {max_depth}")
    print(f"Initial threshold: {initial_threshold}\n")
    
    # Track MSE at each depth
    mse_at_depth = []
    
    # Build tree recursively
    def build_tree(x_subset, y_subset, indices, depth, threshold):
        # Calculate MSE at current depth
        y_pred = np.mean(y_subset)
        mse = np.mean((y_subset - y_pred) ** 2)
        
        node = {
            'depth': depth,
            'n_samples': len(y_subset),
            'prediction': y_pred,
            'mse': mse,
            'indices': indices
        }
        
        # Base case: reached max depth or too few samples
        if depth >= max_depth or len(y_subset) < 2:
            return node
        
        # Find best split threshold for this node
        # Try splitting at current threshold
        left_mask = x_subset <= threshold
        right_mask = x_subset > threshold
        
        # If split is invalid, return leaf
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return node
        
        # Calculate MSE reduction from split
        left_mse = np.mean((y_subset[left_mask] - np.mean(y_subset[left_mask])) ** 2)
        right_mse = np.mean((y_subset[right_mask] - np.mean(y_subset[right_mask])) ** 2)
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        weighted_mse = (n_left * left_mse + n_right * right_mse) / len(y_subset)
        
        # Store split info
        node['threshold'] = threshold
        node['mse_after_split'] = weighted_mse
        
        # For next level, use median of each subset as new threshold
        left_threshold = np.median(x_subset[left_mask]) if np.sum(left_mask) > 0 else threshold
        right_threshold = np.median(x_subset[right_mask]) if np.sum(right_mask) > 0 else threshold
        
        # Recursively build left and right subtrees
        node['left'] = build_tree(
            x_subset[left_mask], 
            y_subset[left_mask],
            indices[left_mask],
            depth + 1,
            left_threshold
        )
        node['right'] = build_tree(
            x_subset[right_mask], 
            y_subset[right_mask],
            indices[right_mask],
            depth + 1,
            right_threshold
        )
        
        return node
    
    # Build the tree
    indices = np.arange(len(x))
    tree = build_tree(x, y, indices, 0, initial_threshold)
    
    # Calculate MSE at each depth level
    def collect_mse_by_depth(node, depth_mse_dict):
        depth = node['depth']
        if depth not in depth_mse_dict:
            depth_mse_dict[depth] = []
        depth_mse_dict[depth].append((node['mse'], node['n_samples']))
        
        if 'left' in node:
            collect_mse_by_depth(node['left'], depth_mse_dict)
        if 'right' in node:
            collect_mse_by_depth(node['right'], depth_mse_dict)
    
    depth_mse_dict = {}
    collect_mse_by_depth(tree, depth_mse_dict)
    
    # Calculate weighted average MSE at each depth
    for depth in range(max_depth + 1):
        if depth in depth_mse_dict:
            mse_samples = depth_mse_dict[depth]
            total_samples = sum(n for _, n in mse_samples)
            weighted_mse = sum(mse * n for mse, n in mse_samples) / total_samples
            mse_at_depth.append(weighted_mse)
            print(f"Depth {depth}: MSE = {weighted_mse:.6f}, Nodes = {len(mse_samples)}")
        else:
            mse_at_depth.append(None)
    
    print(f"\nMSE at each depth: {mse_at_depth}\n")
    
    return tree, mse_at_depth
