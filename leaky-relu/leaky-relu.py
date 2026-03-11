import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    # Write code here
    x = np.asarray(x, dtype=float)  # handles list, scalar, or NumPy array
    return np.where(x > 0, x, alpha * x)  # fully vectorized, no loops
    pass