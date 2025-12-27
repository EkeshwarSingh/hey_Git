import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
mu = np.array([2, -3, 1])
Sigma = np.array([
    [1, 1, 1],
    [1, 3, 2],
    [1, 2, 2]
])
sample_sizes = [5, 50, 200, 500, 1000]

for n in sample_sizes:
    # Draw n samples from the multivariate normal distribution
    X = np.random.multivariate_normal(mu, Sigma, n)
    
    # Compute sample mean (vector)
    X_bar = np.mean(X, axis=0)
    
    # Compute sample covariance matrix (ndarray)
    S = np.cov(X, rowvar=False, bias=False)
    
    print(f"\nSample size n = {n}")
    print("Sample mean (X̄):")
    print(X_bar)
    print("Sample covariance matrix (S):")
    print(S)
    print("Difference X̄ - μ:")
    print(X_bar - mu)
    print("Difference S - Σ:")
    print(S - Sigma)

def two_sum(nums: list[int], target: int) -> list[int]:
    """
    Finds two numbers in a list that sum up to a target value.

    Args:
        nums: A list of integers.
        target: The target integer sum.

    Returns:
        A list containing the indices of the two numbers.
    """
    # Create a hash map to store numbers we've seen and their indices
    num_map = {}  # {number: index}

    # Enumerate provides both the index and the value
    for i, num in enumerate(nums):
        # Calculate the complement needed
        complement = target - num

        # Check if the complement is already in our map
        if complement in num_map:
            # If it is, we found our pair
            return [num_map[complement], i]

        # If the complement is not found, add the current number and its index to the map
        num_map[num] = i

# --- Example Usage ---
nums_list = [2, 7, 11, 15]
target_value = 9
result = two_sum(nums_list, target_value)
print(f"Indices: {result}")  # Output: Indices: [0, 1]