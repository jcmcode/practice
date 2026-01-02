"""
================================================================================
COMPREHENSIVE NUMPY GUIDE - Arrays, Operations, and Data Manipulation
================================================================================
NumPy (Numerical Python) is the fundamental package for numerical computing in Python.
It provides:
- N-dimensional array objects (ndarrays)
- Broadcasting capabilities
- Mathematical and statistical functions
- Linear algebra operations
- Random number generation
- Fourier transform capabilities
- Integration with other libraries (Pandas, SciPy, Matplotlib)
================================================================================
"""

import numpy as np


# ==============================================================================
# 1. CREATING NUMPY ARRAYS - Foundation of NumPy
# ==============================================================================

print("=" * 80)
print("1. CREATING NUMPY ARRAYS")
print("=" * 80)

# Create arrays from Python lists
arr1d = np.array([1, 2, 3, 4, 5])
print(f"\n1D Array from list: {arr1d}")
print(f"Shape: {arr1d.shape}, Data type: {arr1d.dtype}")

arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\n2D Array (Matrix):\n{arr2d}")
print(f"Shape: {arr2d.shape}")

arr3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"\n3D Array (Tensor):\n{arr3d}")
print(f"Shape: {arr3d.shape}")

# Create arrays with specific values
zeros = np.zeros((3, 4))  # Create array of all zeros
print(f"\nZeros array (3x4):\n{zeros}")

ones = np.ones((2, 5), dtype=int)  # Create array of all ones
print(f"\nOnes array (2x5):\n{ones}")

# Create arrays with ranges
range_arr = np.arange(0, 10, 2)  # Start, stop, step
print(f"\nArray from arange(0, 10, 2): {range_arr}")

linspace_arr = np.linspace(0, 1, 5)  # 5 evenly spaced values from 0 to 1
print(f"\nArray from linspace(0, 1, 5): {linspace_arr}")

logspace_arr = np.logspace(0, 2, 4)  # Logarithmically spaced (10^0 to 10^2)
print(f"\nArray from logspace(0, 2, 4): {logspace_arr}")

# Identity and diagonal matrices
identity = np.eye(3)  # 3x3 identity matrix
print(f"\nIdentity matrix (3x3):\n{identity}")

diagonal = np.diag([1, 2, 3])  # Matrix with values on diagonal
print(f"\nDiagonal matrix:\n{diagonal}")

# Empty array (uninitialized)
empty_arr = np.empty((2, 3))  # Contains arbitrary values
print(f"\nEmpty array (2x3) - contains arbitrary values:\n{empty_arr}")

# Full array - fill with specific value
full_arr = np.full((2, 3), 7)
print(f"\nFull array filled with 7:\n{full_arr}")


# ==============================================================================
# 2. ARRAY ATTRIBUTES - Understanding Array Properties
# ==============================================================================

print("\n" + "=" * 80)
print("2. ARRAY ATTRIBUTES")
print("=" * 80)

example_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

print(f"\nArray:\n{example_array}")
print(f"Shape: {example_array.shape}")  # Dimensions: (rows, columns)
print(f"Size: {example_array.size}")    # Total number of elements
print(f"ndim: {example_array.ndim}")    # Number of dimensions
print(f"dtype: {example_array.dtype}")  # Data type of elements
print(f"itemsize: {example_array.itemsize}")  # Size of each element in bytes
print(f"nbytes: {example_array.nbytes}")      # Total bytes consumed by array


# ==============================================================================
# 3. DATA TYPES IN NUMPY - Specifying Precision and Memory Usage
# ==============================================================================

print("\n" + "=" * 80)
print("3. DATA TYPES")
print("=" * 80)

int_array = np.array([1, 2, 3], dtype=np.int32)
print(f"\nInt32 array: {int_array}, dtype: {int_array.dtype}")

float_array = np.array([1.5, 2.3, 3.7], dtype=np.float64)
print(f"Float64 array: {float_array}, dtype: {float_array.dtype}")

complex_array = np.array([1+2j, 3+4j], dtype=np.complex128)
print(f"Complex array: {complex_array}, dtype: {complex_array.dtype}")

bool_array = np.array([True, False, True], dtype=bool)
print(f"Boolean array: {bool_array}, dtype: {bool_array.dtype}")

# Convert data types
converted = int_array.astype(float)
print(f"Int array converted to float: {converted}, dtype: {converted.dtype}")


# ==============================================================================
# 4. RESHAPING AND RESIZING ARRAYS - Changing Array Structure
# ==============================================================================

print("\n" + "=" * 80)
print("4. RESHAPING AND RESIZING")
print("=" * 80)

original = np.arange(12)
print(f"Original array: {original}, shape: {original.shape}")

# Reshape - change dimensions without changing data
reshaped = original.reshape((3, 4))
print(f"Reshaped to (3, 4):\n{reshaped}")

reshaped_3d = original.reshape((2, 3, 2))
print(f"Reshaped to (2, 3, 2):\n{reshaped_3d}")

# Flatten - convert to 1D
flattened = reshaped.flatten()
print(f"Flattened back to 1D: {flattened}")

# Ravel - similar to flatten but returns view (not copy)
raveled = reshaped.ravel()
print(f"Raveled to 1D: {raveled}")

# Transpose - swap axes
transposed = reshaped.T
print(f"Transposed (3, 4) to (4, 3):\n{transposed}")

# Expand dimensions
arr_2d = np.array([[1, 2, 3]])
expanded = np.expand_dims(arr_2d, axis=0)
print(f"\nOriginal shape: {arr_2d.shape}")
print(f"After expand_dims: {expanded.shape}")

# Squeeze - remove single-dimensional entries
squeezed = np.squeeze(expanded)
print(f"After squeeze: {squeezed.shape}")


# ==============================================================================
# 5. INDEXING AND SLICING - Accessing Array Elements
# ==============================================================================

print("\n" + "=" * 80)
print("5. INDEXING AND SLICING")
print("=" * 80)

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(f"Array:\n{arr}")

# Basic indexing
print(f"\nElement at [0, 0]: {arr[0, 0]}")
print(f"Element at [2, 3]: {arr[2, 3]}")
print(f"Last element: {arr[-1, -1]}")

# Slicing
print(f"\nFirst row: {arr[0, :]}")
print(f"First column: {arr[:, 0]}")
print(f"Middle 2x2 block:\n{arr[0:2, 1:3]}")
print(f"Every other row:\n{arr[::2, :]}")

# Boolean indexing - select elements based on condition
mask = arr > 5
print(f"\nElements greater than 5: {arr[mask]}")

even_elements = arr[arr % 2 == 0]
print(f"Even elements: {even_elements}")

# Fancy indexing - using integer arrays
indices = np.array([0, 2])
print(f"\nRows at indices [0, 2]:\n{arr[indices, :]}")

# Accessing multiple elements
selected = arr[[0, 2], [1, 3]]
print(f"Elements at (0,1) and (2,3): {selected}")


# ==============================================================================
# 6. MATHEMATICAL OPERATIONS - Element-wise and Aggregation Functions
# ==============================================================================

print("\n" + "=" * 80)
print("6. MATHEMATICAL OPERATIONS")
print("=" * 80)

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# Basic arithmetic operations (element-wise)
print(f"a: {a}")
print(f"b: {b}")
print(f"a + b: {a + b}")
print(f"a - b: {a - b}")
print(f"a * b: {a * b}")
print(f"b / a: {b / a}")
print(f"a ** 2: {a ** 2}")
print(f"np.sqrt(a): {np.sqrt(a)}")
print(f"np.exp(a): {np.exp(a)}")
print(f"np.log(b): {np.log(b)}")

# Trigonometric functions
angles = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
print(f"\nsin({angles}): {np.sin(angles)}")
print(f"cos({angles}): {np.cos(angles)}")

# Aggregation functions - reduce array to single value
print(f"\nSum of a: {np.sum(a)}")
print(f"Product of a: {np.prod(a)}")
print(f"Mean of a: {np.mean(a)}")
print(f"Median of a: {np.median(a)}")
print(f"Min of a: {np.min(a)}")
print(f"Max of a: {np.max(a)}")
print(f"Standard deviation: {np.std(a)}")
print(f"Variance: {np.var(a)}")

# Aggregation along axes
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\nMatrix:\n{matrix}")
print(f"Sum along axis 0 (columns): {np.sum(matrix, axis=0)}")
print(f"Sum along axis 1 (rows): {np.sum(matrix, axis=1)}")
print(f"Mean along axis 1: {np.mean(matrix, axis=1)}")

# Cumulative operations
print(f"\nCumulative sum of a: {np.cumsum(a)}")
print(f"Cumulative product of a: {np.cumprod(a)}")

# Rounding functions
decimals = np.array([1.235, 2.476, 3.891])
print(f"\nRound {decimals}: {np.round(decimals)}")
print(f"Floor {decimals}: {np.floor(decimals)}")
print(f"Ceil {decimals}: {np.ceil(decimals)}")


# ==============================================================================
# 7. BROADCASTING - Powerful Mechanism for Operating on Different-Sized Arrays
# ==============================================================================

print("\n" + "=" * 80)
print("7. BROADCASTING")
print("=" * 80)

"""
Broadcasting allows operations between arrays of different shapes.
Rules:
1. If arrays have different number of dimensions, pad smaller one with 1s on left
2. Arrays are compatible if dimensions are equal or one is 1
3. The dimension with size 1 is stretched to match the other
"""

# Example 1: 1D broadcast to 2D
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
scalar = 10
result = arr_2d + scalar  # scalar broadcasts to all elements
print(f"Array shape (2, 3) + scalar:")
print(result)

# Example 2: 1D array broadcast to 2D
arr_1d = np.array([1, 2, 3])
arr_2d = np.array([[10], [20], [30]])  # Shape (3, 1)
result = arr_2d + arr_1d  # 1D broadcasts along columns
print(f"\nArray (3, 1) + array (3,):")
print(result)

# Example 3: More complex broadcasting
a = np.array([[1, 2, 3], [4, 5, 6]])  # Shape (2, 3)
b = np.array([[1], [2]])               # Shape (2, 1)
result = a * b
print(f"\nArray (2, 3) * array (2, 1):")
print(result)

# Example 4: Broadcasting with different dimensions
c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # Shape (2, 2, 2)
d = np.array([10, 100])                              # Shape (2,)
result = c + d
print(f"\nArray (2, 2, 2) + array (2,):")
print(result)


# ==============================================================================
# 8. LINEAR ALGEBRA - Matrix Operations and Decompositions
# ==============================================================================

print("\n" + "=" * 80)
print("8. LINEAR ALGEBRA")
print("=" * 80)

# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"Matrix A:\n{A}")
print(f"Matrix B:\n{B}")

# Element-wise multiplication
print(f"\nElement-wise multiplication (A * B):\n{A * B}")

# Matrix multiplication (dot product)
dot_product = np.dot(A, B)
print(f"Matrix multiplication (A @ B or np.dot(A, B)):\n{dot_product}")

# Or use @ operator
at_product = A @ B
print(f"Using @ operator (A @ B):\n{at_product}")

# Transpose
print(f"\nTranspose of A:\n{A.T}")

# Determinant
determinant = np.linalg.det(A)
print(f"Determinant of A: {determinant}")

# Inverse
inverse = np.linalg.inv(A)
print(f"Inverse of A:\n{inverse}")

# Rank
rank = np.linalg.matrix_rank(A)
print(f"Rank of A: {rank}")

# Trace (sum of diagonal elements)
trace = np.trace(A)
print(f"Trace of A: {trace}")

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\nEigenvalues of A: {eigenvalues}")
print(f"Eigenvectors of A:\n{eigenvectors}")

# Singular Value Decomposition (SVD)
U, s, Vt = np.linalg.svd(A)
print(f"\nSingular Value Decomposition:")
print(f"U:\n{U}")
print(f"Singular values: {s}")
print(f"V^T:\n{Vt}")

# Solving linear systems (Ax = b)
b = np.array([1, 2])
x = np.linalg.solve(A, b)
print(f"\nSolving Ax = b where b = {b}")
print(f"Solution x: {x}")

# Matrix norm
norm = np.linalg.norm(A)
print(f"Frobenius norm of A: {norm}")

# QR decomposition
Q, R = np.linalg.qr(A)
print(f"\nQR Decomposition:")
print(f"Q:\n{Q}")
print(f"R:\n{R}")


# ==============================================================================
# 9. RANDOM NUMBER GENERATION - Creating Random Data
# ==============================================================================

print("\n" + "=" * 80)
print("9. RANDOM NUMBER GENERATION")
print("=" * 80)

# Set seed for reproducibility
np.random.seed(42)

# Uniform random numbers [0, 1)
uniform = np.random.random((3, 4))
print(f"Uniform random array [0, 1):\n{uniform}")

# Uniform random integers
random_ints = np.random.randint(1, 100, size=10)
print(f"Random integers [1, 100): {random_ints}")

# Normal distribution (Gaussian)
normal = np.random.normal(loc=0, scale=1, size=1000)
print(f"Normal distribution - mean: {np.mean(normal):.4f}, std: {np.std(normal):.4f}")

# Standard normal distribution
standard_normal = np.random.standard_normal((3, 3))
print(f"Standard normal array:\n{standard_normal}")

# Other distributions
binomial = np.random.binomial(n=10, p=0.5, size=5)
print(f"Binomial distribution: {binomial}")

poisson = np.random.poisson(lam=3, size=5)
print(f"Poisson distribution: {poisson}")

exponential = np.random.exponential(scale=1, size=5)
print(f"Exponential distribution: {exponential}")

# Random choice
choices = np.random.choice(['a', 'b', 'c'], size=10)
print(f"Random choices from list: {choices}")

# Shuffle
arr_to_shuffle = np.arange(10)
np.random.shuffle(arr_to_shuffle)
print(f"Shuffled array: {arr_to_shuffle}")

# Permutation (returns new array without modifying original)
arr_original = np.arange(10)
permuted = np.random.permutation(arr_original)
print(f"Original: {arr_original}, Permuted: {permuted}")


# ==============================================================================
# 10. SORTING AND SEARCHING - Organizing and Finding Data
# ==============================================================================

print("\n" + "=" * 80)
print("10. SORTING AND SEARCHING")
print("=" * 80)

unsorted = np.array([5, 2, 8, 1, 9, 3])
print(f"Unsorted array: {unsorted}")

# Sort (returns new array)
sorted_arr = np.sort(unsorted)
print(f"Sorted (ascending): {sorted_arr}")

# Sort in descending order
sorted_desc = np.sort(unsorted)[::-1]
print(f"Sorted (descending): {sorted_desc}")

# Argsort - returns indices that would sort the array
indices = np.argsort(unsorted)
print(f"Indices that would sort: {indices}")

# Lexsort - sort by multiple columns
names = np.array(['Alice', 'Bob', 'Alice', 'Charlie'])
ages = np.array([30, 25, 25, 35])
sorted_indices = np.lexsort((ages, names))
print(f"\nSort by name then age:")
print(f"Names: {names[sorted_indices]}")
print(f"Ages: {ages[sorted_indices]}")

# Searching
print(f"\nSearching in {unsorted}:")
print(f"Where value is 8: {np.where(unsorted == 8)}")
print(f"Where value > 5: {np.where(unsorted > 5)}")

# Searchsorted - find insertion positions
sorted_arr = np.array([1, 2, 3, 5, 8, 9])
print(f"Searchsorted for 4 in {sorted_arr}: {np.searchsorted(sorted_arr, 4)}")

# Extract min/max
print(f"Maximum value: {np.max(unsorted)}")
print(f"Index of maximum: {np.argmax(unsorted)}")
print(f"Minimum value: {np.min(unsorted)}")
print(f"Index of minimum: {np.argmin(unsorted)}")


# ==============================================================================
# 11. COMBINING AND SPLITTING ARRAYS - Merging and Dividing Data
# ==============================================================================

print("\n" + "=" * 80)
print("11. COMBINING AND SPLITTING ARRAYS")
print("=" * 80)

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Concatenate (join along existing axis)
concatenated = np.concatenate([a, b])
print(f"Concatenate [1,2,3] and [4,5,6]: {concatenated}")

# Stack arrays
a_2d = np.array([[1, 2], [3, 4]])
b_2d = np.array([[5, 6], [7, 8]])

stacked_v = np.vstack([a_2d, b_2d])  # Vertical stack (row-wise)
print(f"Vertical stack:\n{stacked_v}")

stacked_h = np.hstack([a_2d, b_2d])  # Horizontal stack (column-wise)
print(f"Horizontal stack:\n{stacked_h}")

stacked_d = np.dstack([a_2d, b_2d])  # Depth stack
print(f"Depth stack shape: {stacked_d.shape}")

# Column and row stack
col_stacked = np.column_stack([a, b])
print(f"Column stack:\n{col_stacked}")

row_stacked = np.row_stack([a_2d, b_2d])
print(f"Row stack:\n{row_stacked}")

# Splitting arrays
arr = np.arange(12).reshape((3, 4))
print(f"\nOriginal array:\n{arr}")

# Horizontal split
h_split = np.hsplit(arr, 2)  # Split into 2 equal parts horizontally
print(f"Horizontal split into 2 parts:\n{h_split[0]}\nand\n{h_split[1]}")

# Vertical split
v_split = np.vsplit(arr, 3)  # Split into 3 equal parts vertically
print(f"Vertical split into 3 parts: {len(v_split)} arrays")

# Split at specific indices
split_at_indices = np.split(arr, [1, 2], axis=0)  # Split at rows 1 and 2
print(f"Split at indices [1, 2] along axis 0: {len(split_at_indices)} arrays")


# ==============================================================================
# 12. UNIQUE AND DUPLICATE OPERATIONS - Finding Unique Values
# ==============================================================================

print("\n" + "=" * 80)
print("12. UNIQUE AND DUPLICATE OPERATIONS")
print("=" * 80)

arr = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5])
print(f"Array with duplicates: {arr}")

# Unique values
unique_vals = np.unique(arr)
print(f"Unique values: {unique_vals}")

# Unique with counts
unique_vals, counts = np.unique(arr, return_counts=True)
print(f"Unique values and their counts: {dict(zip(unique_vals, counts))}")

# Unique with indices
unique_vals, indices = np.unique(arr, return_index=True)
print(f"Unique values and first occurrence indices: {dict(zip(unique_vals, indices))}")

# Remove duplicates using unique
arr_2d = np.array([[1, 2], [2, 3], [1, 2]])
unique_rows = np.unique(arr_2d, axis=0)
print(f"\nUnique rows:\n{unique_rows}")


# ==============================================================================
# 13. SET OPERATIONS - Mathematical Operations on Arrays
# ==============================================================================

print("\n" + "=" * 80)
print("13. SET OPERATIONS")
print("=" * 80)

set_a = np.array([1, 2, 3, 4, 5])
set_b = np.array([3, 4, 5, 6, 7])

# Intersection
intersection = np.intersect1d(set_a, set_b)
print(f"Intersection of {set_a} and {set_b}: {intersection}")

# Union
union = np.union1d(set_a, set_b)
print(f"Union: {union}")

# Difference (elements in first but not in second)
difference = np.setdiff1d(set_a, set_b)
print(f"Difference (set_a - set_b): {difference}")

# Exclusive OR (symmetric difference)
xor = np.setxor1d(set_a, set_b)
print(f"Symmetric difference: {xor}")

# Check if element is in array
print(f"\nIs 3 in set_a? {np.isin(3, set_a)}")
mask = np.isin(set_a, set_b)
print(f"Elements of set_a that are in set_b: {set_a[mask]}")


# ==============================================================================
# 14. LOGICAL OPERATIONS - Boolean Operations and Comparisons
# ==============================================================================

print("\n" + "=" * 80)
print("14. LOGICAL OPERATIONS")
print("=" * 80)

a = np.array([True, True, False, False])
b = np.array([True, False, True, False])

# Logical operations
print(f"a: {a}")
print(f"b: {b}")
print(f"a AND b: {np.logical_and(a, b)}")
print(f"a OR b: {np.logical_or(a, b)}")
print(f"a XOR b: {np.logical_xor(a, b)}")
print(f"NOT a: {np.logical_not(a)}")

# All and any
numbers = np.array([2, 4, 6, 8])
print(f"\nArray: {numbers}")
print(f"All elements > 0? {np.all(numbers > 0)}")
print(f"Any element > 5? {np.any(numbers > 5)}")

# All and any along axis
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\nMatrix:\n{matrix}")
print(f"All elements > 0? {np.all(matrix > 0)}")
print(f"All along axis 0 (columns > 0)? {np.all(matrix > 0, axis=0)}")
print(f"Any along axis 1 (rows have > 3)? {np.any(matrix > 3, axis=1)}")


# ==============================================================================
# 15. COPYING ARRAYS - Understanding References vs Copies
# ==============================================================================

print("\n" + "=" * 80)
print("15. COPYING ARRAYS")
print("=" * 80)

original = np.array([1, 2, 3, 4, 5])

# No copy - just reference
reference = original
reference[0] = 999
print(f"After modifying reference:")
print(f"Original: {original}")
print(f"Reference: {reference}")

# Reset
original[0] = 1

# Shallow copy
shallow_copy = original.copy()
shallow_copy[0] = 888
print(f"\nAfter modifying shallow copy:")
print(f"Original: {original}")
print(f"Shallow copy: {shallow_copy}")

# Deep copy (useful for multidimensional arrays)
matrix = np.array([[1, 2, 3], [4, 5, 6]])
deep_copy = np.copy(matrix)
deep_copy[0, 0] = 777
print(f"\nAfter modifying deep copy:")
print(f"Original: {matrix}")
print(f"Deep copy: {deep_copy}")


# ==============================================================================
# 16. MEMORY AND PERFORMANCE - Understanding NumPy Efficiency
# ==============================================================================

print("\n" + "=" * 80)
print("16. MEMORY AND PERFORMANCE CONSIDERATIONS")
print("=" * 80)

# Memory layout (C vs Fortran order)
arr_c = np.array([[1, 2, 3], [4, 5, 6]], order='C')
arr_f = np.array([[1, 2, 3], [4, 5, 6]], order='F')

print(f"C-order (row-major) flags: {arr_c.flags['C_CONTIGUOUS']}")
print(f"Fortran-order (column-major) flags: {arr_f.flags['F_CONTIGUOUS']}")

# Strides (how to move through array in memory)
arr = np.arange(12).reshape((3, 4))
print(f"\nArray:\n{arr}")
print(f"Strides: {arr.strides}")  # (16, 4) means 16 bytes to next row, 4 bytes to next col

# Views vs copies
view = arr[:]
copy = arr.copy()
print(f"View shares memory with original? {np.shares_memory(arr, view)}")
print(f"Copy shares memory with original? {np.shares_memory(arr, copy)}")


# ==============================================================================
# 17. WORKING WITH MISSING DATA - NaN and Inf
# ==============================================================================

print("\n" + "=" * 80)
print("17. MISSING DATA AND SPECIAL VALUES")
print("=" * 80)

# NaN (Not a Number) - represents missing or undefined data
arr_with_nan = np.array([1, 2, np.nan, 4, 5])
print(f"Array with NaN: {arr_with_nan}")
print(f"Is NaN? {np.isnan(arr_with_nan)}")
print(f"Number of NaNs: {np.sum(np.isnan(arr_with_nan))}")

# Infinity
arr_with_inf = np.array([1, 2, np.inf, -np.inf, 5])
print(f"\nArray with inf: {arr_with_inf}")
print(f"Is inf? {np.isinf(arr_with_inf)}")

# Removing NaN
valid_data = arr_with_nan[~np.isnan(arr_with_nan)]
print(f"Array with NaN removed: {valid_data}")

# NaN-aware functions
arr_with_nan = np.array([1, 2, np.nan, 4, 5])
print(f"\nNaN-aware functions:")
print(f"Mean (ignoring NaN): {np.nanmean(arr_with_nan)}")
print(f"Sum (ignoring NaN): {np.nansum(arr_with_nan)}")
print(f"Std (ignoring NaN): {np.nanstd(arr_with_nan)}")
print(f"Min (ignoring NaN): {np.nanmin(arr_with_nan)}")
print(f"Max (ignoring NaN): {np.nanmax(arr_with_nan)}")


# ==============================================================================
# 18. VECTORIZATION - The Core Advantage of NumPy
# ==============================================================================

print("\n" + "=" * 80)
print("18. VECTORIZATION - Why NumPy is Fast")
print("=" * 80)

"""
Vectorization means writing code that works on entire arrays without explicit loops.
This is MUCH faster than Python loops because:
1. NumPy operations are implemented in C
2. No Python interpreter overhead per element
3. Better CPU cache usage

Example: Calculate element-wise operations
"""

# Traditional Python loop
python_list = list(range(1000))
result_loop = [x ** 2 for x in python_list]

# Vectorized NumPy operation
numpy_array = np.arange(1000)
result_vec = numpy_array ** 2

print(f"Both produce same result: {np.array_equal(result_loop, result_vec)}")
print("NumPy vectorized operation is ~100x faster than Python loop!")

# More vectorization examples
a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])

# Instead of: [a[i] * b[i] for i in range(len(a))]
result = a * b
print(f"\nVectorized multiplication: {result}")

# Instead of: [a[i] + b[i] for i in range(len(a))]
result = a + b
print(f"Vectorized addition: {result}")


# ==============================================================================
# 19. CONDITIONAL OPERATIONS - Using where() and select()
# ==============================================================================

print("\n" + "=" * 80)
print("19. CONDITIONAL OPERATIONS")
print("=" * 80)

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# np.where - conditional replacement
result = np.where(arr > 5, arr * 2, arr)
print(f"Array: {arr}")
print(f"np.where(arr > 5, arr*2, arr): {result}")

# Nested where
result = np.where(arr < 3, 'small', np.where(arr < 7, 'medium', 'large'))
print(f"Categorize: {result}")

# np.select - multiple conditions
conditions = [arr < 3, arr < 7, arr >= 7]
choices = ['small', 'medium', 'large']
result = np.select(conditions, choices)
print(f"Using select: {result}")

# Extract based on condition
evens = arr[arr % 2 == 0]
odds = arr[arr % 2 == 1]
print(f"Even numbers: {evens}")
print(f"Odd numbers: {odds}")


# ==============================================================================
# 20. ADVANCED INDEXING - Power User Techniques
# ==============================================================================

print("\n" + "=" * 80)
print("20. ADVANCED INDEXING TECHNIQUES")
print("=" * 80)

# Integer array indexing
arr = np.array([10, 20, 30, 40, 50])
indices = np.array([0, 2, 4])
result = arr[indices]
print(f"Array: {arr}")
print(f"arr[[0, 2, 4]]: {result}")

# Negative indices work from end
print(f"arr[[-1, -2]]: {arr[[-1, -2]]}")

# Multi-dimensional indexing
matrix = np.arange(12).reshape((3, 4))
print(f"\nMatrix:\n{matrix}")

row_indices = np.array([0, 2, 1])
col_indices = np.array([0, 1, 2])
result = matrix[row_indices, col_indices]
print(f"Elements at (0,0), (2,1), (1,2): {result}")

# Meshgrid for 2D indexing
x = np.array([0, 1, 2])
y = np.array([0, 1, 2, 3])
X, Y = np.meshgrid(x, y)
print(f"\nMeshgrid X:\n{X}")
print(f"Meshgrid Y:\n{Y}")


# ==============================================================================
# 21. FOURIER TRANSFORMS - Frequency Domain Analysis
# ==============================================================================

print("\n" + "=" * 80)
print("21. FOURIER TRANSFORMS - Frequency Analysis")
print("=" * 80)

# Create a signal (combination of sine waves)
t = np.linspace(0, 1, 100)
signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)

# Compute FFT
fft = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(signal))

print(f"Signal shape: {signal.shape}")
print(f"FFT result shape: {fft.shape}")
print(f"Frequencies shape: {frequencies.shape}")
print(f"Top frequency components (magnitude):")
magnitude = np.abs(fft)
top_indices = np.argsort(magnitude)[-3:]
for idx in top_indices[::-1]:
    print(f"  Frequency: {frequencies[idx]:.3f}, Magnitude: {magnitude[idx]:.1f}")


# ==============================================================================
# 22. PRACTICAL EXAMPLE - Data Analysis Pipeline
# ==============================================================================

print("\n" + "=" * 80)
print("22. PRACTICAL EXAMPLE - DATA ANALYSIS PIPELINE")
print("=" * 80)

# Simulate data: student grades
grades = np.random.randint(50, 100, size=100)
print(f"Sample grades: {grades[:10]}")

# Statistics
print(f"Mean: {np.mean(grades):.2f}")
print(f"Median: {np.median(grades):.2f}")
print(f"Std Dev: {np.std(grades):.2f}")
print(f"Min: {np.min(grades)}, Max: {np.max(grades)}")

# Categorize grades
grade_categories = np.where(grades >= 90, 'A',
                            np.where(grades >= 80, 'B',
                                   np.where(grades >= 70, 'C',
                                          np.where(grades >= 60, 'D', 'F'))))
unique_grades, counts = np.unique(grade_categories, return_counts=True)
print(f"Grade distribution: {dict(zip(unique_grades, counts))}")

# Filter and analyze
pass_count = np.sum(grades >= 60)
fail_count = np.sum(grades < 60)
print(f"Pass rate: {pass_count}/{len(grades)} = {pass_count/len(grades)*100:.1f}%")


# ==============================================================================
# 23. RECOMMENDATIONS AND BEST PRACTICES
# ==============================================================================

print("\n" + "=" * 80)
print("23. NUMPY BEST PRACTICES")
print("=" * 80)

print("""
✓ BEST PRACTICES:
  1. Use vectorized operations instead of loops
  2. Choose appropriate data types to save memory
  3. Pre-allocate arrays when size is known
  4. Use views when possible to save memory (but be careful!)
  5. Use inplace operations +=, -=, etc. when modifying arrays
  6. Leverage broadcasting to avoid explicit loops
  7. Use numpy.random.seed() for reproducibility
  8. Profile code to identify bottlenecks

✓ COMMON USE CASES:
  - Mathematical computations
  - Signal and image processing
  - Statistical analysis and data science
  - Linear algebra and matrix operations
  - Scientific computing and simulations
  - Data preprocessing and cleaning
  - Time series analysis
  - Fourier analysis and transforms

✗ AVOID:
  - Using NumPy for simple scalar operations
  - Modifying array shape without reshape()
  - Assuming arrays are independent when using slices
  - Comparing with == instead of np.array_equal() for arrays
  - Forgetting that NaN != NaN (use np.isnan())

✓ INTEGRATION:
  - Pandas: uses NumPy under the hood for DataFrames
  - Matplotlib: plotting works seamlessly with NumPy arrays
  - SciPy: builds on NumPy for advanced scientific computing
  - Scikit-learn: ML library built on NumPy
  - TensorFlow/PyTorch: support NumPy arrays for easy conversion
""")

print("\n" + "=" * 80)
print("END OF NUMPY COMPREHENSIVE GUIDE")
print("=" * 80)
