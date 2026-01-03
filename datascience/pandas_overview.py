"""
================================================================================
COMPREHENSIVE PANDAS GUIDE - Data Analysis and Manipulation
================================================================================
Pandas is a powerful data manipulation and analysis library built on top of NumPy.
It provides:
- Series: 1D labeled array
- DataFrame: 2D labeled table (like a spreadsheet or SQL table)
- Advanced indexing and slicing
- Data alignment and reshaping
- Grouping and aggregation
- Time series functionality
- Missing data handling
- Input/Output tools
- Statistical analysis
================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ==============================================================================
# 1. CREATING SERIES - 1D Labeled Data Structures
# ==============================================================================

print("=" * 80)
print("1. CREATING PANDAS SERIES")
print("=" * 80)

# Series from list
series1 = pd.Series([1, 2, 3, 4, 5])
print(f"Series from list:\n{series1}\n")

# Series with custom index
series2 = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(f"Series with custom index:\n{series2}\n")

# Series from dictionary
data_dict = {'name': 'Alice', 'age': 30, 'city': 'New York'}
series3 = pd.Series(data_dict)
print(f"Series from dictionary:\n{series3}\n")

# Series from scalar (broadcasted)
series4 = pd.Series(100, index=['x', 'y', 'z'])
print(f"Series filled with scalar:\n{series4}\n")

# Series with specific dtype
series5 = pd.Series([1, 2, 3], dtype='float64')
print(f"Series with float64 dtype:\n{series5}\n")

# Series properties
print(f"Series dtype: {series2.dtype}")
print(f"Series index: {series2.index.tolist()}")
print(f"Series values: {series2.values}")
print(f"Series name: {series2.name}")
print(f"Series shape: {series2.shape}\n")


# ==============================================================================
# 2. CREATING DATAFRAMES - 2D Labeled Data Structures
# ==============================================================================

print("=" * 80)
print("2. CREATING PANDAS DATAFRAMES")
print("=" * 80)

# DataFrame from dictionary of lists
df1 = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'City': ['New York', 'London', 'Paris', 'Tokyo'],
    'Salary': [70000, 80000, 90000, 100000]
})
print(f"DataFrame from dictionary:\n{df1}\n")

# DataFrame from list of lists
data = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
df2 = pd.DataFrame(data, columns=['A', 'B', 'C'])
print(f"DataFrame from list of lists:\n{df2}\n")

# DataFrame from NumPy array
arr = np.random.randn(3, 4)
df3 = pd.DataFrame(arr, columns=['Col1', 'Col2', 'Col3', 'Col4'])
print(f"DataFrame from NumPy array:\n{df3}\n")

# DataFrame with custom index and columns
df4 = pd.DataFrame(
    np.random.randn(3, 3),
    index=['row1', 'row2', 'row3'],
    columns=['X', 'Y', 'Z']
)
print(f"DataFrame with custom index:\n{df4}\n")

# DataFrame properties
print(f"DataFrame shape: {df1.shape}")
print(f"DataFrame columns: {df1.columns.tolist()}")
print(f"DataFrame index: {df1.index.tolist()}")
print(f"DataFrame dtypes:\n{df1.dtypes}\n")


# ==============================================================================
# 3. SERIES INDEXING AND SLICING
# ==============================================================================

print("=" * 80)
print("3. SERIES INDEXING AND SLICING")
print("=" * 80)

series = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])
print(f"Series:\n{series}\n")

# Label-based indexing (loc)
print(f"series['c']: {series['c']}")
print(f"series[['a', 'c', 'e']]: {series[['a', 'c', 'e']].tolist()}")

# Position-based indexing (iloc)
print(f"series.iloc[0]: {series.iloc[0]}")
print(f"series.iloc[1:4]: {series.iloc[1:4].tolist()}")

# Slicing
print(f"series['b':'d']: {series['b':'d'].tolist()}")

# Boolean indexing
print(f"Values > 25: {series[series > 25].tolist()}")

# Get specific index/value
print(f"Index at position 2: {series.index[2]}")
print(f"Value at position 2: {series.iloc[2]}")


# ==============================================================================
# 4. DATAFRAME INDEXING AND SELECTION
# ==============================================================================

print("\n" + "=" * 80)
print("4. DATAFRAME INDEXING AND SELECTION")
print("=" * 80)

df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [10, 20, 30, 40],
    'C': [100, 200, 300, 400],
    'D': ['x', 'y', 'z', 'w']
})
print(f"DataFrame:\n{df}\n")

# Select single column (returns Series)
print(f"df['A']:\n{df['A']}\n")

# Select multiple columns
print(f"df[['A', 'C']]:\n{df[['A', 'C']]}\n")

# Select rows by position (iloc)
print(f"df.iloc[0]:  # First row\n{df.iloc[0]}\n")
print(f"df.iloc[0:2]:  # First two rows\n{df.iloc[0:2]}\n")

# Select rows by label (loc)
print(f"df.loc[0]:  # Row with index label 0\n{df.loc[0]}\n")

# Select specific element
print(f"df.loc[0, 'A']: {df.loc[0, 'A']}")
print(f"df.iloc[0, 0]: {df.iloc[0, 0]}\n")

# Boolean indexing
print(f"Rows where A > 2:\n{df[df['A'] > 2]}\n")

# Filter multiple conditions
print(f"Rows where A > 1 AND B > 15:\n{df[(df['A'] > 1) & (df['B'] > 15)]}\n")

# Using .at and .iat for single value (faster)
print(f"df.at[0, 'A']: {df.at[0, 'A']}")
print(f"df.iat[0, 0]: {df.iat[0, 0]}\n")


# ==============================================================================
# 5. VIEWING AND INSPECTING DATA
# ==============================================================================

print("=" * 80)
print("5. VIEWING AND INSPECTING DATA")
print("=" * 80)

df = pd.DataFrame(np.random.randn(100, 5), columns=['A', 'B', 'C', 'D', 'E'])

# Head and tail
print(f"df.head(3):\n{df.head(3)}\n")
print(f"df.tail(2):\n{df.tail(2)}\n")

# Info - overview of DataFrame
print(f"df.info():\n")
df.info()

# Describe - statistical summary
print(f"\ndf.describe():\n{df.describe()}\n")

# Shape and size
print(f"Shape: {df.shape}")
print(f"Size: {df.size}")

# Columns and index information
print(f"Columns: {df.columns.tolist()}")
print(f"Index: {df.index.tolist()[:5]}... (showing first 5)\n")

# Value counts (unique values)
df_small = pd.DataFrame({'Color': ['red', 'blue', 'red', 'green', 'blue', 'red']})
print(f"df_small:\n{df_small}")
print(f"Value counts:\n{df_small['Color'].value_counts()}\n")


# ==============================================================================
# 6. ADDING AND REMOVING COLUMNS
# ==============================================================================

print("=" * 80)
print("6. ADDING AND REMOVING COLUMNS")
print("=" * 80)

df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35]
})
print(f"Original DataFrame:\n{df}\n")

# Add new column
df['City'] = ['New York', 'London', 'Paris']
print(f"After adding 'City':\n{df}\n")

# Add column with calculation
df['Age + 10'] = df['Age'] + 10
print(f"After adding 'Age + 10':\n{df}\n")

# Add column based on condition
df['Adult'] = df['Age'] >= 30
print(f"After adding 'Adult':\n{df}\n")

# Drop column
df_dropped = df.drop('Age + 10', axis=1)
print(f"After dropping 'Age + 10':\n{df_dropped}\n")

# Drop multiple columns
df_dropped2 = df.drop(['Adult', 'City'], axis=1)
print(f"After dropping multiple columns:\n{df_dropped2}\n")

# Rename columns
df_renamed = df.rename(columns={'Age': 'Years', 'City': 'Location'})
print(f"After renaming:\n{df_renamed}\n")


# ==============================================================================
# 7. HANDLING MISSING DATA
# ==============================================================================

print("=" * 80)
print("7. HANDLING MISSING DATA")
print("=" * 80)

df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, np.nan, 8],
    'C': [9, 10, 11, 12]
})
print(f"DataFrame with NaN:\n{df}\n")

# Check for missing data
print(f"df.isnull():\n{df.isnull()}\n")
print(f"df.notnull():\n{df.notnull()}\n")

# Count missing values
print(f"Missing values per column:\n{df.isnull().sum()}\n")

# Drop rows with any NaN
df_dropna = df.dropna()
print(f"After dropna():\n{df_dropna}\n")

# Drop rows where all values are NaN
df_dropna_all = df.dropna(how='all')
print(f"After dropna(how='all'):\n{df_dropna_all}\n")

# Fill missing values
df_filled_constant = df.fillna(0)
print(f"After fillna(0):\n{df_filled_constant}\n")

# Forward fill (propagate previous value)
df_ffill = df.fillna(method='ffill')
print(f"After fillna(method='ffill'):\n{df_ffill}\n")

# Backward fill
df_bfill = df.fillna(method='bfill')
print(f"After fillna(method='bfill'):\n{df_bfill}\n")

# Interpolate (fill based on surrounding values)
df_interpolate = df.interpolate()
print(f"After interpolate():\n{df_interpolate}\n")


# ==============================================================================
# 8. DATA TYPE CONVERSION AND CASTING
# ==============================================================================

print("=" * 80)
print("8. DATA TYPE CONVERSION")
print("=" * 80)

df = pd.DataFrame({
    'String': ['1', '2', '3'],
    'Int': [10, 20, 30],
    'Float': [1.5, 2.5, 3.5]
})
print(f"Original dtypes:\n{df.dtypes}\n")

# Convert to different types
df['String_to_int'] = pd.to_numeric(df['String'])
df['Float_to_int'] = df['Float'].astype('int64')
df['Int_to_string'] = df['Int'].astype('string')
print(f"After conversions:\n{df.dtypes}\n")

# Convert string to datetime
dates = pd.Series(['2024-01-01', '2024-02-01', '2024-03-01'])
dates_converted = pd.to_datetime(dates)
print(f"Converted to datetime:\n{dates_converted}\n")

# Categorical data
df_cat = pd.DataFrame({'Color': ['red', 'blue', 'red', 'green', 'blue']})
df_cat['Color'] = df_cat['Color'].astype('category')
print(f"Categorical dtype:\n{df_cat.dtypes}\n")


# ==============================================================================
# 9. BASIC STATISTICS AND AGGREGATION
# ==============================================================================

print("=" * 80)
print("9. BASIC STATISTICS AND AGGREGATION")
print("=" * 80)

df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': [100, 200, 300, 400, 500]
})
print(f"DataFrame:\n{df}\n")

# Basic statistics
print(f"Sum:\n{df.sum()}\n")
print(f"Mean:\n{df.mean()}\n")
print(f"Median:\n{df.median()}\n")
print(f"Std Dev:\n{df.std()}\n")
print(f"Min:\n{df.min()}\n")
print(f"Max:\n{df.max()}\n")

# Describe all
print(f"Describe:\n{df.describe()}\n")

# Statistics on specific column
print(f"A column stats:\n{df['A'].describe()}\n")

# Cumulative operations
print(f"Cumulative sum:\n{df['A'].cumsum()}\n")
print(f"Cumulative product:\n{df['A'].cumprod()}\n")

# Correlation matrix
print(f"Correlation matrix:\n{df.corr()}\n")

# Covariance matrix
print(f"Covariance matrix:\n{df.cov()}\n")


# ==============================================================================
# 10. GROUPBY AND AGGREGATION - Most Powerful Feature
# ==============================================================================

print("=" * 80)
print("10. GROUPBY AND AGGREGATION")
print("=" * 80)

df = pd.DataFrame({
    'Department': ['Sales', 'Sales', 'Marketing', 'Marketing', 'IT', 'IT'],
    'Employee': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
    'Salary': [50000, 55000, 60000, 65000, 70000, 75000],
    'Year': [2023, 2023, 2023, 2024, 2023, 2024]
})
print(f"DataFrame:\n{df}\n")

# Group by single column
grouped = df.groupby('Department')['Salary'].mean()
print(f"Average salary by department:\n{grouped}\n")

# Group by with aggregation
grouped_agg = df.groupby('Department')['Salary'].agg(['mean', 'sum', 'count'])
print(f"Multiple aggregations:\n{grouped_agg}\n")

# Group by multiple columns
grouped_multi = df.groupby(['Department', 'Year'])['Salary'].mean()
print(f"Average salary by department and year:\n{grouped_multi}\n")

# Custom aggregation function
def salary_range(x):
    return x.max() - x.min()

grouped_custom = df.groupby('Department')['Salary'].agg(salary_range)
print(f"Salary range by department:\n{grouped_custom}\n")

# Multiple columns aggregation
grouped_dict = df.groupby('Department').agg({
    'Salary': 'mean',
    'Employee': 'count'
}).rename(columns={'Employee': 'Count'})
print(f"Multiple column aggregation:\n{grouped_dict}\n")

# Transform - apply function and keep original index
df['Salary_scaled'] = df.groupby('Department')['Salary'].transform(lambda x: (x - x.mean()) / x.std())
print(f"After transform (normalized salary):\n{df[['Department', 'Salary', 'Salary_scaled']]}\n")


# ==============================================================================
# 11. SORTING AND RANKING
# ==============================================================================

print("=" * 80)
print("11. SORTING AND RANKING")
print("=" * 80)

df = pd.DataFrame({
    'Name': ['Charlie', 'Alice', 'Bob', 'David'],
    'Age': [35, 25, 30, 28],
    'Score': [85, 92, 78, 88]
})
print(f"Original DataFrame:\n{df}\n")

# Sort by column
sorted_name = df.sort_values('Name')
print(f"Sorted by Name:\n{sorted_name}\n")

# Sort descending
sorted_score = df.sort_values('Score', ascending=False)
print(f"Sorted by Score (descending):\n{sorted_score}\n")

# Sort by multiple columns
sorted_multi = df.sort_values(['Age', 'Score'], ascending=[True, False])
print(f"Sorted by Age (asc) then Score (desc):\n{sorted_multi}\n")

# Sort by index
sorted_index = df.sort_index()
print(f"Sorted by index:\n{sorted_index}\n")

# Ranking
df['Age_rank'] = df['Age'].rank()
df['Score_rank'] = df['Score'].rank(method='dense')
print(f"After ranking:\n{df[['Name', 'Age', 'Age_rank', 'Score_rank']]}\n")


# ==============================================================================
# 12. MERGING AND JOINING DATAFRAMES
# ==============================================================================

print("=" * 80)
print("12. MERGING AND JOINING DATAFRAMES")
print("=" * 80)

# Create sample dataframes
df1 = pd.DataFrame({
    'ID': [1, 2, 3],
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Department': ['Sales', 'IT', 'HR']
})

df2 = pd.DataFrame({
    'ID': [1, 2, 3],
    'Salary': [50000, 70000, 60000]
})

print(f"df1:\n{df1}\n")
print(f"df2:\n{df2}\n")

# Merge on common column
merged = pd.merge(df1, df2, on='ID')
print(f"Merged on ID:\n{merged}\n")

# Merge with different key names
df3 = pd.DataFrame({
    'EmployeeID': [1, 2, 3],
    'Bonus': [5000, 7000, 6000]
})
merged2 = pd.merge(df1, df3, left_on='ID', right_on='EmployeeID')
print(f"Merged with different keys:\n{merged2}\n")

# Different join types
df_left = pd.DataFrame({'key': ['A', 'B', 'C'], 'value_left': [1, 2, 3]})
df_right = pd.DataFrame({'key': ['B', 'C', 'D'], 'value_right': [20, 30, 40]})

print(f"Left DataFrame:\n{df_left}\n")
print(f"Right DataFrame:\n{df_right}\n")

# Inner join (only matching keys)
inner = pd.merge(df_left, df_right, on='key', how='inner')
print(f"Inner join:\n{inner}\n")

# Left join (all from left)
left = pd.merge(df_left, df_right, on='key', how='left')
print(f"Left join:\n{left}\n")

# Right join (all from right)
right = pd.merge(df_left, df_right, on='key', how='right')
print(f"Right join:\n{right}\n")

# Outer join (all from both)
outer = pd.merge(df_left, df_right, on='key', how='outer')
print(f"Outer join:\n{outer}\n")

# Concatenate (combine along rows or columns)
concat_rows = pd.concat([df_left, df_right], ignore_index=True)
print(f"Concatenate along rows:\n{concat_rows}\n")

# Join by index
df_a = pd.DataFrame({'A': [1, 2, 3]}, index=['x', 'y', 'z'])
df_b = pd.DataFrame({'B': [10, 20, 30]}, index=['x', 'y', 'z'])
joined = df_a.join(df_b)
print(f"Join by index:\n{joined}\n")


# ==============================================================================
# 13. RESHAPING DATA - Pivot, Melt, Stack, Unstack
# ==============================================================================

print("=" * 80)
print("13. RESHAPING DATA")
print("=" * 80)

# Sample data
df = pd.DataFrame({
    'Date': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
    'Product': ['A', 'B', 'A', 'B'],
    'Sales': [100, 150, 120, 160]
})
print(f"Original DataFrame:\n{df}\n")

# Pivot table
pivot = df.pivot(index='Date', columns='Product', values='Sales')
print(f"Pivot table:\n{pivot}\n")

# Pivot table with aggregation
df2 = pd.DataFrame({
    'Date': ['2024-01-01', '2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
    'Product': ['A', 'B', 'A', 'B', 'B'],
    'Sales': [100, 150, 120, 160, 170]
})
pivot_agg = pd.pivot_table(df2, index='Date', columns='Product', values='Sales', aggfunc='sum')
print(f"Pivot with aggregation:\n{pivot_agg}\n")

# Melt (unpivot) - convert columns to rows
melted = pivot.reset_index().melt(id_vars='Date', var_name='Product', value_name='Sales')
print(f"Melt (unpivot):\n{melted}\n")

# Stack - convert columns to rows (hierarchical)
stacked = pivot.stack()
print(f"Stack:\n{stacked}\n")

# Unstack - convert index levels to columns (hierarchical)
unstacked = stacked.unstack()
print(f"Unstack:\n{unstacked}\n")

# Transpose
transposed = pivot.T
print(f"Transpose:\n{transposed}\n")


# ==============================================================================
# 14. TIME SERIES - Working with Dates and Times
# ==============================================================================

print("=" * 80)
print("14. TIME SERIES DATA")
print("=" * 80)

# Create date range
dates = pd.date_range('2024-01-01', periods=5, freq='D')
print(f"Date range:\n{dates}\n")

# Create time series
ts = pd.Series(np.random.randn(5), index=dates)
print(f"Time series:\n{ts}\n")

# DataFrame with datetime index
df_ts = pd.DataFrame({
    'Close': [100, 102, 101, 103, 105],
    'Volume': [1000, 1200, 1100, 1300, 1400]
}, index=pd.date_range('2024-01-01', periods=5, freq='D'))
print(f"DataFrame with datetime index:\n{df_ts}\n")

# Resample - change frequency
resampled = df_ts['Close'].resample('2D').mean()  # Every 2 days
print(f"Resampled to 2-day frequency:\n{resampled}\n")

# Rolling window
rolling = df_ts['Close'].rolling(window=2).mean()
print(f"Rolling 2-day average:\n{rolling}\n")

# Expanding window
expanding = df_ts['Close'].expanding().mean()
print(f"Expanding average:\n{expanding}\n")

# Extract date components
df_ts['Year'] = df_ts.index.year
df_ts['Month'] = df_ts.index.month
df_ts['Day'] = df_ts.index.day
df_ts['DayOfWeek'] = df_ts.index.dayofweek
print(f"After extracting date components:\n{df_ts}\n")

# Shift data (lag/lead)
df_temp = df_ts[['Close']].copy()
df_temp['Close_Lag1'] = df_temp['Close'].shift(1)
df_temp['Close_Lead1'] = df_temp['Close'].shift(-1)
print(f"Shifted data:\n{df_temp}\n")

# Percentage change
pct_change = df_ts['Close'].pct_change()
print(f"Percentage change:\n{pct_change}\n")


# ==============================================================================
# 15. STRING OPERATIONS
# ==============================================================================

print("=" * 80)
print("15. STRING OPERATIONS")
print("=" * 80)

df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Email': ['alice@example.com', 'bob@example.com', 'charlie@example.com']
})
print(f"Original DataFrame:\n{df}\n")

# String methods (accessed via .str)
print(f"Uppercase:\n{df['Name'].str.upper()}\n")
print(f"Lowercase:\n{df['Name'].str.lower()}\n")
print(f"Length:\n{df['Name'].str.len()}\n")

# Check if contains substring
print(f"Contains 'li':\n{df['Name'].str.contains('li')}\n")

# Extract substring
print(f"First 3 characters:\n{df['Name'].str[:3]}\n")

# Replace
print(f"Replace 'Alice' with 'Alicia':\n{df['Name'].str.replace('Alice', 'Alicia')}\n")

# Split
print(f"Split email:\n{df['Email'].str.split('@')}\n")

# Extract using regex
print(f"Extract domain:\n{df['Email'].str.extract(r'@(\w+\.\w+)')}\n")


# ==============================================================================
# 16. APPLY AND MAP FUNCTIONS - Custom Operations
# ==============================================================================

print("=" * 80)
print("16. APPLY AND MAP FUNCTIONS")
print("=" * 80)

df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [10, 20, 30, 40]
})
print(f"Original DataFrame:\n{df}\n")

# Apply function to column
def double(x):
    return x * 2

df['A_doubled'] = df['A'].apply(double)
print(f"Apply to column:\n{df}\n")

# Lambda function
df['B_halved'] = df['B'].apply(lambda x: x / 2)
print(f"Using lambda:\n{df[['B', 'B_halved']]}\n")

# Apply to entire row
df['Sum'] = df.apply(lambda row: row['A'] + row['B'], axis=1)
print(f"Apply to rows:\n{df[['A', 'B', 'Sum']]}\n")

# Map function on Series (element-wise)
mapping = {1: 'one', 2: 'two', 3: 'three', 4: 'four'}
df['A_name'] = df['A'].map(mapping)
print(f"Map function:\n{df[['A', 'A_name']]}\n")

# Apply with multiple arguments
df['C'] = df.apply(lambda row: row['A'] if row['B'] > 15 else row['B'], axis=1)
print(f"Conditional apply:\n{df[['A', 'B', 'C']]}\n")


# ==============================================================================
# 17. FILTERING AND QUERY - Multiple Ways to Filter Data
# ==============================================================================

print("=" * 80)
print("17. FILTERING AND QUERY")
print("=" * 80)

df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 35, 40, 28],
    'Score': [85, 92, 78, 88, 95]
})
print(f"Original DataFrame:\n{df}\n")

# Simple condition
filtered1 = df[df['Age'] > 30]
print(f"Age > 30:\n{filtered1}\n")

# Multiple conditions with & (AND)
filtered2 = df[(df['Age'] > 25) & (df['Score'] > 85)]
print(f"Age > 25 AND Score > 85:\n{filtered2}\n")

# Multiple conditions with | (OR)
filtered3 = df[(df['Age'] > 35) | (df['Score'] > 90)]
print(f"Age > 35 OR Score > 90:\n{filtered3}\n")

# Using .isin() for multiple values
filtered4 = df[df['Name'].isin(['Alice', 'Charlie'])]
print(f"Names in ['Alice', 'Charlie']:\n{filtered4}\n")

# Query method (more readable)
filtered5 = df.query('Age > 25 and Score > 85')
print(f"Using query():\n{filtered5}\n")

# Using between
filtered6 = df[df['Age'].between(25, 35)]
print(f"Age between 25 and 35:\n{filtered6}\n")


# ==============================================================================
# 18. DUPLICATES - Finding and Removing Duplicate Rows
# ==============================================================================

print("=" * 80)
print("18. HANDLING DUPLICATES")
print("=" * 80)

df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob'],
    'Age': [25, 30, 25, 35, 30],
    'City': ['NY', 'London', 'NY', 'Paris', 'London']
})
print(f"DataFrame with duplicates:\n{df}\n")

# Check for duplicates
print(f"Duplicate rows:\n{df.duplicated()}\n")

# Check duplicates in specific column
print(f"Duplicate names:\n{df.duplicated(subset=['Name'])}\n")

# Show duplicate rows
print(f"Show all duplicate rows:\n{df[df.duplicated(keep=False)]}\n")

# Drop duplicates
df_unique = df.drop_duplicates()
print(f"After drop_duplicates():\n{df_unique}\n")

# Drop duplicates keeping first occurrence
df_first = df.drop_duplicates(keep='first')
print(f"Keep first occurrence:\n{df_first}\n")

# Drop duplicates keeping last occurrence
df_last = df.drop_duplicates(keep='last')
print(f"Keep last occurrence:\n{df_last}\n")


# ==============================================================================
# 19. INPUT/OUTPUT - Reading and Writing Data
# ==============================================================================

print("=" * 80)
print("19. INPUT/OUTPUT OPERATIONS")
print("=" * 80)

# Create sample DataFrame
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Score': [85, 92, 88]
})

# Write to CSV
csv_path = '/tmp/sample.csv'
df.to_csv(csv_path, index=False)
print(f"Written to CSV: {csv_path}")

# Read from CSV
df_read = pd.read_csv(csv_path)
print(f"Read from CSV:\n{df_read}\n")

# Write to Excel (requires openpyxl)
# excel_path = '/tmp/sample.xlsx'
# df.to_excel(excel_path, index=False)
# print(f"Written to Excel: {excel_path}")

# Read from Excel
# df_excel = pd.read_excel(excel_path)

# Write to JSON
json_path = '/tmp/sample.json'
df.to_json(json_path)
print(f"Written to JSON: {json_path}")

# Read from JSON
df_json = pd.read_json(json_path)
print(f"Read from JSON:\n{df_json}\n")

# Write to SQL (requires SQLAlchemy)
# from sqlalchemy import create_engine
# engine = create_engine('sqlite:////tmp/database.db')
# df.to_sql('table_name', engine, if_exists='replace')

# Display options
print(f"Max rows display: {pd.get_option('display.max_rows')}")
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 100)
print("Display options updated\n")


# ==============================================================================
# 20. MULTI-LEVEL INDEXING (HIERARCHICAL)
# ==============================================================================

print("=" * 80)
print("20. MULTI-LEVEL INDEXING")
print("=" * 80)

# Create MultiIndex DataFrame
arrays = [
    ['A', 'A', 'B', 'B'],
    [1, 2, 1, 2]
]
index = pd.MultiIndex.from_arrays(arrays, names=['Letter', 'Number'])
df_multi = pd.DataFrame(
    np.random.randn(4, 3),
    index=index,
    columns=['X', 'Y', 'Z']
)
print(f"MultiIndex DataFrame:\n{df_multi}\n")

# Access with MultiIndex
print(f"df_multi.loc['A']:\n{df_multi.loc['A']}\n")
print(f"df_multi.loc[('A', 1)]:\n{df_multi.loc[('A', 1)]}\n")

# Stack and unstack
stacked = df_multi.stack()
print(f"Stacked:\n{stacked}\n")

unstacked = stacked.unstack()
print(f"Unstacked:\n{unstacked}\n")

# Swap index levels
swapped = df_multi.swaplevel()
print(f"Swapped levels:\n{swapped}\n")


# ==============================================================================
# 21. WINDOW FUNCTIONS - Advanced Aggregations
# ==============================================================================

print("=" * 80)
print("21. WINDOW FUNCTIONS")
print("=" * 80)

df = pd.DataFrame({
    'Date': pd.date_range('2024-01-01', periods=10),
    'Value': [10, 12, 11, 13, 15, 14, 16, 18, 17, 19]
})
print(f"Original DataFrame:\n{df}\n")

# Rolling mean
df['Rolling_Mean_3'] = df['Value'].rolling(window=3).mean()
print(f"Rolling mean (window=3):\n{df[['Value', 'Rolling_Mean_3']]}\n")

# Rolling sum
df['Rolling_Sum_3'] = df['Value'].rolling(window=3).sum()
print(f"Rolling sum (window=3):\n{df[['Value', 'Rolling_Sum_3']]}\n")

# Rolling std dev
df['Rolling_Std_3'] = df['Value'].rolling(window=3).std()
print(f"Rolling std dev:\n{df[['Value', 'Rolling_Std_3']]}\n")

# Expanding operations
df['Expanding_Mean'] = df['Value'].expanding().mean()
df['Expanding_Max'] = df['Value'].expanding().max()
print(f"Expanding operations:\n{df[['Value', 'Expanding_Mean', 'Expanding_Max']]}\n")

# Exponential weighted moving average
df['EWMA'] = df['Value'].ewm(span=3).mean()
print(f"Exponential weighted MA:\n{df[['Value', 'EWMA']]}\n")


# ==============================================================================
# 22. PRACTICAL EXAMPLE - Data Analysis Pipeline
# ==============================================================================

print("=" * 80)
print("22. PRACTICAL EXAMPLE - DATA ANALYSIS PIPELINE")
print("=" * 80)

# Create sample sales data
np.random.seed(42)
df_sales = pd.DataFrame({
    'Date': pd.date_range('2024-01-01', periods=30),
    'Product': np.random.choice(['A', 'B', 'C'], 30),
    'Quantity': np.random.randint(1, 20, 30),
    'Price': np.random.uniform(10, 100, 30),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], 30)
})

print(f"Raw sales data (first 10 rows):\n{df_sales.head(10)}\n")

# Add revenue column
df_sales['Revenue'] = df_sales['Quantity'] * df_sales['Price']

# Remove duplicates
df_sales = df_sales.drop_duplicates()

# Group by product
product_summary = df_sales.groupby('Product').agg({
    'Revenue': ['sum', 'mean'],
    'Quantity': 'sum'
}).round(2)
print(f"Summary by product:\n{product_summary}\n")

# Group by region and date
daily_regional = df_sales.groupby(['Date', 'Region'])['Revenue'].sum().unstack()
print(f"Daily revenue by region (first 5 days):\n{daily_regional.head()}\n")

# Time-based analysis
df_sales['Month'] = df_sales['Date'].dt.month
monthly_revenue = df_sales.groupby('Month')['Revenue'].sum()
print(f"Monthly revenue:\n{monthly_revenue}\n")

# Top performers
top_days = df_sales.groupby('Date')['Revenue'].sum().nlargest(5)
print(f"Top 5 revenue days:\n{top_days}\n")


# ==============================================================================
# 23. CATEGORICAL DATA - Working with Categories
# ==============================================================================

print("=" * 80)
print("23. CATEGORICAL DATA")
print("=" * 80)

df = pd.DataFrame({
    'Product': pd.Categorical(['A', 'B', 'A', 'C', 'B', 'A'], categories=['A', 'B', 'C'], ordered=False),
    'Sales': [100, 200, 150, 300, 250, 120]
})
print(f"DataFrame with categorical:\n{df}\n")
print(f"Product dtype: {df['Product'].dtype}\n")

# Category info
print(f"Categories: {df['Product'].cat.categories.tolist()}")
print(f"Number of categories: {df['Product'].cat.ncat}\n")

# Category operations
print(f"Value counts:\n{df['Product'].value_counts()}\n")

# Add new category
df['Product'] = df['Product'].cat.add_categories(['D'])
print(f"After adding category 'D': {df['Product'].cat.categories.tolist()}\n")

# Remove category
df['Product'] = df['Product'].cat.remove_categories(['D'])
print(f"After removing category 'D': {df['Product'].cat.categories.tolist()}\n")


# ==============================================================================
# 24. RECOMMENDATIONS AND BEST PRACTICES
# ==============================================================================

print("=" * 80)
print("24. PANDAS BEST PRACTICES")
print("=" * 80)

print("""
✓ BEST PRACTICES:
  1. Use .loc and .iloc consistently (not [] for ambiguous cases)
  2. Copy DataFrames when needed to avoid SettingWithCopyWarning
  3. Use .query() for complex conditions (more readable)
  4. Use .apply() with axis parameter correctly
  5. Use groupby() and transform() instead of loops
  6. Set dtypes correctly at import time
  7. Use .isin() for multiple value filtering
  8. Avoid chained indexing (use .loc or .assign())
  9. Use .at and .iat for single value access (faster)
  10. Use pd.concat() instead of deprecated .append()

✓ COMMON USE CASES:
  - Data cleaning and preprocessing
  - Time series analysis
  - Exploratory data analysis (EDA)
  - Data transformation and reshaping
  - Statistical analysis
  - Merging/joining multiple datasets
  - Data aggregation and reporting
  - Working with missing data
  - Feature engineering
  - Data validation

✗ AVOID:
  - Modifying DataFrame during iteration
  - Using loops instead of vectorized operations
  - Comparing arrays with == (use .equals() or element-wise)
  - Overwriting original data without explicit copy()
  - Using deprecated functions
  - Complex nested operations without clarity
  - Ignoring data types until problems appear
  - Creating unnecessary copies of large DataFrames

✓ PERFORMANCE TIPS:
  - Use appropriate dtypes (category for strings with few unique values)
  - Use query() for large DataFrames
  - Use groupby() with as_index=False for performance
  - Pre-filter data before expensive operations
  - Use NumPy where possible for numerical operations
  - Use .set_index() and .reset_index() strategically

✓ INTEGRATION WITH OTHER LIBRARIES:
  - NumPy: DataFrame uses NumPy arrays internally
  - Matplotlib/Seaborn: direct plotting from DataFrames
  - Scikit-learn: convert to NumPy arrays for ML models
  - SQLAlchemy: read/write from databases
  - Dask: for parallel/distributed computing
  - PySpark: for big data processing
  - Polars: faster alternative for large datasets
""")

print("\n" + "=" * 80)
print("END OF PANDAS COMPREHENSIVE GUIDE")
print("=" * 80)
