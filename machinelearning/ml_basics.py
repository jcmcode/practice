"""
MACHINE LEARNING BASICS - COMPREHENSIVE CHEAT SHEET
===================================================
This file covers fundamental ML concepts, algorithms, and practical implementations.
"""

# ============================================================================
# 1. IMPORTING ESSENTIAL LIBRARIES
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             mean_squared_error, mean_absolute_error, r2_score)

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================

# --- Loading Data ---
# CSV files are the most common format
# df = pd.read_csv('data.csv')
# df = pd.read_excel('data.xlsx')
# df = pd.read_json('data.json')

# --- Exploratory Data Analysis (EDA) ---
# df.head()          # First 5 rows
# df.tail()          # Last 5 rows
# df.info()          # Data types and non-null counts
# df.describe()      # Statistical summary
# df.shape           # (rows, columns)
# df.columns         # Column names
# df.isnull().sum()  # Count missing values per column
# df.duplicated().sum()  # Count duplicate rows

# --- Handling Missing Values ---
def handle_missing_values_example():
    """Different strategies for handling missing data"""
    data = {'A': [1, 2, np.nan, 4], 'B': [5, np.nan, 7, 8], 'C': [9, 10, 11, 12]}
    df = pd.DataFrame(data)
    
    # Drop rows with any missing values
    df_dropped = df.dropna()
    
    # Drop columns with any missing values
    df_dropped_cols = df.dropna(axis=1)
    
    # Fill with a constant value
    df_filled = df.fillna(0)
    
    # Fill with mean (for numerical columns)
    df_mean_filled = df.fillna(df.mean())
    
    # Fill with median
    df_median_filled = df.fillna(df.median())
    
    # Forward fill (use previous value)
    df_ffill = df.fillna(method='ffill')
    
    # Backward fill (use next value)
    df_bfill = df.fillna(method='bfill')
    
    return df_filled

# --- Feature Scaling ---
def feature_scaling_example():
    """Scaling features to same range improves model performance"""
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    
    # Standardization (z-score normalization): mean=0, std=1
    # Formula: (x - mean) / std
    scaler_standard = StandardScaler()
    data_standardized = scaler_standard.fit_transform(data)
    print("Standardized:", data_standardized)
    
    # Min-Max Scaling: scales to range [0, 1]
    # Formula: (x - min) / (max - min)
    scaler_minmax = MinMaxScaler()
    data_normalized = scaler_minmax.fit_transform(data)
    print("Normalized:", data_normalized)
    
    return data_standardized, data_normalized

# --- Encoding Categorical Variables ---
def encoding_example():
    """Converting categorical data to numerical format"""
    
    # Label Encoding: Convert categories to numbers (0, 1, 2, ...)
    # Use for ordinal data (Low, Medium, High)
    categories = ['Low', 'Medium', 'High', 'Low', 'High']
    le = LabelEncoder()
    encoded = le.fit_transform(categories)
    print("Label Encoded:", encoded)  # [1, 2, 0, 1, 0]
    
    # One-Hot Encoding: Create binary columns for each category
    # Use for nominal data (Red, Blue, Green)
    colors = np.array(['Red', 'Blue', 'Green', 'Red']).reshape(-1, 1)
    ohe = OneHotEncoder(sparse_output=False)
    one_hot = ohe.fit_transform(colors)
    print("One-Hot Encoded:\n", one_hot)
    # [[0, 1, 0],  # Red
    #  [1, 0, 0],  # Blue
    #  [0, 0, 1],  # Green
    #  [0, 1, 0]]  # Red
    
    return encoded, one_hot

# --- Train-Test Split ---
def train_test_split_example():
    """Split data into training and testing sets"""
    # X = features, y = target variable
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([0, 0, 1, 1, 1])
    
    # 80% training, 20% testing
    # random_state ensures reproducibility
    # stratify maintains class proportions in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")
    return X_train, X_test, y_train, y_test

# ============================================================================
# 3. SUPERVISED LEARNING - CLASSIFICATION
# ============================================================================

# --- Logistic Regression ---
# Binary classification using logistic function (sigmoid)
# Outputs probability between 0 and 1
# Good for: Linear decision boundaries, probability estimates
def logistic_regression_example():
    """Binary classification with Logistic Regression"""
    from sklearn.linear_model import LogisticRegression
    
    # Sample data
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    # Create and train model
    model = LogisticRegression()
    model.fit(X, y)
    
    # Predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)  # Returns [prob_class_0, prob_class_1]
    
    # Coefficients (weights) and intercept
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    
    return model, predictions

# --- Decision Trees ---
# Tree-based model that splits data based on feature values
# Good for: Non-linear patterns, interpretability, handling both numerical and categorical
def decision_tree_classifier_example():
    """Classification using Decision Trees"""
    from sklearn.tree import DecisionTreeClassifier
    
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    # Parameters:
    # max_depth: Maximum tree depth (prevents overfitting)
    # min_samples_split: Minimum samples required to split a node
    # min_samples_leaf: Minimum samples required in a leaf node
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)
    
    # Feature importance
    print(f"Feature Importances: {model.feature_importances_}")
    
    return model

# --- Random Forest ---
# Ensemble of decision trees (bagging method)
# Reduces overfitting, more robust than single decision tree
def random_forest_example():
    """Ensemble method: Random Forest"""
    from sklearn.ensemble import RandomForestClassifier
    
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]])
    y = np.array([0, 0, 0, 1, 1, 1, 1])
    
    # Parameters:
    # n_estimators: Number of trees in the forest
    # max_depth: Maximum depth of each tree
    # min_samples_split: Minimum samples to split a node
    model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X, y)
    
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    return model

# --- K-Nearest Neighbors (KNN) ---
# Classifies based on majority vote of k nearest neighbors
# Good for: Small datasets, multi-class problems
def knn_example():
    """K-Nearest Neighbors classification"""
    from sklearn.neighbors import KNeighborsClassifier
    
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    # Parameters:
    # n_neighbors: Number of neighbors to consider (typically odd number)
    # weights: 'uniform' (all equal) or 'distance' (closer points have more influence)
    # metric: Distance metric ('euclidean', 'manhattan', etc.)
    model = KNeighborsClassifier(n_neighbors=3, weights='uniform')
    model.fit(X, y)
    
    predictions = model.predict(X)
    
    return model

# --- Support Vector Machine (SVM) ---
# Finds optimal hyperplane that separates classes
# Good for: High-dimensional data, clear margin of separation
def svm_example():
    """Support Vector Machine for classification"""
    from sklearn.svm import SVC
    
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    # Parameters:
    # kernel: 'linear', 'rbf' (radial basis function), 'poly', 'sigmoid'
    # C: Regularization parameter (smaller = more regularization)
    # gamma: Kernel coefficient for 'rbf', 'poly', 'sigmoid'
    model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    model.fit(X, y)
    
    predictions = model.predict(X)
    
    return model

# --- Naive Bayes ---
# Probabilistic classifier based on Bayes' theorem
# Assumes feature independence
# Good for: Text classification, spam detection
def naive_bayes_example():
    """Naive Bayes classification"""
    from sklearn.naive_bayes import GaussianNB
    
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    # GaussianNB: Assumes features follow a normal distribution
    # MultinomialNB: For discrete counts (e.g., word counts)
    # BernoulliNB: For binary/boolean features
    model = GaussianNB()
    model.fit(X, y)
    
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    return model

# --- Gradient Boosting ---
# Ensemble method that builds trees sequentially
# Each tree corrects errors of previous trees
def gradient_boosting_example():
    """Gradient Boosting classification"""
    from sklearn.ensemble import GradientBoostingClassifier
    
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    # Parameters:
    # n_estimators: Number of boosting stages
    # learning_rate: Shrinks contribution of each tree
    # max_depth: Maximum depth of individual trees
    model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    )
    model.fit(X, y)
    
    return model

# ============================================================================
# 4. SUPERVISED LEARNING - REGRESSION
# ============================================================================

# --- Linear Regression ---
# Models linear relationship between features and target
# Formula: y = mx + b (or y = w1*x1 + w2*x2 + ... + b)
def linear_regression_example():
    """Simple and Multiple Linear Regression"""
    from sklearn.linear_model import LinearRegression
    
    # Sample data
    X = np.array([[1], [2], [3], [4], [5]])  # Features
    y = np.array([2, 4, 6, 8, 10])           # Target
    
    # Create and train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predictions
    predictions = model.predict(X)
    
    # Model parameters
    print(f"Coefficient (slope): {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    
    # Multiple features
    X_multi = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y_multi = np.array([5, 8, 11, 14, 17])
    
    model_multi = LinearRegression()
    model_multi.fit(X_multi, y_multi)
    
    return model, predictions

# --- Ridge Regression (L2 Regularization) ---
# Linear regression with L2 penalty to prevent overfitting
# Adds penalty proportional to square of coefficient magnitudes
def ridge_regression_example():
    """Ridge Regression with L2 regularization"""
    from sklearn.linear_model import Ridge
    
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([5, 8, 11, 14, 17])
    
    # alpha: Regularization strength (higher = more regularization)
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    
    return model

# --- Lasso Regression (L1 Regularization) ---
# Linear regression with L1 penalty
# Can perform feature selection (sets some coefficients to zero)
def lasso_regression_example():
    """Lasso Regression with L1 regularization"""
    from sklearn.linear_model import Lasso
    
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([5, 8, 11, 14, 17])
    
    # alpha: Regularization strength
    model = Lasso(alpha=0.1)
    model.fit(X, y)
    
    # Check which features were selected (non-zero coefficients)
    print(f"Coefficients: {model.coef_}")
    
    return model

# --- Polynomial Regression ---
# Models non-linear relationships using polynomial features
def polynomial_regression_example():
    """Polynomial Regression for non-linear patterns"""
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 4, 9, 16, 25])  # y = x^2
    
    # Create polynomial features (e.g., x, x^2, x^3)
    # degree=2: includes x and x^2
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Fit linear regression on polynomial features
    model = LinearRegression()
    model.fit(X_poly, y)
    
    predictions = model.predict(X_poly)
    
    return model, X_poly

# --- Decision Tree Regressor ---
# Tree-based regression
def decision_tree_regressor_example():
    """Decision Tree for regression"""
    from sklearn.tree import DecisionTreeRegressor
    
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])
    
    model = DecisionTreeRegressor(max_depth=3, random_state=42)
    model.fit(X, y)
    
    return model

# --- Random Forest Regressor ---
def random_forest_regressor_example():
    """Random Forest for regression"""
    from sklearn.ensemble import RandomForestRegressor
    
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])
    
    model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X, y)
    
    return model

# ============================================================================
# 5. UNSUPERVISED LEARNING
# ============================================================================

# --- K-Means Clustering ---
# Groups data into k clusters based on similarity
# Minimizes within-cluster variance
def kmeans_example():
    """K-Means Clustering"""
    from sklearn.cluster import KMeans
    
    # Sample data
    X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    
    # Parameters:
    # n_clusters: Number of clusters
    # init: Initialization method ('k-means++' is default and recommended)
    # n_init: Number of times algorithm runs with different seeds
    model = KMeans(n_clusters=2, random_state=42)
    model.fit(X)
    
    # Cluster labels for each point
    labels = model.labels_
    
    # Cluster centers
    centers = model.cluster_centers_
    
    # Predict cluster for new data
    new_point = np.array([[0, 0]])
    cluster = model.predict(new_point)
    
    print(f"Cluster labels: {labels}")
    print(f"Cluster centers:\n{centers}")
    
    return model

# --- Hierarchical Clustering ---
# Creates tree of clusters (dendrogram)
# Agglomerative (bottom-up) or divisive (top-down)
def hierarchical_clustering_example():
    """Hierarchical Clustering"""
    from sklearn.cluster import AgglomerativeClustering
    
    X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    
    # Parameters:
    # n_clusters: Number of clusters to find
    # linkage: 'ward', 'complete', 'average', 'single'
    model = AgglomerativeClustering(n_clusters=2, linkage='ward')
    labels = model.fit_predict(X)
    
    print(f"Cluster labels: {labels}")
    
    return labels

# --- DBSCAN (Density-Based Clustering) ---
# Groups points that are closely packed together
# Marks outliers as noise
def dbscan_example():
    """DBSCAN - Density-Based Clustering"""
    from sklearn.cluster import DBSCAN
    
    X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
    
    # Parameters:
    # eps: Maximum distance between two samples to be neighbors
    # min_samples: Minimum points in neighborhood to form cluster
    model = DBSCAN(eps=3, min_samples=2)
    labels = model.fit_predict(X)
    
    # -1 indicates outliers/noise
    print(f"Cluster labels: {labels}")
    
    return labels

# --- Principal Component Analysis (PCA) ---
# Dimensionality reduction technique
# Projects data onto principal components (directions of maximum variance)
def pca_example():
    """PCA for dimensionality reduction"""
    from sklearn.decomposition import PCA
    
    # Sample data with 4 features
    X = np.array([[1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12], 
                  [4, 8, 12, 16], [5, 10, 15, 20]])
    
    # Reduce to 2 components
    # n_components: Number of components to keep
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    
    # Explained variance ratio: proportion of variance explained by each component
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_)}")
    
    # Can also specify variance to retain
    pca_variance = PCA(n_components=0.95)  # Keep 95% of variance
    
    return X_reduced

# ============================================================================
# 6. MODEL EVALUATION METRICS
# ============================================================================

# --- Classification Metrics ---
def classification_metrics_example():
    """Common metrics for classification problems"""
    
    # True labels and predictions
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
    
    # Accuracy: (TP + TN) / Total
    # Proportion of correct predictions
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc}")
    
    # Precision: TP / (TP + FP)
    # Of all predicted positives, how many are actually positive?
    # Use when false positives are costly
    prec = precision_score(y_true, y_pred)
    print(f"Precision: {prec}")
    
    # Recall (Sensitivity): TP / (TP + FN)
    # Of all actual positives, how many did we predict?
    # Use when false negatives are costly
    rec = recall_score(y_true, y_pred)
    print(f"Recall: {rec}")
    
    # F1-Score: Harmonic mean of precision and recall
    # F1 = 2 * (precision * recall) / (precision + recall)
    # Balances precision and recall
    f1 = f1_score(y_true, y_pred)
    print(f"F1-Score: {f1}")
    
    # Confusion Matrix:
    # [[TN, FP],
    #  [FN, TP]]
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    
    # Classification Report: All metrics together
    report = classification_report(y_true, y_pred)
    print("Classification Report:")
    print(report)
    
    return acc, prec, rec, f1

# --- Regression Metrics ---
def regression_metrics_example():
    """Common metrics for regression problems"""
    
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    
    # Mean Absolute Error (MAE): Average absolute difference
    # MAE = (1/n) * Σ|y_true - y_pred|
    # Easy to interpret, same units as target
    mae = mean_absolute_error(y_true, y_pred)
    print(f"MAE: {mae}")
    
    # Mean Squared Error (MSE): Average squared difference
    # MSE = (1/n) * Σ(y_true - y_pred)^2
    # Penalizes large errors more heavily
    mse = mean_squared_error(y_true, y_pred)
    print(f"MSE: {mse}")
    
    # Root Mean Squared Error (RMSE): Square root of MSE
    # Same units as target
    rmse = np.sqrt(mse)
    print(f"RMSE: {rmse}")
    
    # R² Score (Coefficient of Determination): Proportion of variance explained
    # R² = 1 - (SS_res / SS_tot)
    # Range: (-∞, 1], where 1 is perfect prediction
    r2 = r2_score(y_true, y_pred)
    print(f"R² Score: {r2}")
    
    return mae, mse, rmse, r2

# ============================================================================
# 7. CROSS-VALIDATION
# ============================================================================

# --- K-Fold Cross-Validation ---
# Splits data into k folds, trains on k-1 folds, validates on remaining fold
# Repeats k times, each fold used as validation once
def cross_validation_example():
    """K-Fold Cross-Validation"""
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.linear_model import LogisticRegression
    
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], 
                  [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    model = LogisticRegression()
    
    # Simple cross-validation
    # cv: Number of folds
    scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-validation scores: {scores}")
    print(f"Mean CV score: {scores.mean()}")
    print(f"Std CV score: {scores.std()}")
    
    # Custom K-Fold
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores_custom = cross_val_score(model, X, y, cv=kfold)
    
    return scores

# --- Stratified K-Fold ---
# Maintains class proportions in each fold
# Important for imbalanced datasets
def stratified_kfold_example():
    """Stratified K-Fold Cross-Validation"""
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.linear_model import LogisticRegression
    
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    model = LogisticRegression()
    skfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skfold)
    
    return scores

# ============================================================================
# 8. HYPERPARAMETER TUNING
# ============================================================================

# --- Grid Search ---
# Exhaustively searches through specified parameter grid
def grid_search_example():
    """Grid Search for hyperparameter tuning"""
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], 
                  [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10]
    }
    
    model = RandomForestClassifier(random_state=42)
    
    # GridSearchCV trains model with every combination
    # cv: Number of cross-validation folds
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X, y)
    
    # Best parameters and score
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")
    
    # Use best model
    best_model = grid_search.best_estimator_
    
    return best_model

# --- Random Search ---
# Randomly samples from parameter distributions
# More efficient than grid search for large parameter spaces
def random_search_example():
    """Random Search for hyperparameter tuning"""
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from scipy.stats import randint
    
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], 
                  [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    # Define parameter distributions
    param_dist = {
        'n_estimators': randint(50, 200),
        'max_depth': randint(3, 10),
        'min_samples_split': randint(2, 20)
    }
    
    model = RandomForestClassifier(random_state=42)
    
    # n_iter: Number of parameter settings sampled
    random_search = RandomizedSearchCV(
        model, param_dist, n_iter=10, cv=3, random_state=42
    )
    random_search.fit(X, y)
    
    print(f"Best parameters: {random_search.best_params_}")
    
    return random_search.best_estimator_

# ============================================================================
# 9. FEATURE ENGINEERING
# ============================================================================

# --- Feature Selection ---
def feature_selection_example():
    """Selecting most important features"""
    from sklearn.feature_selection import SelectKBest, f_classif, RFE
    from sklearn.ensemble import RandomForestClassifier
    
    # Sample data with 5 features
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    # SelectKBest: Select k highest scoring features
    # f_classif: ANOVA F-value for classification
    selector = SelectKBest(score_func=f_classif, k=3)
    X_new = selector.fit_transform(X, y)
    print(f"Selected features: {selector.get_support()}")
    
    # Recursive Feature Elimination (RFE)
    # Recursively removes least important features
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    rfe = RFE(estimator=model, n_features_to_select=3)
    X_rfe = rfe.fit_transform(X, y)
    print(f"RFE selected features: {rfe.support_}")
    
    return X_new, X_rfe

# --- Feature Importance ---
def feature_importance_example():
    """Getting feature importance from tree-based models"""
    from sklearn.ensemble import RandomForestClassifier
    
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    y = np.array([0, 0, 1, 1])
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Feature importances (sum to 1)
    importances = model.feature_importances_
    print(f"Feature importances: {importances}")
    
    # Can sort and select top features
    indices = np.argsort(importances)[::-1]
    print(f"Feature ranking: {indices}")
    
    return importances

# ============================================================================
# 10. HANDLING IMBALANCED DATA
# ============================================================================

# --- Class Weights ---
def class_weight_example():
    """Handle imbalanced data with class weights"""
    from sklearn.linear_model import LogisticRegression
    
    # Imbalanced dataset: 90 class 0, 10 class 1
    X = np.random.rand(100, 2)
    y = np.array([0] * 90 + [1] * 10)
    
    # class_weight='balanced': automatically adjusts weights
    # inversely proportional to class frequencies
    model = LogisticRegression(class_weight='balanced')
    model.fit(X, y)
    
    # Can also specify custom weights
    model_custom = LogisticRegression(class_weight={0: 1, 1: 9})
    model_custom.fit(X, y)
    
    return model

# --- Resampling Techniques ---
# Note: Requires imbalanced-learn library (pip install imbalanced-learn)
"""
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def resampling_example():
    # Imbalanced dataset
    X = np.random.rand(100, 2)
    y = np.array([0] * 90 + [1] * 10)
    
    # Random Oversampling: Duplicate minority class samples
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    
    # Random Undersampling: Remove majority class samples
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    
    # SMOTE: Synthetic Minority Over-sampling Technique
    # Creates synthetic samples for minority class
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    return X_resampled, y_resampled
"""

# ============================================================================
# 11. PIPELINES
# ============================================================================

# --- Creating Pipelines ---
# Chains multiple steps (preprocessing + model) together
def pipeline_example():
    """Creating ML pipelines"""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    # Pipeline automatically applies steps in order
    # Each step is a tuple: ('name', transformer/estimator)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),           # Step 1: Scale features
        ('classifier', LogisticRegression())    # Step 2: Train model
    ])
    
    # Fit applies all transformations and trains model
    pipeline.fit(X, y)
    
    # Predict applies transformations then predicts
    predictions = pipeline.predict(X)
    
    # Can use in cross-validation
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(pipeline, X, y, cv=3)
    
    return pipeline

# --- Pipeline with Grid Search ---
def pipeline_grid_search_example():
    """Combining pipelines with grid search"""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    
    X = np.random.rand(50, 2)
    y = np.random.randint(0, 2, 50)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC())
    ])
    
    # Parameter names use 'step_name__parameter_name' format
    param_grid = {
        'svm__C': [0.1, 1, 10],
        'svm__kernel': ['linear', 'rbf'],
        'svm__gamma': ['scale', 'auto']
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=3)
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    
    return grid_search.best_estimator_

# ============================================================================
# 12. MODEL PERSISTENCE (SAVING AND LOADING)
# ============================================================================

# --- Using Pickle ---
def model_persistence_pickle_example():
    """Save and load models using pickle"""
    import pickle
    from sklearn.linear_model import LogisticRegression
    
    X = np.array([[1, 2], [2, 3], [3, 4]])
    y = np.array([0, 0, 1])
    
    model = LogisticRegression()
    model.fit(X, y)
    
    # Save model
    # with open('model.pkl', 'wb') as f:
    #     pickle.dump(model, f)
    
    # Load model
    # with open('model.pkl', 'rb') as f:
    #     loaded_model = pickle.load(f)
    
    # predictions = loaded_model.predict(X)
    
    return model

# --- Using Joblib (recommended for large models) ---
def model_persistence_joblib_example():
    """Save and load models using joblib"""
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    
    X = np.array([[1, 2], [2, 3], [3, 4]])
    y = np.array([0, 0, 1])
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    
    # Save model
    # joblib.dump(model, 'model.joblib')
    
    # Load model
    # loaded_model = joblib.load('model.joblib')
    
    # predictions = loaded_model.predict(X)
    
    return model

# ============================================================================
# 13. COMMON ML WORKFLOW
# ============================================================================

def complete_ml_workflow_example():
    """End-to-end machine learning workflow"""
    
    # Step 1: Load and explore data
    # df = pd.read_csv('data.csv')
    # print(df.head())
    # print(df.info())
    # print(df.describe())
    
    # Step 2: Handle missing values
    # df = df.dropna()  # or df.fillna()
    
    # Step 3: Encode categorical variables
    # df = pd.get_dummies(df, columns=['categorical_column'])
    
    # Step 4: Split features and target
    # X = df.drop('target_column', axis=1)
    # y = df['target_column']
    
    # Step 5: Train-test split
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42
    # )
    
    # Step 6: Feature scaling
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
    
    # Step 7: Train model
    # model = RandomForestClassifier(n_estimators=100, random_state=42)
    # model.fit(X_train_scaled, y_train)
    
    # Step 8: Make predictions
    # y_pred = model.predict(X_test_scaled)
    
    # Step 9: Evaluate model
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy}")
    # print(classification_report(y_test, y_pred))
    
    # Step 10: Hyperparameter tuning (optional)
    # param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}
    # grid_search = GridSearchCV(model, param_grid, cv=5)
    # grid_search.fit(X_train_scaled, y_train)
    
    # Step 11: Save model
    # import joblib
    # joblib.dump(model, 'final_model.joblib')
    
    pass

# ============================================================================
# 14. IMPORTANT CONCEPTS AND TIPS
# ============================================================================

"""
BIAS-VARIANCE TRADEOFF
----------------------
- Bias: Error from overly simplistic assumptions (underfitting)
- Variance: Error from sensitivity to training data fluctuations (overfitting)
- Goal: Find sweet spot between bias and variance

High Bias (Underfitting):
- Model too simple
- Poor performance on both training and test data
- Solution: More complex model, more features, less regularization

High Variance (Overfitting):
- Model too complex
- Great on training data, poor on test data
- Solution: More data, regularization, simpler model, cross-validation

OVERFITTING VS UNDERFITTING
----------------------------
Overfitting:
- Model memorizes training data instead of learning patterns
- Training accuracy >> Test accuracy
- Solutions:
  * More training data
  * Regularization (L1, L2, dropout)
  * Simpler model
  * Cross-validation
  * Early stopping
  * Feature selection

Underfitting:
- Model too simple to capture patterns
- Both training and test accuracy are low
- Solutions:
  * More complex model
  * More features
  * Less regularization
  * Train longer

REGULARIZATION
--------------
Prevents overfitting by penalizing large coefficients

L1 (Lasso): Adds |coefficient| penalty
- Can set coefficients to exactly zero (feature selection)
- Use when you have many features and want to select important ones

L2 (Ridge): Adds coefficient² penalty
- Shrinks coefficients but rarely makes them zero
- Use when you want to keep all features but prevent overfitting

Elastic Net: Combines L1 + L2
- Balance between feature selection and coefficient shrinkage

FEATURE SCALING - WHEN TO USE
------------------------------
Models that REQUIRE scaling:
- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)
- Neural Networks
- Linear/Logistic Regression (when using regularization)
- Principal Component Analysis (PCA)
- K-Means Clustering

Models that DON'T require scaling:
- Decision Trees
- Random Forest
- Gradient Boosting
- XGBoost, LightGBM

TRAIN-TEST SPLIT BEST PRACTICES
--------------------------------
- Typical split: 70-30, 80-20, or 75-25 (train-test)
- Use stratify for classification to maintain class proportions
- Set random_state for reproducibility
- For small datasets, use cross-validation instead

CHOOSING THE RIGHT ALGORITHM
-----------------------------
Classification:
- Logistic Regression: Baseline, interpretable, fast
- Random Forest: Good default choice, handles non-linearity
- SVM: High-dimensional data, clear separation
- Naive Bayes: Text classification, fast
- Neural Networks: Complex patterns, large datasets

Regression:
- Linear Regression: Baseline, interpretable, linear relationships
- Ridge/Lasso: When you have many features or multicollinearity
- Random Forest: Non-linear patterns, robust
- Gradient Boosting: Often best performance

Clustering:
- K-Means: Known number of clusters, spherical clusters
- DBSCAN: Unknown number of clusters, arbitrary shapes, noise
- Hierarchical: Want dendrogram, small datasets

EVALUATION METRICS - WHEN TO USE
---------------------------------
Classification:
- Accuracy: Balanced classes, all errors equally important
- Precision: When false positives are costly (spam detection)
- Recall: When false negatives are costly (disease detection)
- F1-Score: Imbalanced classes, balance precision and recall
- ROC-AUC: Compare models, probability ranking

Regression:
- MAE: Easy to interpret, same units as target
- MSE/RMSE: Penalize large errors more
- R²: Proportion of variance explained

CROSS-VALIDATION BEST PRACTICES
--------------------------------
- K-Fold: Standard choice, k=5 or k=10
- Stratified K-Fold: Classification with imbalanced classes
- Leave-One-Out: Very small datasets (computationally expensive)
- Time Series: Use TimeSeriesSplit (maintains temporal order)

HYPERPARAMETER TUNING TIPS
---------------------------
- Start with Grid Search for small parameter spaces
- Use Random Search for large parameter spaces (more efficient)
- Use cross-validation to evaluate each combination
- Don't tune on test set (causes overfitting)
- Common parameters to tune:
  * Tree models: n_estimators, max_depth, min_samples_split
  * SVM: C, kernel, gamma
  * KNN: n_neighbors, weights
  * Neural Networks: learning_rate, layers, neurons

DATA LEAKAGE - AVOID AT ALL COSTS
----------------------------------
Data leakage = Test data information leaks into training process

Common causes:
1. Scaling before splitting
   Wrong: scale entire dataset, then split
   Right: split first, then scale training and test separately

2. Feature engineering using test data
   Wrong: create features using statistics from entire dataset
   Right: create features using only training data

3. Target leakage: Features that include information about target
   Example: Using "total_purchases_next_month" to predict churn

MODEL SELECTION CHECKLIST
--------------------------
1. Understand your problem (classification, regression, clustering)
2. Explore and clean your data
3. Split data (train-test or cross-validation)
4. Start with simple baseline model
5. Try 2-3 different algorithms
6. Evaluate using appropriate metrics
7. Tune hyperparameters of best model(s)
8. Validate on test set (only once!)
9. Check for overfitting/underfitting
10. Save final model

COMMON MISTAKES TO AVOID
-------------------------
1. Not splitting data before preprocessing
2. Using test set multiple times during development
3. Ignoring class imbalance
4. Not scaling features when needed
5. Not handling missing values properly
6. Overfitting to training data
7. Not using cross-validation
8. Using wrong evaluation metric
9. Not checking for data leakage
10. Not setting random_state for reproducibility
"""

# ============================================================================
# 15. QUICK REFERENCE - SKLEARN IMPORTS
# ============================================================================

"""
CLASSIFICATION MODELS
---------------------
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

REGRESSION MODELS
-----------------
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

CLUSTERING MODELS
-----------------
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

DIMENSIONALITY REDUCTION
------------------------
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

PREPROCESSING
-------------
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, 
                                   LabelEncoder, OneHotEncoder)
from sklearn.impute import SimpleImputer

MODEL SELECTION
---------------
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     KFold, StratifiedKFold,
                                     GridSearchCV, RandomizedSearchCV)

METRICS
-------
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             mean_squared_error, mean_absolute_error, r2_score,
                             roc_auc_score, roc_curve)

FEATURE SELECTION
-----------------
from sklearn.feature_selection import SelectKBest, RFE, f_classif, chi2

PIPELINE
--------
from sklearn.pipeline import Pipeline, make_pipeline

MODEL PERSISTENCE
-----------------
import pickle
import joblib
"""

# ============================================================================
# END OF MACHINE LEARNING CHEAT SHEET
# ============================================================================

if __name__ == "__main__":
    print("Machine Learning Basics Cheat Sheet")
    print("=" * 50)
    print("\nThis file contains comprehensive examples of:")
    print("- Data preprocessing techniques")
    print("- Supervised learning (classification & regression)")
    print("- Unsupervised learning (clustering & dimensionality reduction)")
    print("- Model evaluation metrics")
    print("- Cross-validation")
    print("- Hyperparameter tuning")
    print("- Feature engineering")
    print("- Pipelines")
    print("- Model persistence")
    print("\nUncomment and run individual functions to see examples in action!")
