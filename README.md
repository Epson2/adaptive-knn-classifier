# adaptive-knn-classifier
Custom machine learning classifier using Mahalanobis distance and weighted KNN for binary classification.

## Features

- Data normalization (mean and standard deviation scaling)
- Mahalanobis distance metric for anisotropic data
- Adaptive neighborhood radius
- Weighted KNN voting
- Inner and outer neighbor regions for improved classification
- Custom implementation without ML frameworks

## How It Works

1. **Training Phase**
   - Normalize training data.
   - Extract positive class samples.
   - Compute covariance matrix for Mahalanobis distance.
   - Store training features and labels.

2. **Prediction Phase**
   - Normalize test data.
   - Compute Mahalanobis distance to training points.
   - Define adaptive radius based on positive cluster density.
   - Select neighbors from inner and outer regions.
   - Perform weighted voting to determine class prediction.

## Technologies

- Python
- NumPy
- Machine Learning Concepts
- Linear Algebra
- Distance Metrics
