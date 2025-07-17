# MovieLens Collaborative Filtering

This project implements and evaluates various recommendation algorithms using the MovieLens 100K dataset. The implementation includes comprehensive parameter tuning and performance evaluation across multiple collaborative filtering approaches.

## Dataset

The dataset used is the MovieLens 100K dataset, which contains 100,000 ratings (1-5) from 943 users on 1,682 movies. The data is split into training (80%) and testing (20%) sets using stratified sampling, with test items filtered to include only those with >30 ratings to ensure statistical significance.

## Methods Implemented

### 1. User Average Rating (Baseline)

- **Implementation**: Calculate average rating for each user and predict this value for all test items
- **Performance**: MAE: 0.8258, RMSE: 1.0311
- **Purpose**: Serves as a baseline for comparison

### 2. Item Average Rating (Baseline)

- **Implementation**: Calculate average rating for each item and predict this value for all test users
- **Performance**: MAE: 0.7916, RMSE: 1.0013
- **Purpose**: Another baseline that performs better than user average

### 3. User-based Collaborative Filtering (User KNN-CF)

- **Implementation**: Find similar users based on rating patterns and predict ratings using weighted averages
- **Similarity Metrics**: Pearson correlation, Cosine similarity, Euclidean distance
- **Parameter Tuning**:
  - **K-value optimization**: Tested k ∈ {5, 10, 20, 30, 50, 100}, optimal k = 30
  - **Similarity function**: Pearson correlation performed best
  - **Significance weighting**: Optimal weight = 25 (for co-rated items threshold)
- **Cross-validation**: 5-fold CV used for all parameter tuning
- **Performance**: MAE: 0.7754, RMSE: 0.9948
- **Key Features**:
  - Mean-centered ratings for Pearson correlation
  - Significance weighting based on number of co-rated items
  - Fallback to user mean when no similar users available

### 4. Item-based Collaborative Filtering (Item KNN-CF)

- **Implementation**: Find similar items based on user rating patterns and predict using item similarities
- **Similarity Metrics**: Adjusted Cosine similarity, Pearson correlation, Euclidean distance
- **Parameter Tuning**:
  - **K-value optimization**: Tested k ∈ {5, 10, 20, 30, 50, 100}, optimal k = 20
  - **Similarity function**: Adjusted Cosine similarity performed best
  - **Significance weighting**: Optimal weight = 25
- **Cross-validation**: 5-fold CV used for all parameter tuning
- **Performance**: MAE: 0.7869, RMSE: 1.0250
- **Key Features**:
  - User-mean centered ratings for adjusted cosine similarity
  - Significance weighting based on number of common users
  - Fallback to item mean when no similar items available

### 5. Hybrid Collaborative Filtering

- **Implementation**: Linear combination of user-based and item-based predictions
- **Formula**: `Hybrid_prediction = λ × User_prediction + (1-λ) × Item_prediction`
- **Lambda optimization**: Tested λ ∈ {0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0}
- **Optimal lambda**: 0.6 (favoring user-based predictions)
- **Performance**: MAE: 0.752, RMSE: 0.961 (best overall performance)
- **Cross-validation**: 5-fold CV for lambda optimization

## Technical Implementation Details

### Similarity Computation

- **Pearson Correlation**: Mean-centered ratings with significance weighting
- **Cosine Similarity**: Dot product normalized by vector magnitudes
- **Euclidean Distance**: Converted to similarity using inverse distance transformation
- **Adjusted Cosine**: User-mean centered for item-based CF

### Evaluation Framework

- **Metrics**: Mean Absolute Error (MAE) and Root Mean Square Error (RMSE)
- **Cross-validation**: 5-fold stratified CV for robust parameter tuning
- **Test filtering**: Only items with >30 ratings included in test set

### Performance Optimization

- **Vectorized operations**: NumPy arrays for efficient computation
- **Memory management**: Sparse matrix representations where applicable
- **Caching**: Pre-computed similarity matrices for hybrid method

## Results Summary

| Method        | MAE       | RMSE      | Improvement over Baseline |
| ------------- | --------- | --------- | ------------------------- |
| User Average  | 0.8258    | 1.0311    | Baseline                  |
| Item Average  | 0.7916    | 1.0013    | 2.9% RMSE improvement     |
| User-based CF | 0.7754    | 0.9948    | 3.5% RMSE improvement     |
| Item-based CF | 0.7869    | 1.0250    | 0.6% RMSE improvement     |
| **Hybrid CF** | **0.752** | **0.961** | **6.8% RMSE improvement** |

## Visualizations Generated

1. **K-value tuning plots**: RMSE vs number of neighbors for both user and item-based CF
2. **Similarity comparison**: Performance comparison across different similarity metrics
3. **Weight tuning**: Significance weight optimization results
4. **Lambda optimization**: Hybrid method parameter tuning visualization
5. **Comprehensive comparison**: Final performance comparison across all methods

## Key Insights

1. **Hybrid approach superiority**: Combining user and item-based methods yields the best performance
2. **Parameter sensitivity**: Proper tuning of k, similarity function, and weights significantly impacts performance
3. **Method complementarity**: User-based CF performs better than item-based CF in this dataset
4. **Baseline importance**: Simple averaging methods provide strong baselines that are hard to beat significantly

## Requirements

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib (for visualizations)

## Files Structure

- `Assignment3_framework_2025.ipynb`: Main implementation notebook
- `ml-100k/`: MovieLens dataset files
- `*.png`: Generated visualization plots
- `README.md`: This documentation file
