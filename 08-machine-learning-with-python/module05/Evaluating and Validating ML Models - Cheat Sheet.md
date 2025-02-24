**Method Name:** classification_report
**Description:** Generates a report with precision, recall, F1-score, and support for each class in classification problems. Useful for model evaluation.
**Hyper-Parameters:**
- `target_names`: List of labels to include in the report.
**Pros:** Provides a comprehensive evaluation of classification models.
**Limitations:** May not provide enough insight for imbalanced datasets.

```python
from sklearn.metrics import classification_report
# y_true: True labels
# y_pred: Predicted labels
# target_names: List of target class names
report = classification_report(y_true, y_pred, target_names=["class1", "class2"])
```

---

**Method Name:** confusion_matrix
**Description:** Computes a confusion matrix to evaluate the classification performance, showing counts of true positives, false positives, true negatives, and false negatives.
**Hyper-Parameters:**
- `labels`: List of class labels to include.
**Pros:** Essential for understanding classification errors.
**Limitations:** Doesn't give insights into prediction probabilities.

```python
from sklearn.metrics import confusion_matrix
# y_true: True labels
# y_pred: Predicted labels
conf_matrix = confusion_matrix(y_true, y_pred)
```

---

**Method Name:** mean_squared_error
**Description:** Calculates the mean squared error (MSE), a common metric for regression models. Lower values indicate better performance.
**Hyper-Parameters:**
- `sample_weight`: Weights to apply to each sample.
**Pros:** Simple and widely used metric.
**Limitations:** Sensitive to outliers, as large errors are squared.

```python
from sklearn.metrics import mean_squared_error
# y_true: True values
# y_pred: Predicted values
# sample_weight: Optional, array of sample weights
mse = mean_squared_error(y_true, y_pred)
```

---

**Method Name:** root_mean_squared_error
**Description:** Calculates the root mean squared error (RMSE), which is the square root of the MSE. RMSE gives more interpretable results as it is in the same units as the target.
**Hyper-Parameters:**
- `sample_weight`: Weights to apply to each sample.
**Pros:** More interpretable than MSE.
**Limitations:** Like MSE, it can be sensitive to large errors and outliers.

```python
from sklearn.metrics import root_mean_squared_error
# y_true: True values
# y_pred: Predicted values
# sample_weight: Optional, array of sample weights
rmse = root_mean_squared_error(y_true, y_pred)
```

---

**Method Name:** mean_absolute_error
**Description:** Measures the average magnitude of errors in predictions, without considering their direction. Useful for understanding the average error size.
**Hyper-Parameters:**
- `sample_weight`: Optional sample weights.
**Pros:** Less sensitive to outliers compared to MSE.
**Limitations:** Does not penalize large errors as much as MSE or RMSE.

```python
from sklearn.metrics import mean_absolute_error
# y_true: True values
# y_pred: Predicted values
mae = mean_absolute_error(y_true, y_pred)
```

---

**Method Name:** r2_score
**Description:** Computes the coefficient of determination (R²), which represents the proportion of variance explained by the model. A higher value indicates a better fit.
**Pros:** Provides a clear indication of model performance.
**Limitations:** Doesn't always represent model quality, especially for non-linear models.

```python
from sklearn.metrics import r2_score
# y_true: True values
# y_pred: Predicted values
r2 = r2_score(y_true, y_pred)
```

---

**Method Name:** silhouette_score
**Description:** Measures the quality of clustering by assessing the cohesion within clusters and separation between clusters. Higher scores indicate better clustering.
**Hyper-Parameters:**
- `metric`: Distance metric to use.
**Pros:** Useful for validating clustering performance.
**Limitations:** Sensitive to outliers and choice of distance metric.

```python
from sklearn.metrics import silhouette_score
# X: Data used in clustering
# labels: Cluster labels for each sample
score = silhouette_score(X, labels, metric='euclidean')
```

---

**Method Name:** silhouette_samples
**Description:** Provides silhouette scores for each individual sample, indicating how well it fits its assigned cluster.
**Hyper-Parameters:**
- `metric`: Distance metric to use.
**Pros:** Offers granular insight into each sample's clustering quality.
**Limitations:** Same as silhouette_score; sensitive to outliers and distance metric.

```python
from sklearn.metrics import silhouette_samples
# X: Data used in clustering
# labels: Cluster labels for each sample
samples = silhouette_samples(X, labels, metric='euclidean')
```

---

**Method Name:** davies_bouldin_score
**Description:** Measures the average similarity ratio of each cluster with the most similar cluster. Lower values indicate better clustering.
**Pros:** Provides a simple, effective clustering evaluation.
**Limitations:** May not work well with highly imbalanced clusters.

```python
from sklearn.metrics import davies_bouldin_score
# X: Data used in clustering
# labels: Cluster labels for each sample
db_score = davies_bouldin_score(X, labels)
```

---

**Method Name:** Voronoi
**Description:** Computes the Voronoi diagram, which partitions space based on the nearest neighbor.
**Pros:** Useful for spatial analysis and clustering.
**Limitations:** Limited to use cases that involve spatial partitioning of data.

```python
from scipy.spatial import Voronoi
# points: Coordinates for Voronoi diagram
vor = Voronoi(points)
```

---

**Method Name:** voronoi_plot_2d
**Description:** Plots the Voronoi diagram in 2D for visualizing clustering results.
**Hyper-Parameters:**
- `show_vertices`: Whether to display the vertices.
**Pros:** Great for visualizing spatial clustering.
**Limitations:** Limited to 2D spaces and large datasets may cause performance issues.

```python
from scipy.spatial import voronoi_plot_2d
# vor: Voronoi diagram object
voronoi_plot_2d(vor, show_vertices=True)
```

---

**Method Name:** matplotlib.patches.Patch
**Description:** Creates custom shapes such as rectangles, circles, or ellipses for adding to plots.
**Hyper-Parameters:**
- `color`: Fills color of the shape.
**Pros:** Versatile for visual customization.
**Limitations:** May not support all shapes or complex customizations.

```python
import matplotlib.patches as patches
# Create a rectangle with specified width, height, and position
rectangle = patches.Rectangle((0, 0), 1, 1, color='blue')
```

---

**Method Name:** explained_variance_score
**Description:** Measures the proportion of variance explained by the model's predictions. A higher score indicates better performance.
**Pros:** Helps in assessing the fit of regression models.
**Limitations:** Not suitable for classification tasks.

```python
from sklearn.metrics import explained_variance_score
# y_true: True values
# y_pred: Predicted values
ev_score = explained_variance_score(y_true, y_pred)
```

---

**Method Name:** Ridge regression
**Description:** Performs ridge regression (L2 regularization) to avoid overfitting by penalizing large coefficients.
**Hyper-Parameters:**
- `alpha`: Regularization strength.
**Pros:** Helps reduce overfitting in regression models.
**Limitations:** May not work well with sparse data.

```python
from sklearn.linear_model import Ridge
# alpha: Regularization strength (larger values indicate stronger regularization)
ridge = Ridge(alpha=1.0)
```

---

**Method Name:** Lasso regression
**Description:** Performs lasso regression (L1 regularization), which encourages sparsity by penalizing the absolute value of coefficients.
**Hyper-Parameters:**
- `alpha`: Regularization strength.
**Pros:** Encourages sparse solutions, useful for feature selection.
**Limitations:** May struggle with multicollinearity.

```python
from sklearn.linear_model import Lasso
# alpha: Regularization strength (larger values indicate stronger regularization)
lasso = Lasso(alpha=0.1)
```

---

**Method Name:** Pipeline
**Description:** Chains multiple steps of preprocessing and modeling into a single object, ensuring efficient workflow.
**Pros:** Simplifies code, ensures reproducibility.
**Limitations:** May not work well with complex pipelines requiring dynamic configurations.

```python
from sklearn.pipeline import Pipeline
# steps: List of tuples with name and estimator/transformer
pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('model', Ridge(alpha=1.0))])
```

---

**Method Name:** GridSearchCV
**Description:** Performs exhaustive search over a specified parameter grid to find the best model configuration.
**Hyper-Parameters:**
- `param_grid`: Dictionary of parameter grids.
**Pros:** Ensures optimal model parameters.
**Limitations:** Computationally expensive for large grids.

```python
from sklearn.model_selection import GridSearchCV
# estimator: Model to be tuned
# param_grid: Dictionary with parameters to search over
grid_search = GridSearchCV(estimator=Ridge(), param_grid={'alpha': [0.1, 1.0, 10.0]})
```

___

**Process Name:** Multiple runs of k-means
**Description:** Executes KMeans clustering multiple times with different random initializations to assess variability in cluster assignments.
**Advantage:** Helps visualize consistency.
**Limitation:** Computationally costly for large datasets.

```python
# Number of runs for KMeans with different random states
n_runs = 4
inertia_values = []
plt.figure(figsize=(12, 12))

# Run K-Means multiple times with different random states
for i in range(n_runs):
    kmeans = KMeans(n_clusters=4, random_state=None)  # Use the default `n_init`
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)
    
    # Plot the clustering result
    plt.subplot(2, 2, i + 1)
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='tab10', alpha=0.6, edgecolor='k')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                c='red', s=200, marker='x', label='Centroids')
    plt.title(f'K-Means Clustering Run {i + 1}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()

plt.tight_layout()
plt.show()

# Print inertia values
for i, inertia in enumerate(inertia_values, start=1):
    print(f'Run {i}: Inertia={inertia:.2f}')
```

---

**Process Name:** Elbow method
**Description:** Evaluates the optimal number of clusters by plotting inertia (within-cluster sum of squares) for different _k_ values.
**Advantage:** Easy to interpret.
**Limitation:** Subjective elbow point.

```python
# Range of k values to test
k_values = range(2, 11)

# Store performance metrics
inertia_values = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_kmeans = kmeans.fit_predict(X)
    
    # Calculate and store metrics
    inertia_values.append(kmeans.inertia_)
    
# Plot the inertia values (Elbow Method)
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.plot(k_values, inertia_values, marker='o')
plt.title('Elbow Method: Inertia vs. k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
```

---

**Process Name:** Silhouette method
**Description:** Determines the optimal number of clusters by evaluating Silhouette Scores for different _k_ values.
**Advantage:** Considers both cohesion and separation.
**Limitation:** High computation for large datasets.

```python
# Range of k values to test
k_values = range(2, 11)

# Store performance metrics
silhouette_scores = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_kmeans = kmeans.fit_predict(X)
    silhouette_scores.append(silhouette_score(X, y_kmeans))
    
# Plot the Silhouette Scores
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 2)
plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette Score vs. k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
```

---

**Process Name:** Davies-Bouldin Index
**Description:** Evaluates clustering performance by calculating the Davies-Bouldin Index (DBI) for different _k_ values.
**Advantage:** Quantifies compactness and separation.
**Limitation:** Sensitive to cluster shapes and density.

```python
# Range of k values to test
k_values = range(2, 11)

# Store performance metrics
davies_bouldin_indices = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_kmeans = kmeans.fit_predict(X)
    davies_bouldin_indices.append(davies_bouldin_score(X, y_kmeans))
    
# Plot the Davies-Bouldin Index
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 3)
plt.plot(k_values, davies_bouldin_indices, marker='o')
plt.title('Davies-Bouldin Index vs. k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Davies-Bouldin Index')
```