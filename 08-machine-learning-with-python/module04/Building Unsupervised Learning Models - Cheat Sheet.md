## UMAP (Uniform Manifold Approximation and Projection)

**Process:** UMAP is used for dimensionality reduction.  
**Key hyper-parameters:**
- `n_neighbors`: Controls the local neighborhood size (default = 15)
- `min_dist`: Controls the minimum distance between points in the embedded space (default = 0.1)
- `n_components`: The dimensionality of the embedding (default = 2)  
**Pros:** High performance, preserves global structure.  
**Cons:** Sensitive to parameters.  
**Common applications:** Data visualization, feature extraction.

```python
from umap.umap_ import UMAP
umap = UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
```

## t-SNE (t-Distributed Stochastic Neighbor Embedding)

**Process:** t-SNE is a nonlinear dimensionality reduction technique.  
**Key hyper-parameters:**
- `n_components`: The number of dimensions for the output (default = 2)
- `perplexity`: Balances attention between local and global aspects of the data (default = 30)
- `learning_rate`: Controls the step size during optimization (default = 200)  
**Pros:** Good for visualizing high-dimensional data.  
**Cons:** Computationally expensive, prone to overfitting.  
**Common applications:** Data visualization, anomaly detection.

```python
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
```

## PCA (Principal Component Analysis)

**Process:** PCA is used for linear dimensionality reduction.  
**Key hyper-parameters:**
- `n_components`: Number of principal components to retain (default = 2)
- `whiten`: Whether to scale the components (default = False)
- `svd_solver`: The algorithm to compute the components (default = 'auto')  
**Pros:** Easy to interpret, reduces noise.  
**Cons:** Linear, may lose information in nonlinear data.  
**Common applications:** Feature extraction, compression.

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
```

## DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

**Process:** DBSCAN is a density-based clustering algorithm.  
**Key hyper-parameters:**
- `eps`: The maximum distance between two points to be considered neighbors (default = 0.5)
- `min_samples`: Minimum number of samples in a neighborhood to form a cluster (default = 5)  
**Pros:** Identifies outliers, does not require the number of clusters.  
**Cons:** Difficult with varying density clusters.  
**Common applications:** Anomaly detection, spatial data clustering.

```python
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
```

## HDBSCAN (Hierarchical DBSCAN)

**Process:** HDBSCAN improves on DBSCAN by handling varying density clusters.  
**Key hyper-parameters:**
- `min_cluster_size`: The minimum size of clusters (default = 5)
- `min_samples`: Minimum number of samples to form a cluster (default = 10)  
**Pros:** Better handling of varying densities.  
**Cons:** Can be slower than DBSCAN.  
**Common applications:** Large datasets, complex clustering problems.

```python
import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
```

## K-Means clustering

**Process:** K-Means is a centroid-based clustering algorithm that groups data into k clusters.  
**Key hyper-parameters:**
- `n_clusters`: Number of clusters (default = 8)
- `init`: Method for initializing the centroids ('k-means++' or 'random', default = 'k-means++')
- `n_init`: Number of times the algorithm will run with different centroid seeds (default = 10)  
**Pros:** Efficient, simple to implement.  
**Cons:** Sensitive to initial cluster centroids.  
**Common applications:** Customer segmentation, pattern recognition.

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
```

___

## Associated Functions Used

**Function/Method Name:** make_blobs  
**Brief Description:** Generates isotropic Gaussian blobs for clustering.

Code Syntax:

```python
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=100, centers=2, random_state=42)
```

**Function/Method Name:** multivariate_normal  
**Brief Description:** Generates samples from a multivariate normal distribution.

Code Syntax:

```python
from numpy.random import multivariate_normal
samples = multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=100)
```

**Function/Method Name:** plotly.express.scatter_3d  
**Brief Description:** Creates a 3D scatter plot using Plotly Express.

Code Syntax:

```python
import plotly.express as px
fig = px.scatter_3d(df, x='x', y='y', z='z')
fig.show()
```

**Function/Method Name:** geopandas.GeoDataFrame  
**Brief Description:** Creates a GeoDataFrame from a Pandas DataFrame.

Code Syntax:

```python
import geopandas as gpd
gdf = gpd.GeoDataFrame(df, geometry='geometry')
```

**Function/Method Name:** geopandas.to_crs  
**Brief Description:** Transforms the coordinate reference system of a GeoDataFrame.

Code Syntax:

```python
gdf = gdf.to_crs(epsg=3857)
```

**Function/Method Name:** contextily.add_basemap  
**Brief Description:** Adds a basemap to a GeoDataFrame plot for context.

Code Syntax:

```python
import contextily as ctx
ax = gdf.plot(figsize=(10, 10))
ctx.add_basemap(ax)
```

**Function/Method Name:** pca.explained_variance_ratio_  
**Brief Description:** Returns the proportion of variance explained by each principal component.

Code Syntax:

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
variance_ratio = pca.explained_variance_ratio_
```