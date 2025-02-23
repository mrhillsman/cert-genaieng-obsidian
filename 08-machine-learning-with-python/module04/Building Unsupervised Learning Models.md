## Clustering
automatically groups data points based on similarities
- identifying music genres
- segmenting user groups
- analyzing market segments
can use one or multiple features to form meaningful clusters
resembles classification but works with unlabeled data independently finding patterns to form clusters
- Common Applications
	- Exploratory Data Analysis
		- uncovers natural groups for targeted marketing (customer segmentation)
	- Pattern Recognition
		- groups objects and aids in image segmentation (detecting medical abnormalities)
	- Anomaly Detection
		- identifies outliers (fraud or equipment malfunctions)
	- Feature Engineering
		- creates new features or reduces dimensionality
		- improves model performance and interpretability
	- Data Summarization
		- simplifies data
		- summarizes into smaller clusters
	- Data Compression
		- reduces data size (image compression)
	- Feature Selection
		- identifies essential features that distinguish clusters

- Types of Clustering
	- Partition-based
		- divides data into non-overlapping groups
		- k-means is most common method
			- identifies k clusters with minimal variance
	- Density-based
		- creates clusters of any shape
		- suitable for irregular clusters and noisy data sets
			- dbscan algorithm
	- Hierarchical
		- organizes data into nested clusters
		- contains smaller sub-clusters
		- generates a dendrogram
		- reveals relationships between clusters
			- agglomerative: merges clusters (bottom up approach)
			- divisive: splits clusters (top down approach)
				- both are intuitive and suitable for small to mid-sized data sets

## k-means clustering
iterative, centroid-based clustering algorithm that partitions data into similar groups based on distance between centroids

divides data into k non-overlapping clusters, k-clusters have *minimal variance around centroids* and *maximal dissimilarity between clusters*

**centroid**
the average position of all points in the cluster
- data points nearest to a centroid are grouped within the same category

**k value**
- higher k value - number of clusters - signifies smaller clusters with greater detail
- lower k value results in larger clusters with less detail

k-means algorithm
- initialize the algorithm
	- choose number of clusters, k
	- randomly select k centroids - can be data points or other points from the feature space
- iteratively assign points to clusters and update centroids
	- compute distance matrix - distance from each point to each centroid
	- assign each point to cluster with nearest centroid
	- update cluster centroids as the mean of the data points
- repeat until centroids stabilize or max iterations reached

![[videoframe_135884.png]]


==doesn't work well with imbalanced clusters==

![[videoframe_159303.png]]

*k-means clustering considerations*
- assumes convex clusters (any line drawn between two points remains in the cluster)
- assumes balanced clusters sizes (clusters contain same number of points)
- sensitive to outliers and noise
- partition based algorithm so it is efficient and scales well to big data


*k-means optimization*
- minimize within-cluster sum of squares:
$$\Large{
\sum_{i=1}^K\sum_{x\in{C_i}}||x-\mu{_i}||^2
}
$$
$\large{K}$ = number of clusters
$\large{C_i}$ = $\large{i^{th}}$ cluster
$\large{x}$ = data point
$\large{\mu_i}$ = centroid of cluster $\large{C_i}$
$\large{||x-\mu_i}||^2$ = squared distance between $\large{x}$ and its cluster's centroid


*determining k*
- choosing k is feasible when
	- data is separable
	- difficult to visualize for high-dimensional spaces
	- consider scatterplots between variable pairs to check for separability

heuristic techniques for determining k

![[videoframe_403103.png]]
- silhouette analysis - measures how similar a data point is to its cluster, known as cohesion, compared to other clusters, known as separation
- elbow method - a plot of the k-means objective function for different numbers of clusters
- davies-bouldin index - measures each cluster's average similarity ration, with the cluster most similar

___

## Dimension Reduction and Feature Engineering
clustering, dimension reduction, and feature engineering are complementary techniques in machine learning and data science
work together to improve model performance, quality, and interpretability
clustering -> helps with feature selection and creation, supports dimension reduction, enhances computational efficiency and scalability
dimension reduction -> simplifies visualization of high-dimensional clustering, aids in feature engineering and improves model quality, reduces the number of features required

**dimension reduction before clustering**
- commonly used as a preprocessing step for clustering
- simplifies data structure and improves outcomes
- dimension reduction techniques
	- PCA (principle component analysis)
	- t-SNE (tee-snee)
	- UMAP
**dimension reduction after clustering**
- clustering results are impacted beyond three dimensions
- advanced dimension reduction techniques
	- project outcomes into two or three dimensions
	- improve visual interpretation
- identifies key patterns obscured in higher dimensions

**clustering for feature selection**

![[videoframe_188452.png]]

![[videoframe_199754.png]]

![[videoframe_227707.png]]

![[videoframe_237111.png]]

![[videoframe_242941.png]]

### Dimension Reduction Algorithms
reduce the number of data set features without sacrificing critical information
simplify the data set for machine learning models


- PCA (Principle Component Analysis)
	- is a linear dimensionality reduction algorithm that assumes dataset features are linearly correlated
	- can transform features into a new set of uncorrelated variables called principal components while retaining as much variance as possible
		- orthogonal to each other
		- principal components are orthogonal to each other and define a new coordinate system for the feature space
		- organized in decreasing order of importance
		- first few components often contain most of the information while the rest tend to represent noise
- t-SNE (t-distributed stochastic neighbor embedding)
	- good at finding clusters in complex, high-dimensional data
	- focuses on preserving similarity of points close together
	- similarity is measured as proximity
	- doesn't scale well and can be difficult to tune
- UMAP (Uniform Manifold Approximation and Projection)
	- often used as an alternative to t-SNE
	- constructs a high-dimensional graph representation of the data based on manifold theory
		- assumes that the data lies on a lower-dimensional manifold embedded in higher-dimensional space
	- optimizes a low-dimensional graph structure that best preserves relationships between points
	- scales better than t-SNE
	- preserves the global structure of the data often providing higher cluster performance than t-SNE

![[videoframe_183472.png]]

![[videoframe_193472.png]]

![[videoframe_205935.png]]

![[videoframe_221801.png]]

___
# Module 4 Summary and Highlights

Congratulations! You have completed this lesson. At this point in the course, you know: 

- Clustering is a machine learning technique used to group data based on similarity, with applications in customer segmentation and anomaly detection.
    
- K-means clustering partitions data into clusters based on the distance between data points and centroids but struggles with imbalanced or non-convex clusters.
    
- Heuristic methods such as silhouette analysis, the elbow method, and the Davies-Bouldin Index help assess k-means performance.
    
- DBSCAN is a density-based algorithm that creates clusters based on density and works well with natural, irregular patterns.
    
- HDBSCAN is a variant of DBSCAN that does not require parameters and uses cluster stability to find clusters.
    
- Hierarchical clustering can be divisive (top-down) or agglomerative (bottom-up) and produces a dendrogram to visualize the cluster hierarchy.
    
- Dimension reduction simplifies data structure, improves clustering outcomes, and is useful in tasks such as face recognition (using eigenfaces).
    
- Clustering and dimension reduction work together to improve model performance by reducing noise and simplifying feature selection.
    
- PCA, a linear dimensionality reduction method, minimizes information loss while reducing dimensionality and noise in data.
    
- t-SNE and UMAP are other dimensionality reduction techniques that map high-dimensional data into lower-dimensional spaces for visualization and analysis.