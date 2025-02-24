## Classification Metrics and Evaluation Techniques

What is supervised learning evaluation?
- Establishes how well a machine learning model can predict the outcome of unseen data
- Essential for understanding model effectiveness
- Involves comparing model predictions to ground truth labels
- During training
	- optimizes predictions based on evaluation metrics
- After training
	- model evaluated to estimate how well it can generalize unseen data
- Essential to both the training and testing phases

*train/test split technique*

![[videoframe_84685.png]]

==common metrics for evaluating classification models==
- accuracy - the ratio of correctly predicted instances to the total number of instances in the data set

![[videoframe_151294.png]]

- confusion matrix - a table that breaks down the number of ground truth instances of a specific class against the number of predicted class instances

![[videoframe_183218.png]]

- precision - how many of the predicted positive instances are actually positive

![[videoframe_267697.png]]

- recall - how many of the actual positive instances are correctly predicted

![[videoframe_301642.png]]

- f1 score - combines precision and recall to represent a model's accuracy

![[videoframe_326433.png]]

![[videoframe_345199.png]]


___

## Regression Metrics and Evaluation Techniques

![[videoframe_114559.png]]

![[videoframe_163385.png]]

![[videoframe_241811.png]]

![[videoframe_291607.png]]


___

## Evaluating Unsupervised Learning Models: Heuristics and Techniques

___

## Cross-Validation and Advanced Model Validation Techniques

___

## Regularization in Regression and Classification

___

## Data Leakage and Other Pitfalls

___

# Module 5 Summary and Highlights

Congratulations! You have completed this lesson. At this point in the course, you know:

- Supervised learning evaluation assesses a model's ability to predict outcomes for unseen data, often using a train/test split to estimate performance.
    
- Key metrics for classification evaluation include accuracy, confusion matrix, precision, recall, and the F1 score, which balances precision and recall.
    
- Regression model evaluation metrics include MAE, MSE, RMSE, R-squared, and explained variance to measure prediction accuracy.
    
- Unsupervised learning models are evaluated for pattern quality and consistency using metrics like Silhouette Score, Davies-Bouldin Index, and Adjusted Rand Index.
    
- Dimensionality reduction evaluation involves Explained Variance Ratio, Reconstruction Error, and Neighborhood Preservation to assess data structure retention.
    
- Model validation, including dividing data into training, validation, and test sets, helps prevent overfitting by tuning hyperparameters carefully.
    
- Cross-validation methods, especially K-fold and stratified cross-validation, support robust model validation without overfitting to test data.
    
- Regularization techniques, such as ridge (L2) and lasso (L1) regression, help prevent overfitting by adding penalty terms to linear regression models.
    
- Data leakage occurs when training data includes information unavailable in real-world data, which is preventable by separating data properly and mindful feature selection.
    
- Common modeling pitfalls include misinterpreting feature importance, ignoring class imbalance, and relying excessively on automated processes without causal analysis.
    
- Feature importance assessments should consider redundancy, scale sensitivity, and avoid misinterpretation, as well as inappropriate assumptions about causation.