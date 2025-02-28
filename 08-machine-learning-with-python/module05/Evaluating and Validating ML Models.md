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
	- the number of correctly classified observations and dividing it by the number of observations

![[videoframe_151294.png]]

- confusion matrix - a table that breaks down the number of ground truth instances of a specific class against the number of predicted class instances

![[videoframe_183218.png]]

- precision - how many of the predicted positive instances are actually positive
	- true positives divided by the sum of true and false positives

![[videoframe_267697.png]]

- recall - how many of the actual positive instances are correctly predicted
	- true positives divided by the sum of true positives and false negatives
	- when opportunity costs is more important recall may be a more important metric

![[videoframe_301642.png]]

- f1 score - combines precision and recall to represent a model's accuracy
	- cases where precision and recall are equally important

![[videoframe_326433.png]]

*The weighted average of the metrics is weighted by the support of each class, or number of flowers in each class*

![[videoframe_345199.png]]


___

## Regression Metrics and Evaluation Techniques
- regression models make prediction errors
- evaluating a regression model is determining how accurately it can predict continuous numerical values

*Provide insight into a model's performances (accuracy, error distribution, error magnitude)*
- MAE, or Mean Absolute Error, is the average absolute difference between the values fitted by the model and the observed historical data.
- MSE, or Mean Squared Error, is the sum of the squared difference between the values fitted by the model and observed values divided by the number of historical points minus the number of parameters in the model.
- RMSE, or Root Mean Squared Error, is the square root of the MSE.
	- popular evaluation metric because it has the same units as the target variable, making it easier to interpret than MSE.
- R-squared is the amount of variance in the dependent variable that the independent variable can explain.
	- also called coefficient of determination and measures the model's goodness of fit.
	- values range from 0 to 1, with 0 being a badly fit model and 1 being a perfect model.

![[videoframe_114559.png]]

- explained variance is the sum of squared differences between the predictions and the average value of the actual target data
- R^2 is a simple measure for describing model performance, claiming that the model explains 85% of the variation in the outcome is readily understood by non-technical people; r2 == 0.85 == 85%
- remember that R^2 assumes the target is linearly related to the input features; can be misleading for nonlinear models

![[videoframe_163385.png]]

- perfect predictor: model perfectly predicts all data points
- mean-value model: model constantly predicts the mean for every data point
- negative R^2: model performed so poorly that unexplained > total

![[videoframe_241811.png]]

- three linear regression results are shown here based on a simulated target variable with an exponential distribution, commonly known as log-normal distribution. 
- the linear model fits three versions of the target variable, the original target, a Box-Cox transformed version of the target, and a logarithmically transformed version.

![[videoframe_251216.png]]

![[videoframe_291607.png]]


___

## Evaluating Unsupervised Learning Models: Heuristics and Techniques
evaluating unsupervised learning models poses unique challenges compared to supervised models, as there are no predefined labels or ground truths for training.

clustering and dimensionality reduction focus on discovering hidden *patterns and structures* so evaluation assesses *pattern quality* for model effectiveness

no one-size-fits-all approach
combination of methods is essential
- heuristics
- domain expertise
- metrics
- ground truth comparisons
- visualization tools

Clustering Heuristics
- internal evaluation metrics - rely on input data
- external evaluation metrics - use ground truth labels when available
- generalizability or stability evaluation - assesses cluster consistency across data variations (perturbations)
- dimensionality reduction - visualizing clustering outcomes such as scatter plots
- cluster-assisted learning - refining clusters through supervised learning evaluations
- domain expertise - feedback and result interpretation

Internal Cluster Evaluation Metrics

![[videoframe_183542.png]]

![[videoframe_218450.png]]

![[videoframe_241976.png]]

External Cluster Evaluation Metrics

![[videoframe_286971.png]]

Dimensionality Reduction Evaluation
- Explained Variance Ration in PCA
	- measures the variance captured by principle components
	- helps determine the acceptable cumulative explained variance
- Reconstruction Error
	- assesses how original data can be reconstructed
	- lower values indicate better information preservation
- Neighborhood Preservation
	- evaluates relationships between high and lower dimension data points
	- used for t-SNE and UMAP (manifold learning algorithms)

___

## Cross-Validation and Advanced Model Validation Techniques

doing your best to optimize your model while not jeopardizing its ability to predict well on unseen data

prevents overfitting when selecting the best configuration by tuning hyperparameters

*data snooping*
- checking performance on the ==test data== before you are done optimizing your model
- a form of data leakage
- avoiding data snooping
	- decouple model tuning from the final evaluation
	- validation: tuning your model on the training data and testing on unseen test data once satisfied

![[videoframe_119488.png]]

Cross-Validation Algorithm

![[videoframe_152402.png]]

- A solution to avoid overfitting your test data while trying to optimize the model's hyperparameters (k-fold cross-validation)

![[videoframe_223812.png]]

![[videoframe_246310.png]]

![[videoframe_279389.png]]

___

## Regularization in Regression and Classification
- regression technique to prevent overfitting
- constrains model complexity during training by discouraging perfect fitting
- penalizes larger coefficients by reducing their magnitude

Regularized Cost Function = MSE + $\large{\lambda}$ * Penalty
- $\large{\lambda}$ = regularization hyperparameter
- Penalty = Ridge, Lasso, and other methods

*Ridge and Lasso Regression* - Ridge and lasso are regularized forms of linear regression that differ only in their cost functions
- regular linear regression has no penalty term
- ridge uses L_2 or sum of squares penalty on coefficients shrinking them
$$\Large{
\lambda||\theta||_2 = \lambda\sum^{N}_{i=1}{\theta_i^2}
}
$$
- lasso uses L_1 or sum of absolute values penalty
$$\Large{
\lambda||\theta||_1 = \lambda\sum^{N}_{i=1}|{\theta_i}|
}
$$
	- could shrink some to zero
	- responds well to feature sparsity
	- useful for feature selection and data compression tasks

==sparse coefficients== - only a small number of variables significantly contribute to a dataset while the remaining have little or no impact

![[videoframe_153382.png]]

![[videoframe_192713.png]]

![[videoframe_230713.png]]

![[videoframe_252160.png]]

- moderately noisy target after each model was trained on 70% of the dataset 

![[videoframe_288988.png]]

![[videoframe_344087.png]]

# Claude

## Introduction

These are my notes on regularization techniques in linear regression. As a student new to machine learning, I'm trying to understand how regularization helps prevent overfitting and when to use different methods like ridge and lasso regression.

## What is Regularization? 🤔

Regularization is a technique used to prevent **overfitting** in regression models. It works by:

- Constraining the model during training
- Discouraging it from overfitting to the training data
- Suppressing the size of model coefficients

> 📝 **Key Insight**: Overfitting happens when a model performs well on training data but poorly on new, unseen data. Regularization helps address this!

The regularized cost function has this general form:

$$\text{Regularized Cost Function} = \text{Mean Squared Error} + \lambda \times \text{Penalty Term}$$

Where:

- $\lambda$ (lambda) is a parameter that controls the influence of the penalty term
- The penalty term measures the size of the coefficients

## Linear Regression Basics 📊

Linear regression models the relationship between variables by fitting a straight line to the data. Predictions are a linear combination of features, and the goal is to minimize the loss function (usually MSE).

Mathematically, the linear regression model is:

$$\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n$$

Where:

- $\hat{y}$ is the predicted value
- $x_i$ are the feature vectors (can be represented as matrix X)
- $\theta_i$ are the unknown weights/coefficients

> 🔍 **TODO**: Look up the matrix form of linear regression and understand how to represent this equation using matrices

## Types of Regularization 🧩

### Regular Linear Regression

- No penalty term
- Simply minimizes MSE
- Prone to overfitting, especially with noisy data

### Ridge Regression

- Uses L2 penalty (sum-of-squares)
- Penalty term: $\lambda \sum_{i=1}^n \theta_i^2$
- Shrinks coefficients toward zero, but rarely makes them exactly zero
- Better than linear regression in noisy environments

### Lasso Regression

- Uses L1 penalty (sum-of-absolute-values)
- Penalty term: $\lambda \sum_{i=1}^n |\theta_i|$
- Can shrink coefficients to exactly zero (feature selection!)
- Excellent for sparse data and feature selection

> ❓ **Question**: What does the lambda parameter actually control? Need to investigate how to choose the right lambda value!

## Signal-to-Noise Ratio (SNR) and Sparsity 📶

Two key concepts in understanding regression performance:

**Sparsity**: When most coefficients are zero or near-zero, with only a few significant coefficients.

- Sparse coefficients = only a small number of variables significantly contribute
- Non-sparse coefficients = many variables contribute

**SNR (Signal-to-Noise Ratio)**: The ratio of useful information (signal) to unwanted noise.

- High SNR = strong signal, low noise (clean data)
- Low SNR = weak signal, high noise (noisy data)

> 📝 **Note**: Sparsity refers to the true nature of the data, not the model!

## Performance in Different Scenarios 📊

### From the graphs and transcript, I observed:

#### 1. Sparse Coefficients, High SNR (Image 6)

- All three methods perform well for non-zero coefficients
- Lasso precisely identifies zero coefficients
- Linear and ridge have some difficulty with zero coefficients

#### 2. Sparse Coefficients, Low SNR (Image 5)

- Linear regression performs terribly (massive overfitting)
- Lasso excellently identifies zero coefficients
- Ridge is better than linear but not as good as lasso

#### 3. Non-sparse Coefficients, High SNR (Image 4)

- All three methods perform well overall
- Ridge shows slight errors
- Lasso still correctly identifies the few zero coefficients

#### 4. Non-sparse Coefficients, Low SNR (Image 3)

- Linear regression performs poorly (overestimates, assigns negative values to positive coefficients)
- Ridge slightly outperforms lasso for non-zero coefficients
- Lasso still better at finding zero coefficients

### Summary Table (Image 1)

|Method|Sparse, High SNR|Sparse, Low SNR|Non-sparse, High SNR|Non-sparse, Low SNR|
|---|---|---|---|---|
|Regular|⭐⭐⭐⭐|⭐|⭐⭐⭐⭐|⭐|
|Ridge|⭐⭐⭐|⭐⭐⭐|⭐⭐⭐|⭐⭐⭐|
|Lasso|⭐⭐⭐⭐⭐|⭐⭐⭐⭐|⭐⭐⭐⭐⭐|⭐⭐⭐⭐|

> 🔎 **Observation**: Lasso consistently performs the best across all scenarios! Regular linear regression is highly sensitive to noise.

## Practical Example: Noisy Target Prediction (Image 2)

This example shows the performance of all three methods on a moderately noisy target variable:

- **Top row**: Scatter plots comparing test predictions vs actual values
    
    - Lasso points cluster tightly around the ideal 45° line
    - Ridge and regular regression show more scatter/error
- **Bottom row**: Time-series plots comparing predictions with actual values
    
    - Lasso (blue) tracks actual values well
    - Ridge and regular regression show more deviation
- **MSE values**:
    
    - Lasso: 305
    - Ridge: 9744
    - Linear: 9855

> 😮 **Wow**: Lasso's MSE is about 30 times smaller than the other methods!

## When to Use Each Method? 🤔

Based on all this information:

1. **Use Linear Regression when**:
    
    - Data has high SNR (very clean data)
    - You need a simple, interpretable model
    - You have plenty of training examples and few features
2. **Use Ridge Regression when**:
    
    - Dealing with multicollinearity (correlated features)
    - You want to keep all features but reduce their impact
    - You have more features than samples
3. **Use Lasso Regression when**:
    
    - Feature selection is desirable
    - You suspect many features are irrelevant
    - You're dealing with noisy data
    - Model simplicity/interpretability is important

> 🔍 **TODO**: Find Python implementations of these methods using scikit-learn!

## Mathematical Details I Need to Understand Better 🧮

1. How exactly do the L1 and L2 penalties work mathematically?
2. Why does L1 (lasso) drive coefficients to exactly zero while L2 (ridge) only shrinks them?
3. What's the geometric interpretation of these penalties?
4. How to choose the optimal lambda value?

> 📚 **Research**: Need to find resources explaining the geometric intuition behind lasso vs ridge regularization

## Python Implementation (To Try Later) 💻

```python
# This is what I think the code would look like, need to verify
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
linear_pred = linear_reg.predict(X_test)
linear_mse = mean_squared_error(y_test, linear_pred)

# Ridge Regression
ridge_reg = Ridge(alpha=1.0)  # alpha is lambda in our notes
ridge_reg.fit(X_train, y_train)
ridge_pred = ridge_reg.predict(X_test)
ridge_mse = mean_squared_error(y_test, ridge_pred)

# Lasso Regression
lasso_reg = Lasso(alpha=1.0)  # alpha is lambda in our notes
lasso_reg.fit(X_train, y_train)
lasso_pred = lasso_reg.predict(X_test)
lasso_mse = mean_squared_error(y_test, lasso_pred)

# Compare coefficients
print("Linear coefficients:", linear_reg.coef_)
print("Ridge coefficients:", ridge_reg.coef_)
print("Lasso coefficients:", lasso_reg.coef_)
```

> ⚠️ **Note to self**: I think `alpha` in scikit-learn is the same as `lambda` in our equations, but need to verify this!

## Key Takeaways 🌟

1. **Regularization** helps prevent overfitting by adding a penalty term to the cost function
2. **Ridge regression** (L2) shrinks coefficients toward zero but rarely makes them exactly zero
3. **Lasso regression** (L1) can shrink coefficients to exactly zero, making it great for feature selection
4. **Linear regression** performs well with clean data but poorly with noisy data
5. **Lasso consistently outperforms** the other methods, especially in noisy environments
6. The **choice of method** depends on your data characteristics and goals

## Questions for Study Group/Professor 🙋‍♂️

1. How do we determine if our data has sparse coefficients before choosing a regularization method?
2. What techniques exist for finding the optimal lambda/alpha parameter?
3. Are there hybrid approaches that combine the benefits of ridge and lasso?
4. How do these methods extend to classification problems?
5. What other regularization methods exist beyond ridge and lasso?

## Additional Research Topics 📝

- [ ] Elastic Net (combines L1 and L2 regularization)
- [ ] Cross-validation for hyperparameter tuning
- [ ] Bayesian interpretation of regularization
- [ ] Standardization's effect on regularization
- [ ] Regularization in neural networks

## Glossary 📖

- **Regularization**: Technique to prevent overfitting by constraining model parameters
- **Overfitting**: When a model performs well on training data but poorly on new data
- **MSE (Mean Squared Error)**: Average of squared differences between predicted and actual values
- **L1 Regularization**: Penalty based on absolute values of coefficients (Lasso)
- **L2 Regularization**: Penalty based on squared values of coefficients (Ridge)
- **Sparsity**: Property where most coefficients are zero/near-zero with only a few significant ones
- **SNR (Signal-to-Noise Ratio)**: Ratio of useful information to unwanted noise in data
- **Lambda (λ)**: Parameter controlling the strength of regularization



___
___

## Data Leakage and Other Pitfalls





# Claude

> 🚨 **IMPORTANT**: Data leakage = when training data includes information NOT available in the real world. This makes models look better than they actually are!

## What is Data Leakage? 🧠

Data leakage occurs when our model's training data includes information that wouldn't be available in real-world scenarios (after deployment).

**Example**: Training a house price prediction model using the average of actual home prices across the entire dataset (including future prices that wouldn't be known at prediction time).

### Types of Data Leakage:

1. **Data snooping** - Training set contains information about testing set
2. **Including future information** - Using tomorrow's stock price to predict today's price
3. **Feature engineering errors** - Creating features using the entire dataset instead of just training data

> 🔍 **TODO**: Research examples of data leakage in real-world ML applications! Maybe find a paper or case study?

### Why is Data Leakage Dangerous?

- Deceives your model → misleadingly good performance during training/validation
- Test data will also contain this leaked data → won't detect issues until production
- Models fail to generalize when deployed

## How to Mitigate Data Leakage 🛡️

1. **Avoid global statistics as features**
    
    - Don't use averages or statistics derived from entire dataset
    - Only use information that would be available at prediction time
2. **Proper data splitting**
    
    - Ensure clean separation between training, validation, and test sets
    - Avoid overlap or contamination between sets
3. **Careful cross-validation**
    
    - Prevent leakage across different validation folds
    - Particularly important with time-dependent data!
4. **Pipeline implementation**
    
    - Fit processing pipeline separately to each training fold
    - Apply resultant fitted pipeline to corresponding validation fold

> ❓ **Question**: What's the difference between validation set and test set? Need to look this up!

## Implementation Example (Python) 💻

```python
# Assume libraries are imported and data loaded

# 1. Split data (if not temporal)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('knn', KNeighborsClassifier())
])

# 3. Set parameter grid
param_grid = {
    'pca__n_components': [2, 5, 10],
    'knn__n_neighbors': [3, 5, 7]
}

# 4. Optimization with grid search + cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 5. Evaluate final model on held-out test set
best_model = grid_search.best_estimator_
accuracy = best_model.score(X_test, y_test)
```

> 📝 **NOTE**: The pipeline ensures that data processing steps are properly contained within cross-validation folds.

## Handling Time-Series Data ⏰

With temporal data, random splitting causes leakage! Instead:

1. Use sequential training/testing sets (training always precedes testing)
2. Implement time-series cross-validation:

```python
# Modified code for time-series data
tscv = TimeSeriesSplit(n_splits=4)

# Use TSCV in grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=tscv)
```

Time-series split creates folds that:

- Maintain temporal order
- Use past data for training, future data for validation
- Training set expands with each split while test set shrinks

This would look like:

$$\text{Split 1: } [T][V][ ][ ]$$ $$\text{Split 2: } [T T][V][ ]$$ $$\text{Split 3: } [T T T][V]$$

Where T = training data and V = validation data

> 🔍 **TODO**: Find visualization of time-series cross-validation to better understand this concept

## Feature Importance Pitfalls 🚩

When interpreting feature importance from ML models:

1. **Correlation ≠ Causation**
    
    - Important features don't necessarily drive outcomes
    - Just because a feature is predictive doesn't mean changing it will affect the target
2. **Redundant/Correlated Features**
    
    - Importance gets shared among similar features
    - Makes them appear less influential than they actually are
3. **Scale Sensitivity**
    
    - Some algorithms (like linear regression) don't account for feature scale
    - Unscaled data can distort importance rankings
4. **Overlooking Interactions**
    
    - Some models rank individual feature importance without accounting for interactions
    - Can underestimate or overestimate their combined impact

**Example**: Two features might seem unimportant individually but their product/interaction could be crucial:

- Linear regression: wouldn't detect this without explicitly adding interaction term
- Random forest: might detect interaction implicitly, but importance is shared

> ❗ **Mathematical Note**: If $x_1$ and $x_2$ individually don't predict $y$ well, but $x_1 \times x_2$ does, some algorithms might miss this relationship entirely!

## Other Common Modeling Pitfalls ⚠️

1. **Poor Feature Engineering**
    
    - Using raw data without appropriate transformation
    - Not selecting relevant features
    - This prevents discovering optimal models
2. **Wrong Evaluation Metrics**
    
    - Choosing inappropriate metrics for your problem
    - Misinterpreting what metrics actually tell you
3. **Ignoring Class Imbalances**
    
    - Not addressing uneven distribution of classes
    - Biases predictions toward majority classes
4. **Blind Automation**
    
    - Relying on AutoML without understanding the models
    - Still need domain knowledge!
5. **Invalid What-If Scenarios**
    
    - If model lacks causal features, hypothetical scenarios may be invalid
    - Without causal relationships, predictions based on changes can be misleading

> 🔍 **TODO**: Research techniques for handling class imbalance (SMOTE, class weights, etc.)

## Key Takeaways 🌟

1. Data leakage is when training data contains information unavailable in production
    
2. Mitigate by:
    
    - Avoiding global statistics as features
    - Ensuring proper data splitting
    - Using appropriate cross-validation techniques
    - Implementing pipelines correctly
3. Feature importance interpretation requires caution:
    
    - Be aware of feature redundancy
    - Consider scale sensitivity
    - Don't assume causation
    - Consider feature interactions
4. Watch for other pitfalls:
    
    - Poor feature selection/engineering
    - Inappropriate metrics
    - Class imbalance issues
    - Over-reliance on automation
    - Non-causal what-if scenarios

## Additional Resources to Explore 📚

- [ ] Find papers on real-world data leakage cases
- [ ] Research time-series cross-validation techniques
- [ ] Learn about causal inference in machine learning
- [ ] Explore methods for feature importance interpretation
- [ ] Study techniques for handling class imbalance

> 📌 **Reminder**: Need to ask instructor about practical examples of feature interaction problems!

## Glossary of Terms 📖

- **Data Leakage**: Training data includes information unavailable in production
- **Cross-validation**: Technique to assess how models generalize to independent datasets
- **Feature Importance**: Measure of a feature's contribution to model predictions
- **Time-series Split**: Cross-validation technique that respects temporal order
- **Class Imbalance**: When classes in classification tasks aren't evenly represented
- **Causation vs. Correlation**: Causal relationships imply one variable directly influences another; correlation just means they tend to move together

## Questions for Study Group 🤔

1. How can we detect data leakage after a model is built?
2. What are some domain-specific examples of feature interactions?
3. How do different ML algorithms handle feature interactions differently?
4. What techniques exist for causal inference in ML?

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