## Introduction to Regression
type of supervised learning that models the relationship between a continuous target variable and explanatory features

deciding which regression type is best depends on the data you have for the dependent (target) variable and the type of model that provides the best  - 

Types:
- Simple - when a single independent (feature) variable estimates/predicts a single dependent (target) variable; can be linear or non-linear
- Multiple - when multiple independent (features) variables estimate/predict a dependent (target) variable; can be linear or non-linear

Applications of Regression
- Forecast sales
- Predict maintenance expenses
- Estimate rainfall
- Spread of infectious diseases

Regression Algorithms
- linear and polynomial
- random forest
- extreme gradient boosting (xgboost)
- k-nearest neighbors (knn)
- support vector machines (svm)
- neural networks

## Introduction to Simple Linear Regression
models linear relationship between continuous target and explanatory features

aims to find the line for minimizing the MSE and this form of regression is commonly known as ordinary least squares (OLS) regressioin

In the early 1800s, Carl Friedrich Gauss and Adrien-Marie Legendre developed a method for calculating the coefficients in a linear equation calculating $\large{\theta_0}$ and $\large{\theta_1}$:

$$
\Large{\hat{y} = \theta_0 + \theta_1 x_1}
$$

$$
\Large{\theta_1 = \frac{n\sum_{i=1}^{n} x_i y_i - \sum_{i=1}^{n} x_i \sum_{i=1}^{n} y_i}{n\sum_{i=1}^{n} x_i^2 - \left(\sum_{i=1}^{n} x_i\right)^2}}
$$
$$
\Large{\theta_0 = \frac{\sum_{i=1}^{n} y_i - \theta_1 \sum_{i=1}^{n} x_i}{n}}
$$

