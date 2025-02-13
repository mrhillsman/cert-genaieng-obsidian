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

x = [2.0, 2.4, 1.5, 3.5, 3.5, 3.5, 3.5, 3.7, 3.7]
y = [196, 221, 136, 255, 244, 230, 232, 255, 267]

##### Explanation of above equations

**High-Level Explanations**

1. **Prediction Equation**  
    $$ \hat{y} = \theta_0 + \theta_1 x $$
    
    - is the predicted (or “fitted”) value of .
    - (the **intercept**) is the value of when .
    - (the **slope**) shows how much changes when changes by one unit.
2. **Slope Formula**  
    $$ \theta_1 = \frac{,n \sum_{i=1}^{n} (x_i , y_i) ;-; \bigl(\sum_{i=1}^{n} x_i\bigr),\bigl(\sum_{i=1}^{n} y_i\bigr)} {,n \sum_{i=1}^{n} x_i^{2} ;-;\bigl(\sum_{i=1}^{n} x_i\bigr)^2} $$
    
    - is the number of data points.
    - is the sum of products .
    - is the sum of squares of .
    - This fraction captures how and vary together (the numerator) and how varies on its own (the denominator).
3. **Intercept Formula**  
    $$ \theta_0 = \frac{\sum_{i=1}^{n} y_i ;-; \theta_1 \sum_{i=1}^{n} x_i}{n} $$
    
    - is the total of all -values.
    - is the slope times the sum of all -values.
    - Dividing by adjusts the line so that it best fits the data points.

---

**Using the given data**

```latex

x = [2.0,\,2.4,\,1.5,\,3.5,\,3.5,\,3.5,\,3.5,\,3.5,\,3.7],
\quad
y = [196,\,221,\,136,\,255,\,244,\,230,\,232,\,255,\,267]
```

After computing all the sums and plugging into the above equations, we get approximate values:

```latex

\theta_1 \approx 44.73,
\quad
\theta_0 \approx 91.54.
```

Thus, our **final regression line** is:

```latex

\hat{y} \;=\; 91.54 \;+\; 44.73 \, x.
```
















for $\large{\theta_1}$'s numerator the $\large{n}$ in the from is the total number of values of $\large{x}$. the minuend (left side of the subtraction) is where you take i value of x and y and multiply them then add up the sum of all of them, what this means is for every value of x take its corresponding value from y, multiply them together and set the product aside, once you have the product of all x and y values add them all together. after you have done this you multiply the resulting product by $\large{n}$. the subtrahend is the sum of all x values multiplied by the sum of all y values.

for 

for $\large{\theta_0}$ the 


Ordinary Lease Squares Regression
- helpful because it is easy to understand and interpret
- doesn't require any tuning
- its solution is just a calculation
- fast, especially for smaller datasets
- may be far too simplistic to capture complexity; nonlinear relationships in data
- outliers can greatly reduce accuracy; too much weight in the calculation





## Multiple Linear Regression
extension of simple linear regression using two or more independent variables to estimate a dependent variable

linear combination of the form
$\large{\hat{y}=\theta_0+\theta_1x_1+\theta_2x_2+...+\theta_nx_n}$

$\large{X=[1,x_1,x_2,\dots,x_n]}$


% Regression model
$$
\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n
$$

% X = [ 1, x_1, x_2, ..., x_n ]
$$
X = [\,1,\,x_1,\,x_2,\,\dots,\,x_n\,]
$$
(Feature vectors)

% Theta as a column vector
$$
\theta = 
\begin{pmatrix}
\theta_0 \\
\theta_1 \\
\vdots \\
\theta_n
\end{pmatrix}
$$
(Weights)

% Matrix-vector form
$$
\hat{y} = X \,\theta
$$
(Matrix-vector form)



better model than simple linear regression
too many variables can cause overfitting (essentially memorize the training data making it a predictor for unseen observations)

used to predict the impact of changes in what-if scenarios
- hypothetical changes to one or more of your model's input features to see the  predicted outcome

**convert categorical independent variables**
- given a binary variable such as car type
	- 0 = manual, 1 = automatic
- more than two classes
	- transform into new Boolean features one for each class

**what-if scenarios**
- can sometimes provide inaccurate findings in the following situations
	- considering impossible scenarios for your model to obtain predictions
	- extrapolate scenarios that are too distant from the realm of the data it was trained on
	- model might depend on more than one variable amongst a group of correlated or collinear variables

correlated variables are no longer independent variables; they are collinear - predictors of each other - what-if scenarios not possible

**dealing with correlation pitfalls**
- remove redundant variables
- select variables:
	- most understood
	- controllable
	- most correlated with the target

Least Squares Linear Regression

Minimizing MSE
$$
\Large{MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
$$
Same as minimizing SE
$$
\Large{SE=\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
$$

Estimating parameters for multiple linear regression
- ordinary least squares estimate coefficients by minimizing MSE; uses linear algebra to calculate optimal theta
- gradient descent (optimization algorithm) starts with random values for each coefficient - useful for large dataset


## Polynomial and Nonlinear Regression

statistical method for modeling the relationship between dependent and independent variables represented by a nonlinear equation
equation could be polynomial, exponential, logarithmic, or any other function that does not use linear parameters
- useful when there is a complex relationship between variables that cannot be captures through a straight line
	- i.e. when using a dataset that follows an exponential growth pattern


### Nonlinear Modeling Techniques

polynomial
- uses an ordinary linear regression to indirectly fit your data to polynomial expressions of the features, rather than the features themselves
nonlinear
- follows the same idea as polynomial but bases its inputs on function of the given features, such as the logarithm or exponential of the features
- does not necessarily reduce to linear regression like polynomial does

data is rarely linear and it is more common data has a background trend that follows a smoothed curve

**overfitting** - model "memorizes" everything, including noise or variations rather than understanding the underlying patterns

polynomial
- nonlinear dependence on the input features but it has a linear dependence on the regression coefficients
- can be transformed into a linear regression problem
- real-world, complex, nonlinear relationships can't be modeled as polynomial

### Applications of nonlinear regression
(can't be modeled as polynomials)

Exponential or Compound Growth
- how investments grow with compound interest rates

Logarithmic
- law of diminishing returns
	- how incremental gains in productivity or profit can reduce as investment in a production factor, such as labor increases

Periodicity
- sinusoidal seasonal variations in a quantity, such as monthly rainfall or temperature

Compound Growth Example (China GDP)
gdp increase over time and the rate of this growth also increases. increasing growth rate characteristic of exponential growth. a reasonable regression model uses an exponential function $\large{\hat{y} = \theta_0 + \theta_1e^x}$

### Linear or Nonlinear Regression?
- analyze scatterplots of target variable against input variable to reveal patterns
- express patterns as functions and determine if
	- linear
	- exponential
	- logarithmic
	- sinusoidal

### Optimizing nonlinear models
- if you have a mathematical expression for your proposed model you can use an optimization technique like gradient descent to find optimal parameters
- if you have decided on a regression model select amongst ml models
	- regression trees
	- random forests
	- neural networks
	- support vector machines
	- gradient boosting machines
	- k-nearest neighbors














