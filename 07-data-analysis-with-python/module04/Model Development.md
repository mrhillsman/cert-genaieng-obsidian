## Model Development
>mathematical equation used to predict a value given one or more other values
>relating one or more independent features to dependent features

the more relevant features you have the more accurate your model is, why?, consider you have two almost identical products like cars, pink cars sell for significantly less, if the features (independent - the input) does not include color, the model will predict the same price even though the pink cars will sell for much less
### Linear Regression / Multiple Linear Regression
single independent feature v multiple independent features

$\Large{x_i}$ -> linear regression -> prediction
$\Large{x_i}$,...$\Large{x_n}$ -> multiple linear regression -> prediction

simple linear regression
- predictor - x
- target - y
$$
\Large{y = b_0 + b_1x}
$$
when you fit or train the model you come up with the intercept and slope

$\Large{b_0}$ - the intercept
$\Large{b_1}$ - the slope

We're skipping how $\Large{b_0}$ and $\Large{b_1}$ are calculated due to the math required but we get them when we fit/train the model. Assuming there is a relationship between highway-mpg and price:

In order to determine the line we take data points from the data set and use them to fit/train the model. The results of the training points are the parameters.

Factors (uncertainty) other than highway-mpg go into determining price and so a small random value is added to the point on the line called **noise**

The closer to 0 the more noise. There is a bell curve displayed where the vertical axis represents the amount/value of the noise and the horizontal axis represents the probability noise will be added

Summary
1. We have a set of training points
2. We use training points to fit/train the model and get parameters $\Large{(b_0,b_1)}$
3. We use the parameters in the model
4. Now we have a model $\Large{\hat{y} = b_0 + b_1 x}$
5. We use the $\Large{\hat{}}$ on the $\Large{y}$ ( $\Large{\hat{y}}$ ) to denote the model is an estimate

Fitting in Python (Simple Linear Regression)

```python
# import linear_model from scikit-learn
from sklearn.linear_model import LinearRegression

# create a linear regression object using the constructor
lm = LinearRegression()

# define the predictor variable and target variable
x = df[['highway-mpg']]
y = df['price']

# use lm.fit(x, y) to fix the model; i.e. find the parameters b_0 and b_1
lm.fit(x, y)

# obtain a prediction; output is an array with the same number of samples as the input x
yhat = lm.predict(x)

# view the intercept b_0 and slope b_1
print(lm.intercept_)
print(lm.coef_)

# relationship of price and highway-mpg
# price = lm.intercept_ - lm.coef_ * highway-mpg
# \hat{y} = b_0 + b_1x
```

Fitting in Python (Multiple Linear Regression)

```python
# extract the 4 predictor variables and store them in Z
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]

# train the model
lm.fit(Z, df['price'])

# obtain a prediction; output will be an array / dataframe with 4 columns - rows correspond to the number of samples - output is an array with same number of elements as number of samples
yhat = lm.predict(Z)

# estimated linear model
# price = lm.intercept_ + b_1 * horsepower + b_2 * curb-weight + b_3 * engine-size + b_4 * highway-mpg
# \hat{y} = b_0 + b_1x_1 + b_2x_2 + b_3x_3 + b_4x_4
```
#### [[Thoughts on Linear Regression]]
### Model Evaluation using Visualization

### Polynomial Regression and Pipelines

### Measure for In-Sample Evaluation

### Prediction and Decision Making

### Cheat Sheet
