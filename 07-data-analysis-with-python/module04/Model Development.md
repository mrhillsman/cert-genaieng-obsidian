## Model Development
>mathematical equation used to predict a value given one or more other values
>relating one or more independent features to dependent features

the more relevant features you have the more accurate your model is, why?, consider you have two almost identical products like cars, pink cars sell for significantly less, if the features (independent - the input) does not include color, the model will predict the same price even though the pink cars will sell for much less
## Linear Regression / Multiple Linear Regression
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

#### [[Thoughts on Linear Regression|Additional Thoughts on Linear Regression]]
## Model Evaluation using Visualization
### Regression Plot
Gives a good estimate of:
- The relationship between two variables
- The strength of the correlation
- The direction of the relationship (positive or negative)

```python
import seaborn as sbn
from mathplotlib import pyplot as plt

# x is the independent variable/feature, y is the dependent variable/target
sbn.regplot(x='highway-mpg', y='price', data=df)
plt.ylim(0,)
```

### Residual Plot
Represents the **error**. The difference (error) is obtained by subtracting the predicted value from the actual value ($\Large{Y_0-\hat{Y}_0})$ then plot the **error** on the vertical axis with the dependent variable as the [[Residual Plot Horizontal Axis|horizontal axis]]

We expect the results to have 0 mean distributed evenly around the x axis with similar variance:

![[videoframe_113976.png]]

When the residual plot is curved like below it suggests the linear assumption is incorrect i.e. a non-linear function:

![[videoframe_141307.png]]

When the variance increases with x our model is incorrect:

![[videoframe_151199.png]]

```python
import seaborn as sbn
from mathplotlib import pyplot as plt

sbn.residplot(df['highway-mpg'], df['price'])
```

### Distribution Plots
Counts the predicted value versus the actual value. Useful for visualizing models with more than one independent variable (feature)

Vertical axis is scaled to make the area under the distribution equal to one.

![[videoframe_234307.png]]

![[videoframe_262626.png]]

```python
import seaborn as sbn

# hist=False because we want distribution not histogram
ax1 = sbn.distplot(df['price'], hist=False, color='r', label='Actual Value')
sbn.distplot(yhat, hist=False, color='b', label='Fitted Values', ax=ax1)
```
## Polynomial Regression and Pipelines
>A special case of the general linear regression model
>Useful for describing curvilinear relationships

Curvilinear Relationships:
What you get by squaring or setting higher-order terms of the predictor variables

- Quadratic - 2nd order polynomial regression
$$\Large{\hat{Y} = b_0 + b_1x_1 + b_2(x_1)^2}$$
- Cubic - 3rd order polynomial regression
$$\Large{\hat{Y} = b_0 + b_1x_1 + b_2(x_1)^2 + b_3(x_1)^3}$$

- Higher Order - when a good fit hasn't been achieved by 2nd or 3rd order

![[videoframe_82541.png]]

```python
import numpy as np
from sk

# calculate polynomial of 3rd order
f = np.polyfit(x, y, 3)
p = np.poly1d(f)

# print out the model
print(p)

# we can also have multi-dimensional polynomial linear regression
# np.polyfit cannot perform this type of regression
from sklearn.preprocessing import PolynomialFeatures

pr = PolynomialFeatures(degree=2, include_bias=False)

# transform the features into a polynomial feature
x_polly = pr.fit_transform(x[['horsepower', 'curb-weight']])

# more intuitive example
pr = PolynomialFeatures(degree=2)
pr.fit_transform([1, 2], include_bias=False)

# we go from x_1 = 1 and x_2 = 2
# to x_1 = 1, x_2 = 2, x_1x_2 = (1)2, x_1^2 = 1, x_2^2 = (2)^2
# i.e. we go from 1, 2 to 1, 2, 2, 1, 4
# we have a new set of features that are a transformed version of our original features

# normalizing each feature simultaneously
from sklearn.preprocessing import StandardScaler

SCALE = StandardScaler()
SCALE.fit(x_data[['horsepower', 'highway-mpg']])

x_scale = SCALE.transform(x_data[['horsepower', 'highway-mpg']])

```
### Pipelines
>Used to help simplify the process of getting a prediction. Pipelines sequentially perform a series of transformations and the last step carries out a prediction.

Normalization -> Polynomial Transform -> Linear Regression
transformations (Normalization and Polynomial Transform), prediction (Linear Regression)

```python
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

Input = [('polynomial', PolynomialFeatures(degree=2)), ('scale', StandardScaler()),('Model', LinearRegression())]

pipe = Pipeline(Input)

# train the pipeline
pipe.fit(df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y)

yhat = pipe.predict(X[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
```

## Measure for In-Sample Evaluation
>numerically determine how good the model fits on dataset; using numbers instead of pictures
>tells us how well our model will fit the data used to train it
>does not tell us how well the trained model can be used to predict new data

- Mean Squared Error (MSE)
	- find the error
	- square the error
	- take the sum of all squared errors and divide by the number of samples

```python
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df['price'], Y_predict_simple_fit)
print(mse)
```

- R-squared / R^2 / Coefficient of Determination
	- how close is the data to the fitted regression line
	- the percentage of variation of the target variable ($\large{y}$) that is explained by the linear model
$$
\Large{R^2=(1 - \frac{\text{MSE of regression line}}{\text{MSE of the avg of the data }(\bar{y})})}
$$
Relatively Good Fit

![[videoframe_126730.png]]

Performs about the same as just using the average of the data points; not a relatively good fit

![[videoframe_179558.png]]

Calculating in Python (LinearRegression)

![[videoframe_202342.png]]

[[Additional Notes on R-squared]]
## Prediction and Decision Making

![[videoframe_99799.png]]

![[videoframe_303190.png]]


### Cheat Sheet
