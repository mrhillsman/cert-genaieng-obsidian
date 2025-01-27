```python
# Process: Linear Regression
# Description: Create a Linear Regression model object
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# Process: Train Linear Regression model
# Description: Train the Linear Regression model on decided data, separating Input and Output attributes.
X = df[['attribute_1', 'attribute_2', ...]]
Y = df['target_attribute']
lr.fit(X, Y)

# Process: Generate output predictions
# Description: Predict the output for a set of Input attribute values.
Y_hat = lr.predict(X)

# Process: Identify the coefficient and intercept
# Description: Identify the slope coefficient and intercept values of the linear regression model.
coeff = lr.coef_
intercept = lr.intercept_

# Process: Residual Plot
# Description: Draw a scatterplot of the residuals after regression.
import seaborn as sns
sns.residplot(x=df[['attribute_1']], y=df[['attribute_2']])

# Process: Distribution Plot
# Description: Plot the distribution of data with respect to a given attribute.
import seaborn as sns
sns.distplot(df['attribute_name'], hist=False)

# Process: Polynomial Regression
# Description: Create polynomial features (single variable) using numpy and fit a model.
f = np.polyfit(x, y, n)
# creates the polynomial features of order n
p = np.poly1d(f)
# p becomes the polynomial model used to generate predicted output
Y_hat = p(x)
# Y_hat is the predicted output

# Process: Multi-variate Polynomial Regression
# Description: Generate a new feature matrix of polynomial combinations for multiple attributes.
from sklearn.preprocessing import PolynomialFeatures
Z = df[['attribute_1', 'attribute_2', ...]]
pr = PolynomialFeatures(degree=n)
Z_pr = pr.fit_transform(Z)

# Process: Pipeline
# Description: Create a pipeline to process data steps automatically (scale, polynomial features, model).
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
Input = [
    ('scale', StandardScaler()),
    ('polynomial', PolynomialFeatures(include_bias=False)),
    ('model', LinearRegression())
]
pipe = Pipeline(Input)
Z = Z.astype(float)
pipe.fit(Z, y)
ypipe = pipe.predict(Z)

# Process: R^2 value
# Description: Measure how close the data is to the fitted regression line (coefficient of determination).

# a. For Linear Regression
X = df[['attribute_1', 'attribute_2', ...]]
Y = df['target_attribute']
lr.fit(X, Y)
R2_score = lr.score(X, Y)

# b. For Polynomial Regression
from sklearn.metrics import r2_score
f = np.polyfit(x, y, n)
p = np.poly1d(f)
R2_score = r2_score(y, p(x))

# Process: MSE value
# Description: Calculate Mean Squared Error, average of the squares of errors (actual - predicted).
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y, Yhat)
```