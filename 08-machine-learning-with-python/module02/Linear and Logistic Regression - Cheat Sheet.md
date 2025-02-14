### Comparing different regression types

**Model Name:** Simple linear regression  
**Purpose:** To predict a dependent variable based on one independent variable.  
**Pros:** Easy to implement, interpret, and efficient for small datasets.  
**Cons:** Not suitable for complex relationships; prone to underfitting.  
**Modeling equation:** y = b0 + b1x  

Code Syntax:

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
```

**Model Name:** Polynomial regression  
**Purpose:** To capture nonlinear relationships between variables.  
**Pros:** Better at fitting nonlinear data compared to linear regression.  
**Cons:** Prone to overfitting with high-degree polynomials.  
**Modeling equation:** y = b0 + b1x + b2x2 + ...  

Code Syntax:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model = LinearRegression().fit(X_poly, y)
```

**Model Name:** Multiple linear regression  
**Purpose:** To predict a dependent variable based on multiple independent variables.  
**Pros:** Accounts for multiple factors influencing the outcome.  
**Cons:** Assumes a linear relationship between predictors and target.  
**Modeling equation:** y = b0 + b1x1 + b2x2 + ...  

Code Syntax:

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
```

**Model Name:** Logistic regression  
**Purpose:** To predict probabilities of categorical outcomes.  
**Pros:** Efficient for binary classification problems.  
**Cons:** Assumes a linear relationship between independent variables and log-odds.  
**Modeling equation:** log(p/(1-p)) = b0 + b1x1 + ...  

Code Syntax:

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, y)
```

___
### Associated functions commonly used

**Function/Method Name:** train_test_split  
**Brief Description:** Splits the dataset into training and testing subsets to evaluate the model's performance.  

Code Syntax:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Function/Method Name:** StandardScaler  
**Brief Description:** Standardizes features by removing the mean and scaling to unit variance.  

Code Syntax:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Function/Method Name:** log_loss  
**Brief Description:** Calculates the logarithmic loss, a performance metric for classification models.  

Code Syntax:

```python
from sklearn.metrics import log_loss
loss = log_loss(y_true, y_pred_proba)
```

**Function/Method Name:** mean_absolute_error  
**Brief Description:** Calculates the mean absolute error between actual and predicted values.  

Code Syntax:

```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)
```

**Function/Method Name:** mean_squared_error  
**Brief Description:** Computes the mean squared error between actual and predicted values.  

Code Syntax:

```python
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true, y_pred)
```

**Function/Method Name:** root_mean_squared_error  
**Brief Description:** Calculates the root mean squared error (RMSE), a commonly used metric for regression tasks.  

Code Syntax:

```python
from sklearn.metrics import mean_squared_error
import numpy as np
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
```

**Function/Method Name:** r2_score  
**Brief Description:** Computes the R-squared value, indicating how well the model explains the variability of the target variable.  

Code Syntax:

```python
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)
```