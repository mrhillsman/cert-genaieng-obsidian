## Model Evaluation and Refinement
>How our model performs in the real world

in-sample data - data used to train - used to build model and discover predictive relationships
out-of-sample data - data used to test - used to evaluate predictive model

![[videoframe_81194.png]]

```python
# random splits the data into training and test sets
from sklearn.model_selection import train_test_split

# x_data: features or independent variables
# y_data: dataset target - df['price']
# x_train, y_train: parts of available data as training set
# x_test, y_test: parts of available data as testing set
# test_size: percentage of the data for testing (0.3 == 30%)
# random_state: number generator used for random sampling
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=0)
```

Generalization performance
- Generalization error is a measure of how well our model does at predicting previously unseen data i.e. the error we obtain using our testing data is an approximation of this error

![[videoframe_165844.png]]

Using a lot of training data, gives us an accurate means of determining how well our model will perform in the real world, but the precision of the performance will be low

in-sample data - data used to train
out-of-sample data - data used to test

More data to train and less to test (90/10) results in accurate but low precision

![[videoframe_213330.png]]

More data to test and less to train results in less accurate but more precise

![[videoframe_233330.png]]

Cross Validation
- most common out-of-sample evaluation metric
- more effective use of data
- split the data into k-equal groups (folds)
	- if you have 4 folds use 3 for training and 1 for testing swapping out the 3 to 1 combination until each fold is used for both training and testing
	- at the end use the average of the results as the estimate for out-of-sample error
```python
from sklearn.model_selection import cross_val_score
import numpy as np

# first argument is the model here we use lr = LinearRegression()
# x_data and y_data are the independent and dependent respectfully
# cv is how many folds/partitions
# 
# By default, if you pass a regressor into `cross_val_score` without specifying a `scoring` parameter, it uses the regressor’s default `.score()` method, which returns the R² value for each fold. So the array you get back is automatically populated with R² scores.
scores = cross_val_score(lr, x_data, y_data, cv=3)

# average of out-of-sample r-squared
result = np.mean(scores)
```

Get the actual predicted values supplied by the model before the r-squared values are calculated
```python
from sklearn.model_selection import cross_val_predict

# first argument is the model here we use lr = LinearRegression()
# x_data and y_data are the independent and dependent respectfully
# cv is how many folds/partitions
yhat = cross_val_predict(lr, x_data, y_data, cv=3)
```

**Key differences** between `cross_val_score` and `cross_val_predict`:

1. **Purpose**
    - **`cross_val_score`**: Evaluates an estimator’s performance by returning an array of scores (one per cross-validation fold).
    - **`cross_val_predict`**: Generates out-of-sample predictions for each data point in the dataset, effectively showing what predictions look like when each fold of data is "unseen."

1. **Arguments**
    - Both take similar arguments (e.g., `estimator`, `X`, `y`, `cv`, etc.).
    - **`cross_val_score`** has a `scoring` parameter controlling what metric gets evaluated (e.g., R², accuracy).
    - **`cross_val_predict`** doesn’t compute a performance metric; it just returns predictions. You’d have to compute any metrics yourself afterward.

2. **Output**
    - **`cross_val_score`**: Returns an array of metric scores (one per fold).
    - **`cross_val_predict`**: Returns a single array of predictions, where each prediction was made by a model trained without that data point's fold.

Essentially:
- Use **`cross_val_score`** if you want cross-validation metrics.
- Use **`cross_val_predict`** if you want cross-validated predictions you can analyze further (e.g., constructing residual plots).
## Overfitting, Underfitting, and Model Selection
Model Selection
-  How to pick the best polynomial order and issues that arise when picking the wrong order
We assume the training points come from a polynomial function + noise
$$
y(x)+\text{noise}
$$
Underfitting - model is too simple to fit the data

![[videoframe_47022.png]]

Overfitting - the model is too flexible and fits the noise rather than the function

![[videoframe_82837.png]]

## Ridge Regression

## Grid Search

### Cheat Sheet
