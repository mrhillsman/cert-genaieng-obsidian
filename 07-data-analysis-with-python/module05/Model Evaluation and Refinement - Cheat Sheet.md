```python
# Process: Splitting data for training and testing
# Description: First separate the target attribute from the rest of the data. Treat the target attribute as the output and the rest of the data as input. Then split into training and testing subsets.
from sklearn.model_selection import train_test_split
y_data = df['target_attribute']
x_data = df.drop('target_attribute', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)

# Process: Cross validation score
# Description: Creating different subsets of training and testing data multiple times and evaluating performance using $R^2$.
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
lre = LinearRegression()
Rcross = cross_val_score(lre, x_data[['attribute_1']], y_data, cv=n)  # n indicates number of folds
Mean = Rcross.mean()
Std_dev = Rcross.std()

# Process: Cross validation prediction
# Description: Use a cross validated model to create prediction of the output.
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
lre = LinearRegression()
yhat = cross_val_predict(lre, x_data[['attribute_1']], y_data, cv=4)

# Process: Ridge Regression and Prediction
# Description: Use Ridge regression to avoid overfitting in a polynomial model. Parameter alpha modifies the effect of higher-order terms.
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train[['attribute_1', 'attribute_2', ...]])
x_test_pr = pr.fit_transform(x_test[['attribute_1', 'attribute_2', ...]])
RigeModel = Ridge(alpha=1)
RigeModel.fit(x_train_pr, y_train)
yhat = RigeModel.predict(x_test_pr)

# Process: Grid Search
# Description: Use Grid Search with cross-validation to find the best alpha value for the Ridge regression model.
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
parameters = [{'alpha': [0.001,0.1,1,10,100,1000,10000,...]}]
RR = Ridge()
Grid1 = GridSearchCV(RR, parameters, cv=4)
Grid1.fit(x_data[['attribute_1','attribute_2',...]], y_data)
BestRR = Grid1.best_estimator_
BestRR.score(x_test[['attribute_1','attribute_2',...]], y_test)
```

# normalize parameter

The `normalize` parameter in `sklearn.linear_model.Ridge` has been deprecated and is no longer supported in recent versions of scikit-learn.

## Deprecation and Removal

The `normalize` parameter was deprecated in scikit-learn version 1.0 and has been removed in subsequent versions[2][4]. This change affects not only Ridge regression but also other linear models like LinearRegression, Lasso, and LassoCV[4].

## Reasons for Deprecation

The deprecation of the `normalize` parameter was part of an effort to improve the consistency and performance of scikit-learn's linear models. The use of `normalize=True` could potentially lead to data leakage in cross-validation scenarios, which is why its removal was recommended[4].

## Alternative Approach

Instead of using the `normalize` parameter, the recommended approach is to use a scikit-learn pipeline with StandardScaler. This method provides more control over the preprocessing step and avoids potential issues with data leakage[4][5]. Here's how you can implement this:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

model = make_pipeline(StandardScaler(with_mean=False), Ridge())
```

This pipeline first standardizes the input features using `StandardScaler` and then applies Ridge regression.

## Impact on Existing Code

If you're using an older version of scikit-learn that still supports the `normalize` parameter, you may see deprecation warnings. In newer versions, attempting to use the `normalize` parameter will result in an error[2].

It's important to update any existing code that relies on the `normalize` parameter to use the pipeline approach instead. This will ensure compatibility with current and future versions of scikit-learn while maintaining the desired preprocessing of your data.

Citations:
[1] https://scikit-learn.org/0.21/modules/generated/sklearn.linear_model.Ridge.html
[2] https://stackoverflow.com/questions/76842353/invalid-parameter-normalize-for-estimator-linearregression/76842878
[3] https://stackoverflow.com/questions/50747922/python-sklearn-ridge-regression-normalize
[4] https://github.com/scikit-learn/scikit-learn/discussions/21238
[5] https://scikit-survival.readthedocs.io/en/v0.21.0/release_notes.html
[6] https://www.reddit.com/r/AskStatistics/comments/ugnhu3/im_currently_learning_ridge_and_lasso_regressions/
[7] https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/modules/generated/sklearn.linear_model.Ridge.html
[8] https://github.com/rapidsai/cuml/issues/4795
[9] https://scikit-learn.org/0.17/modules/generated/sklearn.linear_model.Ridge.html
[10] https://ibex.readthedocs.io/en/latest/_modules/sklearn/linear_model/ridge.html