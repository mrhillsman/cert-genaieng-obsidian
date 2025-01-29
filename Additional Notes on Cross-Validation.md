The difference between `cross_val_score` and `cross_val_predict` lies in **what they return** and **how they handle predictions versus evaluations during cross-validation**. Here’s a mental model to help:

### Mental Model: Exam Scenario

Imagine a class of students taking a series of exams:

1. **`cross_val_score` (Scoring the exams)**:
   - The teacher grades the exams and records the **scores** for each student on each test.
   - The result is a list of grades (scores) for each exam.
   - You don’t see the detailed answers each student wrote during the tests; you just get the grades.

   In machine learning terms, `cross_val_score`:
   - Splits the dataset into training and test sets for each fold.
   - Trains the model on the training set and evaluates it on the test set.
   - Returns an array of **evaluation metrics** (like accuracy, F1 score, etc.)—one for each fold.

---

2. **`cross_val_predict` (Collecting students' answers)**:
   - The teacher gathers and returns the answers each student wrote during the test, as if the test was unseen to them.
   - Instead of grades, you see how each student **answered** every question when they didn’t already know the test questions (i.e., during out-of-sample testing).

   In machine learning terms, `cross_val_predict`:
   - Splits the dataset into training and test sets for each fold.
   - Trains the model on the training set and uses it to **predict the outcomes** for the test set.
   - Returns a single array of **predictions** for the entire dataset. Each prediction is made by a model that never saw the corresponding test data point during training.

---

### Key Difference in Internal Workings

| Feature                  | `cross_val_score`                             | `cross_val_predict`                          |
|--------------------------|-----------------------------------------------|---------------------------------------------|
| **Output**               | Scores for each fold (e.g., accuracy, RMSE)   | Predicted values for all data points        |
| **Purpose**              | Evaluate model performance                   | Generate out-of-sample predictions          |
| **Training/Test Behavior** | Model evaluates test data in each fold and reports the metric. | Model predicts for test data in each fold and combines results. |
| **Final Result Shape**   | Array of scores (one per fold)                | Array of predictions (one per data point)   |

### Analogy Recap
- **`cross_val_score`**: Grading the tests and reporting the scores.
- **`cross_val_predict`**: Collecting the students' answers from the tests to understand how they responded.


```python
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# read the csv into a dataframe
df = pd.read_csv(file_name, header=0)
# return only data that is of the numeric type
df = df._get_numeric_data()
# drop the two Unnamed columns
df.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1, inplace=True)

# store target (what we want to predict) as y_data and all others (what we can use to predict) as x_data
y_data = df['price']
x_data = df.drop('price', axis=1)

# split up each of the target and feature sets into test and train sets using 40% for test set and no shuffling (random_state) prior to splitting
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.4, random_state=0)

# print("test samples: ", x_test.shape[0])
# print("train samples: ", x_train.shape[0])

# instantiate a linear regression model and fit the model
lr = LinearRegression()
lr.fit(x_train[['horsepower']], y_train)

# run cross-validation to get the r-squared scores when horsepower is the feature used to predict the target price using 4 folds
rcross = cross_val_score(lr, x_data[['horsepower']], y_data, cv=4)
print(rcross)
print("The mean of the folds is:", rcross.mean(), "and the standard deviation is:", rcross.std())

# run cross-validation to get the r-squared scores when horsepower is the feature used to predict the target price using 2 folds
rscore = cross_val_score(lr, x_data[['horsepower']], y_data, cv=2)
print(rscore)

# predict the target price using horsepower and output the values in an array stored in yhat
yhat = cross_val_predict(lr, x_data[['horsepower']], y_data, cv=4)

# total number of predictions
print(len(yhat))
# print the entire first entry of the original dataset
print(df.iloc[0])
# print the horsepower value used in the prediction
print(x_data.iloc[0]['horsepower'])
# print the first predicted value
print(yhat[0])
# print the actual target from original dataset
print(y_data[0])
```

```python
# length
201

# original csv row
symboling                3.000000
normalized-losses      122.000000
wheel-base              88.600000
length                   0.811148
width                    0.890278
height                  48.800000
curb-weight           2548.000000
engine-size            130.000000
bore                     3.470000
stroke                   2.680000
compression-ratio        9.000000
horsepower             111.000000
peak-rpm              5000.000000
city-mpg                21.000000
highway-mpg             27.000000
price                13495.000000
city-L/100km            11.190476
diesel                   0.000000
gas                      1.000000
Name: 0, dtype: float64

# horsepower (value of feature used for predicting)
111.0

# predicted target (price)
14141.638075081995

# actual target (price)
13495.0
```

