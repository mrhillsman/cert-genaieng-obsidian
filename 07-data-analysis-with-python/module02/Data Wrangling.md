## Pre-processing
>the process of converting or mapping data from the initial "raw" form into another format in order to prepare the data for further analysis


### Dealing with Missing Values
>whenever a data entry is left empty, values like ? and NaN, or otherwise missing

- check with the data collection source; can they provide what the value should be?
- drop the missing value(s); do what causes the least amount of impact
	- drop the variable
	- drop the data entry
- replace the missing value(s)
	- better than dropping but less accurate because of guessing
	- replacing with the average value of the entire feature is common
	- if averaging is not possible, try using the most common
	- replace based on other functions
- leave it as missing data
```python
# dataframes.dropna() method
# axis=0 drops row, axis=1 drops column
# inplace will modify the data set (dataframe) directly
df.dropna(subset=["column_name"], axis=0, inplace=True)
```

```python
# use to verify
df.dropna(subset=["column_name"], axis=0)

# use to modify
df.dropna(subset=["column_name"], axis=0, inplace=True)
```

```python
# replace missing values
dataframe.replace(missing_value, new_value)
```

```python
mean = df["column_name"].mean()
df["column_name"].replace(np.nan, mean)
```
### Data Formatting
>data from different sources may be in different formats, various units, or in various conventions; standardize in the same format, unit, or convention with pandas
>bringing data into a common standard of expression allowing users to make meaningful comparisons

![[normalization03.png]]

- unit conversion mpg -> L/100km
```python
df["mpg"] = 235/df["mpg"]
df.rename(columns={"mpg": "L/100km"}, inplace=True)
```

- think about ensuring the same entity, type, format as in the above same format or converting from unit to another
- when importing data pandas could assign the wrong data type to the feature
- `dataframe.dtypes` to identify type and `dataframe.astype` to convert data type
	- `df["column"] = df["column"].astype("int")`
### Data Normalization (centering/scaling)
>different columns of numerical data may have very different ranges and direct comparison is often not meaningful
>bring all data into a similar range for more useful comparison; focusing on centering and scaling techniques

![[normalization00.png]]
- further analysis like linear regression the attribute income will intrinsically influence the result more because of its larger value but does not equate to the feature's importance as a predictor; the nature of the data biases the linear regression model to weigh income more heavily than age
#### Some Approaches to Normalization

Simple Feature Scaling

```python
# - divide each value by the max value for the feature
# - new values will range between 0 and 1
df["column"] = df["column"]/df["column"].max()
```

Min-Max

```python
# - take each value and subtract it from the min divide by the range of the feature
# - new values will range between 0 and 1
df["column"] = (df["column"]-df["column"].min())/(df["column"].max()-df["column"].min())
```

Z-score / Standard Score

```python
# - for each value subtract the mu (average) then divide by standard deviation (sigma)
# - new values hover around 0, typically range between -3->3 but can be higher or lower
df["column"] = (df["column"]-df["column"].mean())/df["column"].std()
```

![[normalization01.png]]

### Binning
>creates bigger categories from a set of numerical values
>particularly useful for comparison between groups of data 

```python
# - say the feature price ranges from 527 to 3810
price = [527, 1244, 3810, 975, 876, 558, 1092, ...]
# - bins would be low, med, and high
# - we want 3 bins of equal width so we need 4 numbers as dividers that are equal distance apart
# - linspace will return 4 equally spaced numbers over the specified interval of the price
bins = np.linspace(min(df["price"]), max(df["price"]), 4)
bin_names = ["low", "medium", "high"]
# - pd.cut to segment and sort the data values into bins
df["price-binned"] = pd.cut(df["price"], bin_names, labels=names_of_bins, include_lowest=True)
print(bins)
print(df.tail())
# - histogram can be used to visualize the distribution of the data after being divided into bins
```

![[normalization02.png]]

```bash
(cert-genaieng) [mrhillsman@fedora]$ python binning.py
[ 527.         1621.33333333 2715.66666667 3810.        ]
    manufacturer  category     screen  gpu  os  cpu_core screen_size_inch  cpu_frequency  ram_gb  storage_gb_ssd weight_kg  price price-binned
233       Lenovo         4  IPS Panel    2   1         7            35.56            2.6       8             256       1.7   1891       medium
234      Toshiba         3    Full HD    2   1         5           33.782            2.4       8             256       1.2   1950       medium
235       Lenovo         4  IPS Panel    2   1         5            30.48            2.6       8             256      1.36   2236       medium
236       Lenovo         3    Full HD    3   1         5           39.624            2.5       6             256       2.4    883          low
237      Toshiba         3    Full HD    2   1         5            35.56            2.3       8             256      1.95   1499          low
(cert-genaieng) [mrhillsman@fedora]$
```
### Categorical Vars into Quantitative Vars
>how to convert categorical values into numerical variables to make statistical modeling easier; object type (string) -> numeric type (int, float)

#### One-Hot Encoding
Below the Status feature is currently an object type (string) and for further analysis our data analyst/scientist needs to convert it into a numerical representation

| name    | status      |
| ------- | ----------- |
| Josh    | non-citizen |
| James   | non-citizen |
| Michael | citizen     |
| Juan    | non-citizen |
| Janice  | citizen     |
- Add dummy variables for each unique category
- Assign 0 and 1 to each category

| name    | status      | ... | citizen | non-citizen |
| ------- | ----------- | --- | ------- | ----------- |
| Josh    | non-citizen | ... | 0       | 1           |
| James   | non-citizen | ... | 0       | 1           |
| Michael | citizen     | ... | 1       | 0           |
| Juan    | non-citizen | ... | 0       | 1           |
| Janice  | citizen     | ..  | 1       | 0           |

```python
pd.get_dummies(df["status"])
```

### Cheat Sheet

```python
# Package/Method: Replace missing data with frequency
# Description: Replace the missing values of the data set attribute with the mode common occurring entry in the column.
MostFrequentEntry = df['attribute_name'].value_counts().idxmax()
df['attribute_name'].replace(np.nan, MostFrequentEntry, inplace=True)

# Package/Method: Replace missing data with mean
# Description: Replace the missing values of the data set attribute with the mean of all the entries in the column.
AverageValue = df['attribute_name'].astype(<data_type>).mean(axis=0)
df['attribute_name'].replace(np.nan, AverageValue, inplace=True)

# Package/Method: Fix the data types
# Description: Fix the data types of the columns in the dataframe.
df[['attribute1_name', 'attribute2_name', ...]] = df[['attribute1_name', 'attribute2_name', ...]].astype('data_type')

# Package/Method: Data Normalization
# Description: Normalize the data in a column such that the values are restricted between 0 and 1.
df['attribute_name'] = df['attribute_name'] / df['attribute_name'].max()

# Package/Method: Binning
# Description: Create bins of data for better analysis and visualization.
bins = np.linspace(min(df['attribute_name']), max(df['attribute_name']), n)
GroupNames = ['Group1', 'Group2', 'Group3', ...]
df['binned_attribute_name'] = pd.cut(df['attribute_name'], bins, labels=GroupNames, include_lowest=True)

# Package/Method: Change column name
# Description: Change the label name of a dataframe column.
df.rename(columns={'old_name': 'new_name'}, inplace=True)

# Package/Method: Indicator Variables
# Description: Create indicator variables for categorical data.
dummy_variable = pd.get_dummies(df['attribute_name'])
df = pd.concat([df, dummy_variable], axis=1)
```

### Additional Notes and Code

```python
# update column rounding values to nearest 2 decimals  
df[['column']] = np.round(df[['column']], 2)

# pandas uses NaN and Null interchangeably
# takes a scalar or array-like object and indicates whether values are missing (`NaN` in numeric arrays, `None` or `NaN` in object arrays, `NaT` in datetimelike).
# for scalar input, returns a scalar boolean. for array input, returns an array of boolean indicating whether each corresponding element is missing.
df.isnull()
df['column_name'].isnull()

avg = df['column_name'].astype('float').mean(axis=0)

df['column_name'].value_counts()
df['column_name'].value_counts().idxmax()

df['column_name'].replace(np.nan, value_to_use, inplace=True)

df.rename(columns={"oldname": "newname", "oldname2": "newname2"}, inplace=True)

```