## Exploratory Data Analysis (EDA)
>approach to analyze data in order to: summarize the main characteristics of the data, gain better understanding of the dataset, uncover relationships between different features, extract important features
>Q: what are the characteristics that have the most impact on the car price
### Descriptive Statistics
>describe the basic features of a data set and obtains a short summary about the sample and measures of the data

```python
df.describe()

# Categorical features - can be divided up into different categories or groups and have discrete values
# summarize categorical data
df['column'].value_counts()
```

##### Box Plots
>great way to visualize numeric data, you can easily spot outliers and also see the distribution and skewness of the data

- Box Plot Legend
	- median - represents where the middle data point is
	- upper quartile - shows where the 75th percentile is
	- lower quartile - shows where the 25th percentile is
		- the data between the upper and lower quartile represents the interquartile range
	- upper extreme - calculated as 1.5 times the inter-quartile range above the 75th percentile
	- lower extreme - 1.5 times the IQR below the 25th percentile
	- outliers - individual dots that occur outside the upper and lower extremes

![[boxplots.png]]

- Box Plots for Comparison
	- distribution of different categories of the drive-wheels feature over price feature
	- the distribution of price between rwd and the other categories is distinct while fwd and 4wd are almost indistinguishable

![[boxplots2.png]]

###### Note:
- continuous features/variables - numbers contained in some range (price and engine size)
##### Scatter Plot
>each observation is represented as a point. shows the relationship between two variables/features.

Predictor/independent - feature/variable on the x-axis used to predict an outcome
Target/dependent - feature/variable on the y-axis you are trying to predict
ex: Could engine size possibly predict the price of a car?
- predictor - engine size
- target - price

![[scatterplot.png]]

It is important to add labels so you can quickly distinguish what you are looking at. In this case the initial indication is as engine size goes up price goes up so there is a positive linear relationship between engine size and price.
### GroupBy in Python
>Q: is there any relationship between the different types of "drive system" and the "price" of the vehicles. if so, which type of drive system adds the most value to a vehicle

```python
# can be applied to categorical features
# groups data into subsets according to the different categories
# you can group by single or multiple features
dataframe.groupby()
```

Find the average price of vehicles and observe how they differ between different types of body-style and drive-wheel features:

```python
features = df[['drive-wheels', 'body-style', 'price']]
# only the average price of each category is shown
group = features.groupby(['drive-wheels', 'body-style'], as_index=False).mean()
print(group)
```

##### [[Pivot Tables]]
>display one feature along the columns and the other feature along the rows

```python
pivot = group.pivot(index='drive-wheels', columns='body-style')
```

Labeling the two feature Pivot Table

|                      | **target**  |         |           |     |
| -------------------- | ----------- | ------- | --------- | --- |
| **predictor x-axis** | convertible | hardtop | hatchback | ... |
| **predictor y-axis** |             |         |           |     |
| 4wd                  | 203.10      | 103.34  | 198.20    |     |
| fwd                  | 240.15      | 120.29  | 220.67    |     |
| rwd                  | 220.34      | 116.89  | 176.54    |     |
##### Heatmap
>takes a rectangular grid of data and assigns a color intensity based on the data value at the grid points; great for multiple features

```python
plt.pcolor(pivot, cmap='RdBu')
plt.colorbar()
plt.show()
```

![[heatmap.png]]

### ANOVA (analysis of variance)
>analysis of variance - a statistical method in the variation in a set of observations is divided into distinct components
### Correlation
>statistical metric for measuring to what extent different variables are interdependent; when we look at two variables over time when one changes how does it affect the other

Lung Cancer -> Smoking
Rain -> Umbrella

Correlation does not imply causation; umbrella and rain are correlated but we do not have enough information to say whether the umbrella caused the rain or the rain caused the umbrella

Engine Size -> Price (Correlation Positive Linear Regression)

```python
# replace ? with NaN to convert feature data type from object to float
df["price"].replace('?', np.NaN, inplace=True)
df["price"] = df["price"].astype(float)

df["peak-rpm"].replace('?', np.NaN, inplace=True)  
df["peak-rpm"] = df["peak-rpm"].astype(float)

# correlation positive linear regression
sbn.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,) # regression line indicating relationship between features
plt.show()

# correlation negative linear regression
sbn.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,) # regression line indicating relationship between features
plt.show()

# correlation weak linear regression
sbn.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,) # regression line indicating relationship between features
plt.show()
```

Positive Correlation - very steep line shows that there is a positive linear relationship between the features; increase engine-size, increase in price, because the line goes up this indicates a positive correlation between the features

![[correlation-positive-linear.png]]

Negative Correlation - despite the relationship being negative the slope of the line is steep which means highway-mpg is still a good predictor of price

![[correlation-negative-linear.png]]

Weak Correlation - 

![[correlation-weak-linear.png]]

### Advanced Correlation

#### Pearson Correlation
>measure the strength of the correlation between two features providing two values
>- Correlation coefficient - indicates correlation
>- P-value - how certain you are about the correlation calculated

Correlation coefficient
- value close to +1 implies a large positive correlation
- value close to -1 implies a large negative correlation
- value close to 0 implies no correlation
P-value
- value < 0.001 strong certainty
- 0.001 < value < 0.05 moderate certainty
- 0.05 < value < 0.1 weak certainty
- value > 0.1 no certainty

![[pearson-correlation-examples.png]]

```python
from scipy import stats

# prior to this there is significant wrangling to ensure pearson correlation can be run
coef, pvalue = stats.pearsonr(df['horsepower'], df['price'])  
print(f"Pearson correlation: {coef:.4f}, P-value: {pvalue:.4f}")

# output
# Pearson correlation: 0.7579, P-value: 0.0000
# positive correlation (close to +1), strong certainty (p-value less than 0.001)
```

#### Chi-Square Test
>statistical method used to determine if there is a significant association between two categorical features
>widely used in various fields, including social sciences, marketing, and healthcare, to analyze survey data, experimental results, and observational studies

evaluates whether the frequencies of observed outcomes significantly deviate from expected frequencies, assuming the features are independent
##### Null Hypothesis and Alternative Hypothesis

The chi-square test involves formulating two hypotheses:
- Null Hypothesis $𝐻_0$ - assumes that there is no association between the categorical features, implying that any observed differences are due to random chance.
- Alternative Hypothesis $H_1$ - assumes that there is a significant association between the features, indicating that the observed differences are not due to chance alone.

##### Formula
$$
\Large{\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}}
$$

where:

- $O_i$ is the observed frequency for feature $i$,
- $E_i$ is the expected frequency for feature $i$, calculated as:

$$
\Large{E_i = \frac{(\text{row total} \times \text{column total})}{\text{grand total}}}
$$

The sum is taken over all cells in the contingency table.

The calculated chi-square statistic is then compared to a critical value from the chi-square distribution table. This table provides critical values for different degrees of freedom $( df )$ and significance levels $\alpha$. 

degrees of freedom
$$
\Large{df = (r-1)*(c-1)}
$$
where $r$ is the number of rows and $c$ is the number of columns
##### [[Chi-Square Test Examples]]
##### Applications
1. **Market Research:** Analyzing the association between customer demographics and product preferences.
2. **Healthcare:** Studying the relationship between patient characteristics and disease incidence.
3. **Social Sciences:** Investigating the link between social factors (e.g., education level) and behavioral outcomes (e.g., voting patterns).
4. **Education:** Examining the connection between teaching methods and student performance.
5. **Quality Control:** Assessing the association between manufacturing conditions and product defects.

### Cheat Sheet

```python
# Package/Method: Complete dataframe correlation
# Description: Correlation matrix created using all the attributes of the dataset.
df.corr()

# Package/Method: Specific Attribute correlation
# Description: Correlation matrix created using specific attributes of the dataset.
df[['attribute1','attribute2',...]].corr()

# Package/Method: Scatter Plot
# Description: Create a scatter plot using the data points of the dependent variable along the x-axis and the independent variable along the y-axis.
from matplotlib import pyplot as plt
plt.scatter(df[['attribute_1']], df[['attribute_2']])

# Package/Method: Regression Plot
# Description: Uses the dependent and independent variables in a Pandas data frame to create a scatter plot with a generated linear regression line for the data.
import seaborn as sbn
sbn.regplot(x='attribute_1', y='attribute_2', data=df)

# Package/Method: Box plot
# Description: Create a box-and-whisker plot that uses the pandas dataframe, the dependent, and the independent variables.
import seaborn as sbn
sbn.boxplot(x='attribute_1', y='attribute_2', data=df)

# Package/Method: Grouping by attributes
# Description: Create a group of different attributes of a dataset to create a subset of the data.
df_group = df[['attribute_1','attribute_2',...]]

# Package/Method: GroupBy statements
# Description:
# a. Group the data by different categories of an attribute, displaying the average value of numerical attributes with the same category.
# b. Group the data by different categories of multiple attributes, displaying the average value of numerical attributes with the same category.
# a.
df_group = df_group.groupby(['attribute_1'], as_index=False).mean()
# b.
df_group = df_group.groupby(['attribute_1','attribute_2'], as_index=False).mean()

# Package/Method: Pivot Tables
# Description: Create Pivot tables for better representation of data based on parameters.
grouped_pivot = df_group.pivot(index='attribute_1', columns='attribute_2')

# Package/Method: Pseudocolor plot
# Description: Create a heatmap image using a PsuedoColor plot (or pcolor) using the pivot table as data.
from matplotlib import pyplot as plt
plt.pcolor(grouped_pivot, cmap='RdBu')

# Package/Method: Pearson Coefficient and p-value
# Description: Calculate the Pearson Coefficient and p-value of a pair of attributes.
from scipy import stats
pearson_coef, p_value = stats.pearsonr(df['attribute_1'], df['attribute_2'])
```
## Additional Notes
### [[Plots in Python]]