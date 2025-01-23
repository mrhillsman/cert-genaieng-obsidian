# matplotlib

```python
import matplotlib.pyplot as plt
```

## Standard Line

```python
# expects two arrays as input, `x` and `y`, both of the same size
# `x` is the independent variable
# `y` is the dependent variable
# graph is plotted as shortest line segments joining the `x,y` point pairs ordered in terms of the variable `x`
plt.plot(x, y)
```

![[standard-line.png]]

## Scatter

```python
# present the relationship between two features on a two-dimensional plane. the predictor feature is plotted on the x-axis, while the target feature is plotted on the y-axis.

# used in either of the following situations:
# - we have paired numerical data
# - there are multiple values of the dependent variable for a unique value of an independent variable
# - determining the relationship between variables in some scenarios
plt.scatter(x, y)
```

![[scatter.png]]

## Histogram

```python
# "binned" form of viewing data
# bins can be a number, data points marking bin edges, etc
# x-axis represents bins, y-axis represents number of elements in each bin
plt.hist(x, bins) # left diagram
plt.hist(x, bins, edgecolor='black') # right diagram
```

![[histogram.png]]

## Bar

```python
# y-axis represents average value of data points for a particular category
# x-axis represents the number of elements in the different categories
plt.bar(x, height)
```

![[bar.png]]

## Pseudo Color (Heatmap)

```python
# displays matrix data as an array of colored cells (known as faces)
# x and y 
plt.pcolor(C)
```

![[07-data-analysis-with-python/plots/heatmap.png]]
# seaborn

```python
import seaborn as sbn
```

## Regression

```python
# draws a scatter plot of two variables then fits the regression model and plots the resulting regression line along with a 95% confidence interval for that regression
# x and y parameters can be shared as the dataframe headers to be used, and the data frame itself is passed to the function as well.
sbn.regplot(x='header_1', y='header_2', data=df)
```

![[positive-linear-regression.png]]

## Box and Whisker

```python
# shows the distribution of quantitative data in a way that facilitates comparisons between features or across levels of a categorical feature
# box shows the quartiles of the dataset while the whiskers extend to show the rest of the distribution, except for points that are determined to be "outliers" which appear as dots
sbn.boxplot(data=df, x='gpu', y='price')
```

![[box.png]]

![[box-example.png]]
## Residual

```python
# used to display the quality of polynomial regression
# will regress y on x as a polynomial regression and then draw a scatterplot of the residuals
# residuals are the differences between the observed values of the dependent variable and the predicted values obtained from the regression model
# a residual is a measure of how much a regression line vertically misses a data point, meaning how far off the predictions are from the actual data points
sbn.residplot(data=df, x='header_1', y='header_2')
sbn.residplot(x=df['header_1'], y=df['header_2']) # alternative
```

![[residual.png]]

## KDE (kernel density estimate)

```python
# creates a probability distribution curve for the data based upon its likelihood of occurrence on a specific value - created for a single vector of information
# 
sbn.kdeplot(X)
```

![[kernel-density-estimate.png]]

## Distribution

```python
# has the capacity to combine the histogram and the KDE plots
# creates the distribution curve using the bins of the histogram as a reference for estimation
# you can optionally keep or discard the histogram from being displayed
sbn.distplot(X, hist=False) # left diagram
sbn.distplot(X, hist=True) # right diagram
```

![[distribution.png]]