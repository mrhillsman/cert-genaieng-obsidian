## Data Analysis with Python
### Categories of Libraries
- Scientific Computing
	- Pandas - effective data manipulation and analysis
		- primary instrument of Pandas is a two dimensional table consisting of column and row labels, which is called a data frame; designed to provide easy indexing functionality
	- NumPy - arrays and matrices
	- SciPy - functions for some advanced math problems (integrals, differential equations, optimizations)
- Visualization
	- MatPlotLib - plots and graphs (most popular)
	- Seaborn - based on matplotlib (plots, heat maps, time series, violin plots)
- Algorithmic
	- Scikit-learn - tools for statistical modeling (machine learning; regression, classification, etc) - built on numpy, scipy, and matplotlib
	- Statsmodels - explore data, estimate statistical models, and perform statistical tasks

### Importing and Exporting Data

```python
import pandas as pd
url = "https://address.com/for/some/file.csv"

# header=None will 0 index the headers
# use when there are no headers in the data set
df = pd.read_csv(url, header=None)
# Print only the first n rows, 5 by default, as the data set might be large
df.head()

# We can set the column headers manually since it is more useful
headers = ["some", "list", "of", "headers"]
df.columns=headers
df.head()

# Save to file
path="/some/path/to/file.csv"
df.to_csv(path)
```

### Basic Insights from Data
>understand the data before beginning analysis - overview and locate potential issues

| **Pandas Type**           | **Native Python Type**                     | **Description**                  |
| ------------------------- | ------------------------------------------ | -------------------------------- |
| object                    | string                                     | numbers and strings              |
| int64                     | int                                        | numeric characters               |
| float64                   | float                                      | numeric characters with decimals |
| datetime64, timedelta[ns] | N/A (see datetime module in Python stdlib) | time data                        |

- check
	- data types of features
		- why check data types?
			- pandas automatically assigns types based on the encoding it detects from the original data table - automatically assigned type could be wrong
			- allows experienced data scientist to see with Python functions can be applied to a specific column - verify Python method compatibility

```python
import pandas as pd
# create a data frame named df
df.dtypes # returns data type of each column in a series
df.describe() # returns a statistical summary
```

##### `dataframe.describe()`
- returns a statistical summary
	- count - number of terms in the column
	- mean - average column value
	- std - standard deviation 
	- min - lowest value in column
	- 25%, 50%, and 75% - boundary of each of the quartiles
	- max - highest value in column
- by default skips rows and columns that do not contain numbers
- enable a summary of all the columns including features of object type `dataframe.describe(include='all')`
	- unique, top, and freq added
	- unique - number of distinct objects in the column
	- top - most frequently occurring object
	- freq - number of times the top object appears in the column
- NaN - that particular statistical metric cannot be calculated for that specific column data type
##### `dataframe.info()`
- gives a concise summary of the data frame, prints information about a data frame including the index D type and columns, non-null values, and memory usage.

### Accessing Databases
- DB-API is Python standard API for accessing relational databases
	- allows you to write a single program that works with multiple kinds of relational databases
		- two main concepts
			- connection objects - connect to a database and manage your transactions
			- query objects - cursor objects are used to run queries
				- open a cursor object and then run queries
				- cursor works similar to a cursor in a text processing system, where you scroll down in your result set and get your data into the application
				- cursors are used to scan through the results of a database
		- methods used with connection objects
			- cursor - returns a new cursor object using the connection
			- commit - commit any pending transaction to the database
			- rollback - causes the database to roll back to the start of any pending transaction
			- close - close a database connection

```python
from dmodule import connect

# Create connection object
conn = connect('databasename', 'username', 'password')

# Create a cursor object
curs = conn.cursor()

# Run queries
curs.execute('select * from mytable')
results = curs.fetchall()

# Free up resources
curs.close()
conn.close()
```

### Cheat Sheet

```python
# Package/Method: Read CSV data set
# Description: Read the CSV file containing a data set to a pandas data frame
df = pd.read_csv(<CSV_path>, header=None) # load without header
df = pd.read_csv(<CSV_path>, header=0) # load using first row as header

# Package/Method: Print first few entries
# Description: Print the first few entries (default 5) of the pandas data frame
df.head(n)  # n=number of entries; default 5

# Package/Method: Print last few entries
# Description: Print the last few entries (default 5) of the pandas data frame
df.tail(n)  # n=number of entries; default 5

# Package/Method: Assign header names
# Description: Assign appropriate header names to the data frame
df.columns = headers

# Package/Method: Replace "?" with NaN
# Description: Replace the entries "?" with NaN entry from Numpy library
df = df.replace("?", np.nan)

# Package/Method: Retrieve data types
# Description: Retrieve the data types of the data frame columns
df.dtypes

# Package/Method: Retrieve statistical description
# Description: Retrieve the statistical description of the data set. Defaults use is for only numerical data types. Use `include="all"` for all variables
df.describe()  # default use
df.describe(include = "all")

# Package/Method: Retrieve data set summary
# Description: Retrieve the summary of the data set being used, from the data frame
df.info()

# Package/Method: Save data frame to CSV
# Description: Save the processed data frame to a CSV file with a specified path
df.to_csv(<output CSV path>)
```