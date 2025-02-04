## Overview

- Explain machine learning
- Identify techniques
- Describe applications of machine learning

ML - subset of AI that uses algorithms and requires **feature engineering** by practitioners
DL - distinguishes itself from machine learning by using many-layered neural networks that automatically extract features from highly complex, unstructured big data

- Supervised Learning - trains on labeled data
- Unsupervised Learning - works without labels by finding patterns in data
- Semi-supervised Learning - trains on a relatively small subset of data that is already labeled, and iteratively retrains itself by adding new labels that it generates with reasonably high confidence
- Reinforcement Learning - simulates an artificially intelligent agent interacting with its environment and learns how to make decisions based on feedback from its environment

Selecting Machine Learning Techniques
- Factors
	- problem you're trying to solve
	- type of data you have
	- available resources
	- desired outcome
- Techniques
	- Classification - predict the class or category of a case; benign or malignant
	- Regression/Estimation - predict continuous values; price of house based on its characteristics
	- Clustering - groups of similar cases; similar patients, customer segmentation
	- Association - items or events that co-occur; grocery items usually bought together by a particular customer or market segment
	- Anomaly Detection - discover abnormal or unusual cases; credit card fraud detection
	- Sequence Mining - predict the next event; clickstream analytics in website
	- Dimension Reduction - reduce data size, particularly number of features needed
	- Recommendation Systems - associate preferences with others who have similar tastes and recommends new items

## Machine Learning Model Lifecycle

- Problem Definition
- ETL
	- Data Collection
	- Data Preparation
- Model Development and Evaluation
- Model Deployment

### Problem Definition
user story to define the problem such that the ml solution aligns with the client needs

### Data Collection
determine what kind of the data the company has and identify the sources it will come from
- user data
- product data
- other data
wrangle, aggregate, join, merge, and map data onto one central source reducing the need to deal with multiple databases every time we need to pull data

### Data Preparation
preparing a somewhat final version of the data as most of the time data from multiple sources will contain errors, different formatting, and missing data
- clean to filter out irrelevant data
- extreme values are removed to avoid influencing the data set
- missing values removed or randomly generated
- each data column is in the proper format
- additional feature creation
- exploratory data analysis
	- create plots to visually identify patterns
	- validate data based on information provided by SMEs
	- correlation analysis to identify what variables or features are "important"
	- how data should be split for training and testing
		- random split or recent transaction as test set

### Model Development
explore existing frameworks
- content-based filtering - similarity between products based on product content; identify some correlation between products and purchases for example
- collaborative filtering - similarity between two users based on how they view a product; group users based on characteristics (age, region, skin type, products rated or puchased), take average rating for existing members and assume a new user will be around the average so recommend
- combination of content-based and collaborative filterings

### Model Evaluation
test model is performing well and recommendations represent what the users want
- tune and test on the data set kept earlier for testing
- experiment on a group of users and get feedback
	- rate recommendations
	- collect data on who clicked and bought recommended products
	- other necessary metrics

### Model Deployment
- deploy
- track/monitor quality
- re-train if necessary
- expand if necessary

## Data Scientist v AI Engineer

- Data Scientist - descriptive + predictive
	- Storyteller
	- Take massive amounts of messy real world data and use mathematical models to translate this data into insights
	- Use a lot of descriptive analytics to describe the past
		- exploratory data analysis (EDA)
		- clustering
	- Structured data mostly
		- cleaning and preprocessing important
	- 100s of models and different algorithms
		- more narrow
		- lot smaller generally
		- less compute to train and infer
		- less time to train
	- Typical process
		- pick use case
		- select relevant data
		- train and validate model; feature engineering, cross-validation, hyperparameter training, etc
		- deploy

- AI Engineer (Generative AI Engineer) - prescriptive + generative
	- System builder
	- Use foundation models to build generative AI system that help to transform business processes
	- Prescriptive
		- decision optimization
		- recommendation engine
	- Unstructured data mostly
	- Generally just one model; foundation model
		- more wider scope
		- model a lot larger
		- lots of GPUs + compute
		- weeks to months of training time
	- Typical process
		- pick use
		- skip directly to working with pre-trained model (democratization)
		- prompt engineering
			- chaining
			- PEFT
			- RAG
			- Agent
		- embed in larger system or workflow

## Tools for Machine Learning

What is data?

A collection of **raw** facts, figures, or information used to draw insights, inform decisions, and fuel advanced technologies. Central to every machine learning algorithm and the source of all the information the algorithm uses to discover patterns and make predictions.

Tools?
- provide functionality for machine learning pipelines
- data preprocessing
- building, evaluating, optimizing, and implementing models
- simplify complex tasks
	- handling big data
	- statistical analyses
	- making predictions

- Pandas - data manipulation and analysis
- Scikit-Learn - supervised and unsupervised learning algorithms for linear regression
- Python - analyzing and processing data, developing models
- R - statistical learning, data exploration
- Julia - parallel and distributed numerical computing support
- Scala - processing big data and building ml pipelines
- Java - ml application deployment
- JavaScript - models in browsers to service client-side applications

Data Processing and Analytics
- PostgreSQL
- Hadoop
- Spark
- Kafka
- Pandas
- NumPy
Data Visualization
- Matplotlib
- Seaborn
- ggplot2
- Tableau
Machine Learning
- NumPy
- Pandas
- SciPy
- Scikit-learn
Deep Learning
- TensorFlow
- Keras
- Theano
- PyTorch
Computer Vision
- OpenCV
- Scikit-Image
- TorchVision
Natural Language Processing
- NLTK
- TextBlob
- Stanza
Generative AI
- HuggingFace Transformers
- ChatGPT
- DALL-E
- PyTorch

## Scikit-learn ML Ecosystem

(normalize - finding inconsistent data, missing values, and outliers)

ML Ecosystem - the interconnected tools, frameworks, libraries, platforms, and processes that support developing, deploying, and managing ml models

Machine Learning Pipeline Tasks
- Data Preprocessing
- Train or Test Splitting
- Model Setup and Fitting
- Hyperparameter Tuning with Cross-Validation
- Prediction
- Evaluation
- Model Export

Scikit-learn Workflow
(given: data set x and target variable y as NumPy arrays)
```python
from sklearn import preprocessing
# scale data by standardizing it
X = preprocessing.StandardScalar().fit(X).transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# instantiate classifier model
from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)

clf.fit(X_train, y_train)
# clf.predict(X_test)
yhat = clf.predict(X_test)

# what is a confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, yhat, labels=[1,0]))

# isn't pickle deprecated or not allowed/frowned upon because of security reasons
import pickle
s = pickle.dumps(clf)
```