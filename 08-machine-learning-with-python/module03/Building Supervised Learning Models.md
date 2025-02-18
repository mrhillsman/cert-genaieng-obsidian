## Classification
___
supervised ml method using fully trained models to predict labels on new data. labels form a categorical variable with discrete values.

### Supervised Learning
- understand data in a context when answering a question
- ensures accuracy in predictions
- model adjusts the data to fit the algorithm and classifies it accordingly

### Applications of classification
- problems expressed as associations between feature and target variables
- used to build apps for
	- email filtering
	- speech-to-text
	- handwriting recognition
	- biometric identification
	- document classification

- Churn Prediction - will customer discontinue service
- Customer Segmentation - predict the "category" of a customer
- Advertising - will custom respond to a campaign

binary classifier
![[videoframe_125873.png]]

- Classification Algorithms
	- Naive Bayes
	- Logistic Regression
	- Decision Trees
	- K-nearest Neighbors
	- Support Vector Machines
	- Neural Networks

- Multiclass Prediction (outside of those designed for multiclass)
	- algorithms can be used as components of a larger multiclass classifier
	- strategies
		- one-versus-all
			- binary classifier - one for each class label in the data set; assigned a single label that defines target class
			- task - binary prediction for every data point for a one-versus-the-rest classifier
			- k-classes - k binary classifiers
		- one-versus-one
			- rather than this or those, this or not
			- given 4 classes (red, blue, green, yellow) is it red or is it blue
				- one-versus-all - is it red, is it blue

How do you decide the final label assigned to each point?

**Voting Scheme**
![[videoframe_278524.png]]

*What if there is a tie?* ==Here, we have three classes with the same number of votes. In a scenario where that is possible, it would be better to use an improved scheme, weighing each vote by the confidence level or probability assigned to that class for each classifier. Alternatively, you could try using one-versus-all classification instead.==

![[videoframe_284338.png]]

**one-versus-all** strategy

![[videoframe_213620.png]]

![[videoframe_240684.png]]


## Decision Trees
___
algorithm that can be viewed as a flow chart for classifying data points
- each node is a test
- each branch is the result of a test
- each terminal, or leaf node, assigns its data to a class

![[videoframe_107258.png]]

Training a Decision Tree
- start with a seed node and labeled training data
- find the feature that best splits the data into its pre-labeled classes, according to a pre-selected splitting criterion
- each split partitions the node's input data, each partition is passed along its branch to a new node
- repeat for each new node using each feature only once
	- tree grows until all nodes container a single class each, you run out of features, or a pre-selected stopping criterion is met

![[videoframe_167178.png]]

**Why prune?**
- overfitting if tree is too complex
- too many classes and features capturing noise and irrelevant details
- simplifies decision tree making it amenable to generalization
- more concise and easier to understand
- better predictive accuracy

**Common split measures**
- Information gain (entropy reduction)
- Gini impurity

**What is entropy**
- measure of information disorder or randomness in a data set
- how random the classes in a node are or how uncertain a feature split result is
- look for trees that have the smallest entropy in their nodes
- calculated by using the entropy formula
![[videoframe_321969.png]]

**What is information gain**
- you can consider it the opposite of entropy
- entropy of a tree before split - weighted entropy after split
- increases with decrease in entropy
![[videoframe_361618.png]]

**Advantages of Decision Trees**
- can be visualized
- interpretable
- analysis and prediction

## Regression Trees
___
analogous to a decision tree that predicts continuous values rather than discrete ones
- classification - target is categorical
- regression - target is continuous
a decision tree adapted to solve regression problems

![[videoframe_82528.png]]

- created by recursively splitting data set into subsets to maximize information gain
	- generates a tree-like structure
		- minimizes randomness of classes assigned to split nodes

![[videoframe_121679.png]]

**Predicting Values**

![[videoframe_149470.png]]
(more expensive to compute)

**Splitting Criterion**
- utilize MSE as a measure of target variance
- use weighted average of MSEs to measure the quality of each split
$$\Large{
MSE_\text{Avg} = \frac{1}{N_\text{Total}}(N_\text{Left}*MSE_\text{Left}+N_\text{Right}*MSE_\text{Right})
}
$$

average MSE equals one over the number of observations in the two split nodes, times the sum of the number of observations in the left split times the MSE of the left split, and the number of observations in the right split times the MSE of the right split.

Lower $MSE_\text{Avg}$ means lower variance and therefore higher quality of the split

**Choosing the best split**
- calculate mse for left and right nodules
- calculate weighted average of MSEs
- select split with lowest weighted MSE

![[videoframe_303288.png]]

![[videoframe_322129.png]]


___

## Support Vector Machines (SVM)
maps each data instance as a point in multi-dimensional space where the input features are represented as a value for a specific coordinate
classifies input data by identifying the hyperplane which distinctly differentiates two classes
rudimentary svm is a binary classifier but can be adapted for regression

**primary goal**
create a hyperplane that segregates a data set into two parts and finds the largest margin
- the larger the margin the better the model's accuracy on new, unseen data
- perfect separation is impossible in the real world
	- can incorporate a soft margin which allows it to tolerate misclassifications while maximizing the margin
	- the balance between maximizing margin and minimizing misclassification is controlled by a parameter C
	- smaller C; more misclassifications, softer margin
	- larger C; less misclassification, harder margin (forces stricter separation)

**2D Feature Space**
- decision boundary is a line
- margin is the distance from the hyperplane to the closest points from each class
	- these nearest-point representatives from each class are support vectors

![[videoframe_144333.png]]

**derivation of the optimization**
decision boundary: $\large{w^Tx+b=0}$ 
support vectors:


**svm pros and cons**
- advantages
	- effective in high-dimensional spaces
	- robust to overfitting
	- excels on linear separable data
	- works with weakly separable data
- limitations
	- slow for training on large data sets
	- sensitive to noise and overlapping classes
	- sensitive to kernel and regularization parameters

**applications of svm**
- image classification and handwritten digit recognition
- parsing, spam detection, sentiment analysis
- speech recognition, anomaly detection, and noise filtering


___

## k-Nearest Neighbors
takes a group of labeled data points and uses them to learn how to label other data points - classification and regression - neighbors are data points close to each other with similar features

you have to define mathematically what is meant by a neighbor

**finding the optimal k**
- test a range of values using a labeled test dataset and measure accuracy
- choose k = 1 and use the training part for modeling and calculate the prediction accuracy using all samples in test set
- repeat increasing k finding the best k for model

kNN is a lazy learner
- memorizes training data
- makes predictions based on distance to training data points
**brute force algorithm**
for each query point -> calculate distances -> sort ascending -> select top k labels -> assign class or value

**effect of k in kNN**
- too small
	- values fluctuate
	- overfitting
- too large
	- finer details lost
	- underfitting

![[videoframe_200088.png]]

![[videoframe_241764.png]]

![[videoframe_303965.png]]

![[videoframe_326554.png]]


___

## Bias, Variance, and Ensemble Models

![[videoframe_46484.png]]

![[videoframe_62593.png]]

![[videoframe_78438.png]]

![[videoframe_141807.png]]

![[videoframe_183138.png]]

**Bagging and Boosting**
- well-known ensemble methods that effectively balance bias and variance

Decision or regression trees are commonly chosen as base learners in ensemble learning
- their bias and variance can be easily adapted by altering their depth

![[videoframe_231231.png]]
- perform process multiple times
- average predictions from multiple iterations
	- reduces prediction variance
	- lowers the risk of overfitting

**Boosting**
- builds a series of weak learners
- each learner corrects the previous learner's errors
- systematically reduces prediction error
- final model is a weighted sum of weak learners
increase weights for misclassified data, decrease for correctly classified data, re-weighting focuses on correcting mistakes, update model weights based on performance
- Popular Boosting Algorithms
	- Gradient Boosting
	- XGBoost
	- AdaBoost

![[videoframe_317976.png]]

![[videoframe_345279.png]]

___

# Module 3 Summary and Highlights

Congratulations! You have completed this lesson. At this point in the course, you know: 

- Classification is a supervised machine learning method used to predict labels on new data with applications in churn prediction, customer segmentation, loan default prediction, and multiclass drug prescriptions.
    
- Binary classifiers can be extended to multiclass classification using one-versus-all or one-versus-one strategies.
    
- A decision tree classifies data by testing features at each node, branching based on test results, and assigning classes at leaf nodes.
    
- Decision tree training involves selecting features that best split the data and pruning the tree to avoid overfitting.
    
- Information gain and Gini impurity are used to measure the quality of splits in decision trees.
    
- Regression trees are similar to decision trees but predict continuous values by recursively splitting data to maximize information gain.
    
- Mean Squared Error (MSE) is used to measure split quality in regression trees.
    
- K-Nearest Neighbors (k-NN) is a supervised algorithm used for classification and regression by assigning labels based on the closest labeled data points.
    
- To optimize k-NN, test various k values and measure accuracy, considering class distribution and feature relevance.
    
- Support Vector Machines (SVM) build classifiers by finding a hyperplane that maximizes the margin between two classes, effective in high-dimensional spaces but sensitive to noise and large datasets.
    
- The bias-variance tradeoff affects model accuracy, and methods such as bagging, boosting, and random forests help manage bias and variance to improve model performance.
    
- Random forests use bagging to train multiple decision trees on bootstrapped data, improving accuracy by reducing variance.
