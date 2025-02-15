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

**Tie?** (check weight of vote, use one-versus-all instead, etc)
![[videoframe_284338.png]]

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
