# Supervised Machine Learning 
### Case Study: Analysis of Credit Risk in the Lending Industry
![AlternativeLending_FINS_Funding](https://user-images.githubusercontent.com/115101031/227517220-bb8ce17f-930e-4ab7-b566-dbca2150f2f7.png)


## Big Picture (Methodology)

The main difference between supervised and unsupervised machine learning is that supervised learning uses labeled data, which contains both Features (X variables) and a Target (y variable). 

When using supervised learning, an algorithm iteratively learns to predict the target variable given the features and modifies for the proper response in order to "learn" from the training dataset. This process is referred to as Training or Fitting. Supervised learning models typically produce more accurate results than unsupervised learning but they do require human interaction at the outset in order to correctly identify the data. If the labels in the dataset are not correctly identified, supervised algorithms will learn the wrong details.

Unsupervised learning models, on the other hand, work in an autonomous manner to identify the innate structure of data that has not been labeled. It is important to keep in mind that validating the output variables still calls for some level of human involvement. For instance, an unsupervised learning model can determine that customers who shop online tend to purchase multiple items from the same category at the same time. However, a human analyst would need to check that it makes sense for a recommendation engine to pair Item X with Item Y. 

![Supervised_machine_learning_078cb43a05](https://user-images.githubusercontent.com/115101031/227542594-0dab8d96-3b45-4ef1-86f6-4ea5d953f785.png)

There are two prominent use-cases for supervised learning: Classification and Regression. In both the tasks a supervised algorithm learns from the training data to predict something. If the predicted variable is discrete such as is the case with our problem, where we are trying to differentiate between "healthy and "high-risk lonas, then a classification algorithm is required.

The objective of supervised learning is to forecast results for new data based on a model that has learned from labeled training dataset. The kind of outcomes you can anticipate are known up front in the shape of labeled data. In our analysis, we have split our dataset of collected lending data into training and testing data.  In our first approach, we use Logistic Regression on our original training data to develop a model that helps us, when given particular features, if we can accurately predict healthy loans from high-risk loans.

Linear regression is one of the simplest machine learning algorithms available, it is used to learn to predict continuous value (dependent variable) based on the features (independent variable) in the training dataset. The value of the dependent variable which represents the effect, is influenced by changes in the value of the independent variable.  Logistic Regression is a special case of Linear Regression where target variable (y) is discrete / categorical such as 1 or 0, True or False, Yes or No, Default or No Default.  A log of the odds is used as the dependent variable. Using a logit function, logistic regression makes predictions about the probability that a binary event (like the one in this problem) will occur.

One of the challenges with binary events is the bias that may be created if our dataset is imbalanced (ie. majority and minority classifiers).  This will skew the outcome in favour of the majority classifiers, and consequently give us a false reading of accuracy in our model.  In this case, the predictive accuracy of our model is not really based on our data/the features, but rather the majority cases in our data.  In such cases, scaling oour features to normalize the range of features in a dataset is highly beneficial. Real-world datasets often contain features that are varying in degrees of magnitude, range and units. Therefore, in order for machine learning models to interpret these features on the same scale, we need to perform feature scaling.

Resampling strategies address class imbalance at the data level, by resampling the dataset to reduce the imbalance ratio.  The resampling of an imbalanced dataset occurs before the training of the prediction model and can be seen as a data preprocessing step. Numerous methods have been proposed for resampling imbalanced datasets, which can be categorized into three main strategies: oversampling, undersampling, and hybrid strategies.  In our analysis, we chose an oversampling stategy using RandomOverSampler from the imblearn (imbalanced-learn) library.  Oversampling consists in artificially increasing the proportion of samples from the minority class by randomly duplicating samples of the minority class.  In normalizing our data, both the accuracy, recall, and reliability of our predictions are expected to improve, ensuring that our outcomes are far more reflective of our data as a whole.

Sources:
* https://www.datacamp.com/blog/supervised-machine-learning
* https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_6_ImbalancedLearning/Resampling.html


## Credit Risk Analysis Report

### Overview of the Analysis

In this analysis, we use various techniques to train and evaluate a model based on loan risk. The dataset is comprised of historical lending activity from a peer-to-peer lending services company.  The goal is to build a model that can identify the creditworthiness of borrowers into two categories: 
* Healthy Loans
* High-risk Loans

We will determine the best model by using Logistic Regression with:
* Our original dataset, and 
* A resampled dataset



The f
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
* Describe the stages of the machine learning process you went through as part of this analysis.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).

### Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.



* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

### Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.
