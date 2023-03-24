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

Our process involves:
1) Splitting the data into training and testing sets
* "y" is our "loan_status" outcome (target) and "x" is our features  
* For "loan_status", a value of 0 in the “loan_status” column means that the loan is healthy, and a value of 1 means that the loan has a high risk of defaulting 

2) Creating a Logistic Regression model with the original dataset
* Our data set is comprised of 75036 records with a "loan_status" of "healthy, and 2500 records with a "loan_status" of "high-risk"
* We will evaluate the model's perfomance by calculating the accuracy score of the model, generating a confusion matrix, and creating a classification report

3) Using the RandomOverSampler module from the imbalanced-learn library, we will resample the original data and rerun Logistic Regression on the resampled dataset
* Our resampled data set is comprised of two sets of 56277 equal records each of "healthy and "high-risk" loans
* We will evaluate the revised model's perfomance by calculating the accuracy score of the resampled model, generating a confusion matrix, and creating a classification report

### Results

**Machine Learning Model 1:**
* Balanced accuracy is a metric that one can use when evaluating how good a binary classifier is. It is especially useful when the classes are imbalanced, i.e. one of the two classes appears a lot more often than the other. This happens often in many settings such as when we are trying to detect anomalies.  It is defined as the average of recall obtained on each class. Reviewing the score of the model and predictions seems to show a high degree of confidence in the model.  But this alone does not tell us the whole story.
<img width="337" alt="Accuracy Score (Original Data)" src="https://user-images.githubusercontent.com/115101031/227609513-123a4c17-b24e-4c54-a327-a9c2270ca0b7.png">

* Reviewing the confusion matrix shows that 18678 loans were correctly predicted as "healthy" and 558 loans were correctly predicted as "high-risk".  When compared to the false positive and false negative, the model shows a pretty good degree of accuracy and precision.  Still, the false positives and false negatives could be better.
<img width="567" alt="Confusion Matrix (Original Data)" src="https://user-images.githubusercontent.com/115101031/227602922-ef3a10fc-901d-46b6-9cc8-f61cda1543fe.png">

* The classification report provides more clarity
    * The precision and recall values are highly relevant because the cost of misclassification can be high. 
    * A high precision score is important to minimize false positives, which can lead to a loss of potential customers.
    * On the other hand, a high recall score is important to minimize false negatives, which can lead to significant financial losses.
    * This logistic regression model has high accuracy (0.99). 
    * When predicting healthy loans, RECALL shows the best overall predictive result, with both the highest true positives (healthy loans) and true negatives (high-risk loans).
    * However, this results is also misleading bacause our dataset is imbalanced. Sampling our data to create a more balanced dataset may result in a higher degree of precision.
<img width="430" alt="Classification Report (Original Data)" src="https://user-images.githubusercontent.com/115101031/227609729-6126aa02-47db-4f2a-8393-852ffa9c342a.png">


**Machine Learning Model 2:**
* Balanced accuracy: 
  * Most machine learning algorithms work best when the number of samples in each class is about equal. However, if the dataset is imbalanced, the you may get high accuracy, but the results are misleading as they reflect mostly the majority class, but you fail to represent the minority class, which is most often the point of creating the model in the first place. For example, if the class distribution shows that 99% of the data has the majority class, then any basic classification model like the logistic regression will not be able to identify the minor class data points.
  * We can hypothesize that in our analysis, high-risk loans are a minor class. In a dataset with highly unbalanced classes, the classifier will always “predict” the most common class and, as a result, will have a high accuracy rate.
  * Resampling consists of removing samples from the majority class (under-sampling) and/or adding more examples from the minority class (over-sampling). In over-sampling, random records from the minority class are duplicated. RandomOverSampler generates new samples by random sampling with the replacement of the currently available samples.
  * By resampling our data, the accuracy of our predictions approach 100%.
<img width="338" alt="Accuracy Score (Resampled Data)" src="https://user-images.githubusercontent.com/115101031/227609576-1cf8bbba-4c75-492e-bb5f-8189eca67d24.png">

* he confusion matrix for our resampled dataset shows a small decrease in the predictability from our original dataset in our true positives (healthy loans = 18678) and increase in our true negatives (high-risk loans = 558).  Resampling our dataset has increased our ability to predict high risk loans, but also increases the number of healthy loans that are excluded and labeled as high-risk.  
<img width="561" alt="Confusion Matrix (Resampled Data)" src="https://user-images.githubusercontent.com/115101031/227604332-19dff1b6-23c4-4bae-8231-01d958a33e5a.png">

* The classification report provides more clarity
    * This logistic regression model shows exceptional accuracy (1.00). 
    * When predicting healthy loans, both PRECISION and RECALL shows a superior result (both in the proportion of actual and predicted positives was accurate, producing no falso positives or negatives), with both the highest true positives (healthy loans) and true negatives (high-risk loans).
    * For "high-risk" loans, precision has not changed, meaning that the proportion of positive identifications did not improve with the resampled data.  This may indicate that training the model on fictitious data may be impacted when testing it against new test data.
<img width="433" alt="Classification Report (Resampled Data)" src="https://user-images.githubusercontent.com/115101031/227607597-2b6a2777-e8e4-4261-921d-9fba183b296a.png">

### Summary

Looking at the two classification reports for the original dataset and resampled dataset, it looks as if model performance increased on the test data as a result of resampling the data. We get strong precision and perfect recall on the test dataset, which is a good indication about how well the model is likely to perform in real life.

The resampled data improves on the initial Logical Regression model. The balanced accuracy score increased from 0.942 to 0.995. It improves the predictability of true negatives meaning that it is more effective at distinguishing high-risk loans with high recall and accuracy.

The resampled model generated an accuracy score of 100% which turned out to be higher than the model fitted with imbalanced data. The oversampled model performs better because it catches mistakes such as labeling high-risk loans as healthy loans.

A lending company might favour a model that favours the ability to predict high-risk loans, thus minimizing potential financial losses for the company. The trade-off might be a higher number of healthy loans being tossed out as false negatives (predicted as risky, when they are healthy), but that would be an acceptable loss over predicting a high number of false positives (high-risk loans identified as healthy loans), which might be costly for a lending company when the customer defaults on their loans. Ultimately, increasing our precision for high-risk loans is the favourable choice.

However, it is important to acknowledge that one potential difficulty with oversampling is that, given a real-world dataset, the synthesized samples may not truly belong to the minority class. As a result, training a classifier on these samples while pretending they represent minority may result in incorrect predictions when the model is used in the real world.  Perhaps one option to test, is rather than oversampling the minority class, we undersample the majority class, by artificially decreasing the number of observations that take on a particular value or range of values for that variable. We can evalluate what values are overrepresented in the dataset and decrease the number of observations that take on that value or range of values.
