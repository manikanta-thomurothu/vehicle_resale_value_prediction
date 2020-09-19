# vehicle_resale_value_prediction
                       VEHICLE RESALE VALUE
                    PREDICTION 


ABSTRACT:
The focus of this project is developing machine learning models that can accurately predict the price of a used vehicle based on its features, in order to make informed purchases. We implement and evaluate various learning methods on a dataset consisting of the sale prices of different vehicles and models. Our results show that Random Forest model regression yield the best results and Decision Tree Regression yield moderate results. Conventional multi linear regression also yielded satisfactory results, with the advantage of a significantly lower training time in comparison to the other methods.


Motivation:
Deciding whether a used vehicle is worth the posted price when you see listings online can be difficult. Several factors, including mileage, model, year, etc. can influence the actual worth of a vehicle. From the perspective of a seller, it is also a dilemma to price a used vehicle appropriately. Based on existing data, the aim is to use machine learning algorithms to develop models for predicting used vehicle prices.



Problem:
The prices of new vehicles in the industry is fixed by the manufacturer with some additional costs incurred by the Government in the form of taxes. So, customers buying a new vehicle can be assured of the money they invest to be worthy. But due to the increased price of new vehicles and the incapability of customers to buy new vehicles due to the lack of funds, used vehicle sales are on a global increase. There is a need for a used vehicle price prediction system to effectively determine the worthiness of the vehicle using a variety of features. Even though there are websites that offers this service, their prediction method may not be the best. Besides, different models and systems may contribute on predicting power for a used vehicle’s actual market value. It is important to know their actual market value while both buying and selling.

Solution:
The proposed solution for this is Random Forest Regression (because the output is continuous, we apply regression.
Regression is continuous output based on independent variables.


Source of Data:
Kaggle is an online community for descriptive analysis and predictive modelling. It collects variety of research fields dataset from data analytic practitioners. Data scientists compete to build the best model for both descriptive and predictive analytic. It however allows individual to access their dataset in order create models and also work with other data scientist to solve various real-world analytics problems. The input dataset used in developing this model has been downloaded from Kaggle.
https://www.kaggle.com/orgesleka/used-cars-database

Structure of Dataset:
The dataset contains design characteristics of the used vehicles. This is nicely organized using common format and a standardized set of associate features of used vehicles. Structure of Dataset The dataset contains 20 columns representing the different features, 371528 samples exist. The 20 columns.

Data Pre-processing:
Coming to the step of data pre-processing we use four libraries pandas, NumPy, matplotlib, seaborn. Load the given dataset and check for any null values in dataset. Later The aim was to assess skewness of each variable and detecting outliers. The outliers were eliminated by using IQR (Inter-Quartile-Range).

Correlation Analysis:
A correlation analysis was employed in this study to examine if explanatory variables share the same linear relationship with the outcome variable in order to detect duplications of variables in the dataset. Among other things, highly correlations between variables were observed in the dataset. The Pearson correlation coefficient r, takes a range of values between +1 to -1. A value of 0 indicates that there of no relationship between the two variables. A value less than zero indicate a negative relationship and a value greater than zero connotes a positive association: that is as one unit of variable increases, so does the value of the other variable.


Partition of Data:
The dataset was partitioned into two parts for training and testing purpose: 70% of the entire dataset for training the selected models and 30% for testing purpose. Most importantly, the respective training and validation dataset were randomly sampled to circumvent sampling biasness. Particularly, 10-fold cross validation is a technique to evaluate a predictive model by partitioning the original dataset into a training set to train the model, and a validation/test dataset to assess it. 


Feature Scaling:
Feature scaling is a technique to standardize the independent feature present in the data in a fixed range. It performed to handle highly varying  units. If feature scaling is not done the machine Learning Algorithms tends to weigh greater values are larger and smaller values are lower. For Example, if and algorithm is not using feature scaling method the algorithm thought that 3000meter are greater than 3km but it not true this might Lead to give wrong predictions. So, we use Feature scaling to bring back all the values to same magnitude.


Algorithm Implementation:
As the output is continuous so we are going to use Regression algorithm we use multilinear regression algorithm and Decision Tree Regression and finally Random forest Regression Algorithm of all these we got the accuracy more for Random Forest Regression so we finalize the Random Forest Regression for this project. Random Forest (RF) is an ensemble machine learning proposed by Leo Bierman that combines predictors tree for predictive or classification task based on independent random samples of observations. In order to grow a tree, multiple random samples are drawn with replacement from the training dataset. This connotes that each tree would be grown with its version of the training dataset. Moreover, a subset of the explanatory variables is also randomly selected at each node during the learning process.
 
Evaluation metrics:
The prediction error is defined as the difference between its actual outcome value and its predicted outcome value. In this study, MSE (Mean Square Error) is used.
This is computed by taking the differences between the target and the actual algorithm outputs, squaring them and averaging over all classes and internal validation samples.

Flow Chart:





Linear Regression:
Linear regression is one of the most common algorithms for the regression task. In its simplest form, it attempts to fit a straight hyperplane to your dataset (i.e. a straight line when you only have 2 variables). As you might guess, it works well when there are linear relationships between the variables in your dataset.
In practice, simple linear regression is often outclassed by its regularized counterparts (LASSO, Ridge, and Elastic-Net). Regularization is a technique for penalizing large coefficients in order to avoid overfitting, and the strength of the penalty should be tuned.
    • Strengths: Linear regression is straightforward to understand and explain, and can be regularized to avoid overfitting. In addition, linear models can be updated easily with new data using stochastic gradient descent.
    • Weaknesses: Linear regression performs poorly when there are non-linear relationships. They are not naturally flexible enough to capture more complex patterns, and adding the right interaction terms or polynomials can be tricky and time-consuming.


Multi Linear Regression:
Multiple regression is used to examine the relationship between several independent variables and a dependent variable. While multiple regression models allow you to analyze the relative influences of these independent, or predictor, variables on the dependent, or criterion, variable, these often-complex data sets can lead to false conclusions if they aren't analyzed properly.
    • Strengths: There are two main advantages to analyzing data using a multiple regression model. The first is the ability to determine the relative influence of one or more predictor variables to the criterion value. The second advantage is the ability to identify outliers, or anomalies. 
    • Weaknesses: Any disadvantage of using a multiple regression model usually comes down to the data being used.

Regression Tree (Ensembles):
Regression trees (a.k.a. decision trees) learn in a hierarchical fashion by repeatedly splitting your dataset into separate branches that maximizethe information gain of each split. This branching structure allows regression trees to naturally learn non-linear relationships.
Ensemble methods, such as Random Forests (RF) and Gradient Boosted Trees (GBM), combine predictions from many individual trees. We won't go into their underlying mechanics here, but in practice, RF's often perform very well out-of-the-box while GBM's are harder to tune but tend to have higher performance ceilings.
    • Strengths: Decision trees can learn non-linear relationships, and are fairly robust to outliers. Ensembles perform very well in practice, winning many classical (i.e. non-deep-learning) machine learning competitions.
    • Weaknesses: Unconstrained, individual trees are prone to overfitting because they can keep branching until they memorize the training data. However, this can be alleviated by using ensembles.


Random Forest:
 
Random forests are made up of many decision trees, and there is no correlation between different decision trees. When we perform the classification task, the new input sample enters, and each decision tree in the forest is judged and classified separately. Each decision tree will get its own classification result, and which classification result of the decision tree Most, then random forest will use this result as the final result.

    • Strengths: It can come out with very high dimensional (features) data, and no need to reduce dimension, no need to make feature selection. It can judge the importance of the feature. Can judge the interaction between different features. Not easy to overfit. Training speed is faster, easy to make parallel method. It is relatively simple to implement. For unbalanced data sets, it balances the error. If a large part of the features is lost, accuracy can still be maintained.
 
 
    • Weaknesses: Random forests have been shown to fit over certain noisy classification or regression problems. For data with different values, attributes with more values will have a greater impact on random forests, so the attribute weights generated by random forests on such data are not credible.

Result:
Since we have to predict the price of a vehicle which is a continuous value, we have used regression technique to solve this problem. We have applied various regression algorithms as follows:

       ALGORITHM
                ACCURACY
Multi Linear Regression
71.3%
Decision Tree Regression
80.1%
Random Forest Regression
86.8%



Conclusion:
The main aim of this project is to build an accurate machine learning model which predicts the price of a used vehicle based on its features. The model is built using regression models. Out of all the three regression models used, we can conclude that Random Forest Regressor gives greater accuracy (86.8%).










