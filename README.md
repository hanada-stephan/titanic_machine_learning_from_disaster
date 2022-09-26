# Breast cancer prediction: Project overview

**Tags: Adaboost, random forest, decision tree, EDA, feature engineering, precision, recall**

- This notebook is part of Titanic - Machine Learning from Disaster Kaggle's competition. 
- Created a model that predicted whether the passenger survived or not. 
- Models: decision tree, random forest, and AdaBoost.


## Code and resources

Platform: Jupyter notebooks

Python version: 3.7.6

Packages: itertools, os, matplotlib, pandas, numpy, seaborn and sklearn

## Data set

**Data set URL: https://www.kaggle.com/competitions/titanic**

## Data cleaning

I created the following features:
- number of relatives: number of parents + number of siblings

I looked for missing values which there were in the "Age" (177 out of 687 - 26%) and "Cabin" (687 out of 687 - 100%) training data set's columns. The cabin feature was not relevant for us but Age is a crucial feature for the model and 25% is a high percentage to drop these rows. Rather I filled these records with the passenger's median value. I did the same process for Age missing values in the test set and the only one null value for the Fare feature.

Then I looked for the data set statistics to search for outliers. Also, I checked:
- the percentage of survivors and non-survivors by Pclass, gender, and place of boarding;
- checked the age distribution by gender (not including the filled records);
- fare, number of siblings, parents, and relatives distribution;

## Model building

First, I standardized the data to avoid model bias and then split the data into train and test sets with a test size of 25%.

I created: 
- **decision tree**: to use as our baseline. Then I limited max_depth and min_sample_leaf. 
- **Random forest**: with the best hyperparameters used in the previous models and tested different numbers of estimators of 50, 100, and 200.
- **AdaBoost**: with the same number of estimators used in the random forest algorithm.

## Model performance

The best model performances were random forests with 50 and 100 estimators. But for the submission, I chose the latter due to its robustness.

|MODEL|PRECISION|RECALL|
|-----|---------|------|
|Decision tree|0.687|0.667|
|Decision tree max depth of 10|0.706|0.606|
|Decision tree max depth of 5|0.710|0.667|
|Random forest 200 estimators & max depth of 5|0.759|0.697|
|**Random forest 100 estimators & max depth of 5**|**0.761**|**0.677**|
|Random forest 50 estimators & max depth of 5|0.775|0.697|
|AdaBoost 50 estimators|0.736|0.727|
|AdaBoost 100 estimators|0.727|0.727|
|AdaBoost 200 estimators|0.737|0.707|
