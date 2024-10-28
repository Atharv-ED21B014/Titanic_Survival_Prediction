# TechConative_Task
This project involves building a machine learning model to predict the survival of passengers on the Titanic. Using the classic Kaggle Titanic dataset, we explore data preprocessing, feature engineering, and model building to optimize predictive accuracy.

Table of Contents
Project Overview
Data Description
Installation
Data Preprocessing
Exploratory Data Analysis (EDA)
Feature Engineering
Model Building
Evaluation and Results
Conclusions
Future Work
Project Overview
This repository contains a machine learning pipeline designed to predict the survival of passengers aboard the RMS Titanic. The model leverages Python libraries such as pandas, numpy, seaborn, and scikit-learn, with an emphasis on feature preprocessing and model optimization using grid search.

Data Description
The dataset consists of three files:

train.csv: Training data used for model training and validation.
test.csv: Test data on which predictions are made.
gender_submission.csv: A sample submission file for Kaggle competition format.
Features
The main features in the dataset are:

PassengerId: Unique ID for each passenger.
Pclass: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).
Name: Name of the passenger.
Sex: Gender of the passenger.
Age: Age of the passenger.
SibSp: Number of siblings/spouses aboard.
Parch: Number of parents/children aboard.
Ticket: Ticket number.
Fare: Fare paid for the ticket.
Cabin: Cabin number (if available).
Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).
The target variable is Survived, which indicates whether the passenger survived (1) or did not survive (0).

Installation
To run the code in this repository, the following Python libraries are required:

pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost (for XGBoost classifier)
Install them using:

bash
Copy code
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
Data Preprocessing
Steps performed:

Handling Missing Values:

Imputed missing Age values using the median strategy.
Filled missing Embarked values with the most frequent value.
Replaced missing Fare values with the median.
Processed Cabin feature to extract the first letter of the cabin number, and filled missing values with 'X'.
Dropping Irrelevant Columns:

Removed Ticket and Name columns, as they are not directly useful for prediction.
Dropped PassengerId for model training to avoid any bias.
Feature Transformation:

Binned Age into categories: 'Child', 'Adult', 'Middle-aged', 'Senior'.
Created a new feature FamilySize by adding SibSp and Parch values.
Exploratory Data Analysis (EDA)
Some visualizations were used to understand the relationship between features and survival rates:

Survival Rate Distribution: Visualized with bar plots and pie charts.
Class vs. Survival: First-class passengers had higher survival rates.
Gender vs. Survival: Females had a higher likelihood of survival.
Age Group vs. Survival: Children had a higher survival probability compared to adults.
Feature Engineering
New features were created to potentially enhance model performance:

Cabin was transformed into a categorical feature representing the deck.
Age_group was encoded using label encoding.
FamilySize was calculated as the total number of family members aboard.
Model Building
A machine learning pipeline was set up using scikit-learn:

Preprocessing: Used ColumnTransformer to apply scaling to numeric features and one-hot encoding to categorical features.
Classification Model: Tried various classifiers:
RandomForest
GradientBoosting
AdaBoost
XGBoost
Hyperparameter Tuning: Used GridSearchCV to find the optimal parameters for the classifiers. The best performing model was RandomForest with a maximum depth of 10 and 200 estimators.
