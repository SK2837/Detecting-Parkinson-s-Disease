# Detecting Parkinson's Disease

## Problem Statement
Detecting Parkinson’s Disease – Machine Learning Project

### What is Parkinson’s Disease?
Parkinson’s disease is a progressive disorder of the central nervous system affecting movement and inducing tremors and stiffness. It has five stages and affects more than 1 million individuals every year in India. This chronic condition currently has no cure. Parkinson's is a neurodegenerative disorder that affects dopamine-producing neurons in the brain.

### What is XGBoost?
XGBoost is a new machine learning algorithm designed with speed and performance in mind. XGBoost stands for eXtreme Gradient Boosting and is based on decision trees. In this project, we use the `XGBClassifier` from the `xgboost` library, an implementation of the scikit-learn API for XGBoost classification.

## Objective
To build a model to accurately detect the presence of Parkinson's disease in an individual using XGBoost.

## About the Python Machine Learning Project
In this Python machine learning project, we use the libraries scikit-learn, numpy, pandas, and xgboost to build a model with an Classification Models. We will load the data, extract features and labels, scale the features, split the dataset, build the  Machine Learning Model, and then calculate the model's accuracy.

## Dataset
You will need the UCI ML Parkinson's dataset for this project. The dataset includes 24 columns and 195 records and is only 39.7 KB. You can download it from the UCI Machine Learning Repository or use the provided data file "parkinsons.data".

##Steps for Detecting Parkinson’s Disease 

Steps for Detecting Parkinson’s Disease with XGBoost
### Extract features and labels from the DataFrame
- Features: All columns except status
- Labels: Values in the status column (0 or 1)
- The dataset has 147 ones and 48 zeros in the status column
- Initialize a MinMaxScaler and scale the features between -1 and 1
- The fit_transform() method fits the data and then transforms it
- Split the dataset into training and testing sets (80/20 split)

- This classifier uses gradient boosting algorithms for classification
- Generate y_pred (predicted values for x_test) and calculate the model's accuracy
- Print out the accuracy of the model

## Model Building
We built and evaluated several machine learning models to detect Parkinson's Disease. The following steps were taken:

- Data Resampling: Using RandomOverSampler from the imblearn library to handle class imbalance.
- Feature Scaling: Using MinMaxScaler to scale features between -1 and 1.
- Dimensionality Reduction: Applying PCA to retain 95% of the variance.
- Model Training and Evaluation: Using train-test split (80-20) and training multiple models including:
   - Logistic Regression
   - Decision Tree
   - Random Forest (Gini and Entropy)
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)
   - Gaussian Naive Bayes
   -  Bernoulli Naive Bayes



