
# Football Player Transfer Value Predictor

## Overview
This repository contains code regarding research into predicting the football transfer value of players. It investigates how accurately a player's transfer fee represents their *intrinsic value* (calculated through demographic and performance data) and the effect of market context on this value. 

The ultimate goal of this research is to devise a more accurate, data driven method to calculate a player's transfer value and avoid the systemic overpayment that is prevalent within the football industry.

## Installation
To run the code, simply install the necessary requirements by running the `requirements.txt` file using the following command:

```bash
pip install -r requirements.txt

Data Sources
The data used to perform this investigation can be found in the raw_data folder and is sourced from Transfermarkt and FBref via Kaggle:

Transfermarkt Dataset: Contains the transfer fees (which serve as our target variable), player demographic data (age, nationality, contract information), minutes played, and basic performance statistics.

FBref Dataset (2017-2024 for Europe's Top 5 Leagues): Contains advanced performance metrics.

Data Pipeline
1. Data Generation
The Dataset_generator.py script, located in the data_formation_scripts folder, is used to build the primary dataset for this research.

It extracts all transfers that included a fee from the Transfermarkt data, alongside demographics and basic statistics.

This is then merged with the advanced performance statistics from FBref.

The data is normalized using a mapping system designed to account for performance in more difficult leagues, ultimately aiming to accurately predict intrinsic value.

The resulting dataset is saved as transfer_dataset.csv. The data is split chronologically, using transfers from 2020–2023 for training and transfers from 2024 for testing.

2. Exploratory Data Analysis (EDA)
EDA was performed to gain a deeper understanding of the data. The visualizations and insights from this stage can be found in the exploratory_data_analysis.ipynb notebook.

3. Data Preprocessing
Data preprocessing is handled in the preprocessing.py file, applying the insights gained from the EDA phase. This script generates several datasets stored in the testing-training-data folder to test dimensionality reduction techniques:

raw_data: The base preprocessed dataset.

eda_train & eda_test: Datasets created using correlation reduction.

pca_train & pca_test: Datasets created using Principal Component Analysis (PCA).

Modeling & Evaluation
Three machine learning models were trained to predict player values:

Multiple Linear Regression

Random Forest

CatBoost

The training scripts for these models can be found in the training-scripts folder. All evaluation results are printed directly to the console. Additionally, SHAP (SHapley Additive exPlanations) diagrams are generated for each model to test their capability of predicting intrinsic value, interpret the learning patterns associated with key football characteristics, and determine if these feature importances are accurately reflected in real-world market dynamics.
