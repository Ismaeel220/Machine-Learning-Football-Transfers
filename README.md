# Football Player Transfer Value Predictor

## Overview
This repository contains code regarding research into predicting the football transfer value of players. It investigates how accurately a player's transfer fee represents their *intrinsic value* (calculated through demographic and performance data) and the effect of market context on this value. 

The ultimate goal of this research is to devise a more accurate, data driven method to calculate a player's transfer value and avoid the systemic overpayment that is prevalent within the football industry. By taking into account market contexts which are prevalent in any financial market and are unavoidable and adding these onto a players baseline worth (intrinsic value)

## Installation
To run the code, simply install the necessary requirements by running the `requirements.txt` file using the following command:

```bash
pip install -r requirements.txt
```

## Data Sources
The data used to perform this investigation can be found in the `raw_data` folder and is sourced from Transfermarkt and FBref via Kaggle:
* **[Transfermarkt Dataset](https://www.kaggle.com/datasets/davidcariboo/player-scores):** Contains the transfer fees (target variable), player demographic data (age, nationality, contract information), minutes played, and basic performance statistics.
* **[FBref Dataset (2017-2024 for Europe's Top 5 Leagues)](https://www.kaggle.com/datasets/akshankrithick/fbref-2017-2024-for-europes-top-5-leagues):** Contains advanced performance metrics.

## Data Pipeline & Feature Engineering
The `Dataset_generator.py` script (located in `data_formation_scripts`) builds the primary dataset applying insights from EDA as well as other contexts to ensure data is accurate and to test the central hypothesis of research. 

Key engineering steps include:
* **League Coefficient Normalisation:** Performance metrics are scaled based on league difficulty to ensure comparability across different competitions.
* **Contextual Engineering ("Wealth Tax"):** A specific feature was engineered using Leave-One-Out (LOO) encoding to represent the purchasing club's historical spending power. This prevents data leakage while accurately capturing the "market premium" wealthy clubs are often forced to pay.

Data preprocessing is handled in `preprocessing.py`. To test the impact of the curse of dimensionality, three dataset variations are generated and stored in the `testing-training-data` folder:
1. `raw_data`: The base preprocessed dataset.
2. `eda_train` / `eda_test`: Datasets reduced using correlation filtering.
3. `pca_train` / `pca_test`: Datasets reduced using Principal Component Analysis (PCA).

Data is split chronologically: transfers from **2020–2023 for training** and **2024 for testing**.

## Methodology: A Two Phase Approach
A major limitation of existing research is treating the observed transfer fee as an absolute ground truth, ignoring market inflation. To address this, model training was conducted in two distinct phases:

* **Phase 1 (Intrinsic Value):** Models were trained exclusively on intrinsic player data (performance metrics, age, contract length) to establish a baseline of what a player is theoretically worth based on their sporting output.
* **Phase 2 (Contextual Value):** The best-performing baseline models were retrained with the addition of the contextual "wealth tax" feature to determine how much of the final fee is dictated by the buying club's financial status rather than the player's ability.

## Modeling & Evaluation
Three machine learning models were evaluated across the dataset variations: **Multiple Linear Regression**, **Random Forest**, and **CatBoost**. 

The training scripts can be found in the `training-scripts` folder. Evaluation goes beyond standard error metrics (RMSE, MAE) by utilizing **SHAP (SHapley Additive exPlanations) diagrams** to verify that the models learned logical, real-world football characteristics (e.g., verifying that longer contracts and higher goal contributions positively impact intrinsic value).

## Key Findings
* **Tree Based models are most accurate:** Ensemble methods (specifically CatBoost and Random Forest) proved highly capable of modeling non linear intrinsic value without the need for strict dimensionality reduction like PCA.
* **The Market Premium:** Intrinsic factors alone only explained approximately **30%** of the variance in observed transfer fees. 
* **Context is Decisive:** When the "wealth tax" (club spending power) was introduced in Phase 2, the variance explained by the model nearly doubled (e.g., CatBoost $R^2$ jumped from 0.27 to 0.56). This proves that while intrinsic data provides a rational baseline, external market factors play a massive, quantifiable role in inflating final transfer fees.
