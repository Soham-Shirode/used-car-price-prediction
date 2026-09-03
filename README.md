# Used Car Price Prediction

### Used-Car Price Prediction using Lasso Regression and Streamlit

A machine learning project that predicts the selling price of used cars from vehicle specifications and usage history.

<p align="center">

<img src="https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat-square&logo=pandas&logoColor=white">
<img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=flat-square&logo=scikit-learn&logoColor=white">
<img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white">
<img src="https://img.shields.io/badge/Git-GitHub-181717?style=flat-square&logo=github&logoColor=white">

</p>

---

## Overview

This project implements an end-to-end machine learning pipeline for used-car price prediction.

The workflow includes data cleaning, exploratory data analysis, feature preprocessing, regression model comparison, hyperparameter tuning, model serialization, and an interactive Streamlit application.

The final model is a **Lasso Regression model trained on a log-transformed target variable**.

---

## Results

The final model was evaluated on an unseen test set.

| Metric | Result |
|---|---:|
| R² Score | **0.891** |
| Mean Absolute Error | **₹113,602** |
| Root Mean Squared Error | **₹258,994** |

**Model:** Lasso Regression  
**Target:** `log1p(selling_price)`  
**Alpha:** `0.001`  
**Cross-validation:** 5-fold GridSearchCV  
**Features:** 163

The best cross-validation R² during hyperparameter tuning was **0.925**. The final test-set R² was **0.891**.

---

## Dataset

The original dataset contains **15,411 records** and **14 columns**.

### Features

| Feature | Description |
|---|---|
| `car_name` | Full vehicle name |
| `brand` | Vehicle manufacturer |
| `model` | Vehicle model |
| `vehicle_age` | Age of the vehicle |
| `km_driven` | Distance driven |
| `seller_type` | Seller category |
| `fuel_type` | Fuel type |
| `transmission_type` | Transmission type |
| `mileage` | Vehicle mileage |
| `engine` | Engine displacement |
| `max_power` | Maximum engine power |
| `seats` | Number of seats |

### Target

```text
selling_price
```

After removing two invalid records with `seats = 0`, **15,409 records** were used for modeling.

---

## Exploratory Data Analysis

The dataset was analyzed to understand feature distributions, relationships, and data quality.

Key analysis included:

- Target price distribution
- Numerical feature distributions
- Correlation analysis
- Feature vs. price relationships
- Categorical feature analysis
- Group-wise price comparisons
- Outlier investigation

The target variable was strongly right-skewed, which motivated the use of a logarithmic target transformation.

---

## Data Preprocessing

### Data Cleaning

- Removed `Unnamed: 0`
- Removed `car_name` due to overlap with `model`
- Removed invalid records where `seats = 0`
- Investigated extreme values in numerical features

### Categorical Encoding

One-hot encoding was applied to:

```text
brand
model
seller_type
fuel_type
transmission_type
```

### Train/Test Split

The dataset was split into:

```text
80% Training
20% Testing
```

with `random_state = 42`.

### Feature Scaling

`StandardScaler` was used to standardize the feature matrix.

The scaler was fitted only on the training data to prevent data leakage.

### Target Transformation

The selling price was transformed using:

```python
y_log = np.log1p(y)
```

Predictions were converted back to the original price scale using:

```python
np.expm1(prediction)
```

---

## Model Development

The following regression models were evaluated:

| Model | Role |
|---|---|
| Linear Regression | Baseline |
| Ridge Regression | L2 regularization |
| Lasso Regression | L1 regularization |
| Tuned Lasso | Hyperparameter optimization |
| Log-target Lasso | Log-transformed target |
| Tuned Log-target Lasso | Final model |

### Hyperparameter Tuning

`GridSearchCV` with 5-fold cross-validation was used to tune the Lasso regularization parameter.

The final configuration selected:

```text
alpha = 0.001
```

---

## Streamlit Application

The trained model is integrated into a Streamlit web application.

The application accepts:

- Brand
- Model
- Vehicle age
- Kilometers driven
- Seats
- Fuel type
- Transmission type
- Seller type
- Mileage
- Engine
- Maximum power

The selected brand dynamically determines the available models using the dataset.

The application then applies the same preprocessing pipeline used during training before generating the predicted price.

---

## Model Artifacts

The trained model and preprocessing components are stored using Joblib.

```text
model/
├── lasso_model.pkl
├── scaler.pkl
└── feature_columns.pkl
```

| File | Purpose |
|---|---|
| `lasso_model.pkl` | Trained Lasso model |
| `scaler.pkl` | Fitted feature scaler |
| `feature_columns.pkl` | Feature names and training feature order |

The stored feature columns ensure that application inputs maintain the same feature structure used during model training.

---

## Project Structure

```text
used-car-price-prediction/
│
├── app/
│   └── app.py
│
├── data/
│   └── cardekho_dataset.csv
│
├── model/
│   ├── lasso_model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── notebooks/
│   └── analysis.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

| Path | Purpose |
|---|---|
| `app/` | Streamlit application |
| `data/` | Dataset |
| `model/` | Trained model and preprocessing artifacts |
| `notebooks/` | Analysis, preprocessing, training and evaluation |
| `.gitignore` | Git exclusions |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

---

## Technology Stack

<p align="center">

<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white">
<img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white">
<img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white">
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white">
<img src="https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white">
<img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white">

</p>

---

## Installation

### Clone

```bash
git clone https://github.com/Soham-Shirode/used-car-price-prediction.git
cd used-car-price-prediction
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate on Windows

```powershell
venv\Scripts\Activate.ps1
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app/app.py
```

---

## Reproducing the Model

The complete machine learning workflow is available in:

```text
notebooks/analysis.ipynb
```

The notebook covers:

```text
Data Loading
    ↓
Data Cleaning
    ↓
Exploratory Data Analysis
    ↓
Feature Engineering
    ↓
Categorical Encoding
    ↓
Train/Test Split
    ↓
Feature Scaling
    ↓
Model Training
    ↓
Model Comparison
    ↓
Hyperparameter Tuning
    ↓
Model Evaluation
    ↓
Model Serialization
```

---

## Future Improvements

- Evaluate tree-based regression models
- Experiment with ensemble methods
- Improve outlier handling
- Add additional vehicle and market features
- Add model explainability
- Provide prediction intervals
- Deploy the application publicly


## Author

**Soham Shirode**