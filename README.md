# 🏠 House Price Prediction using XGBoost

A machine learning project that predicts house prices using the
**California Housing dataset** and an **XGBoost Regression** model.

## 📌 Project Overview

This project demonstrates an end-to-end regression workflow for house
price prediction. The California Housing dataset is loaded using
Scikit-learn, converted into a Pandas DataFrame, explored through
descriptive statistics and correlation analysis, and then used to train
an XGBoost regression model.

The notebook evaluates the model on both training and testing data
using:

-   **R² Score**
-   **Mean Absolute Error (MAE)**

It also visualizes the relationship between actual and predicted house
prices.

------------------------------------------------------------------------

## 🎯 Objectives

-   Load and understand a real-world house price dataset.
-   Perform basic data exploration and statistical analysis.
-   Check for missing/null values.
-   Study correlations between housing features.
-   Visualize feature correlations using a heatmap.
-   Separate input features and the target variable.
-   Split the data into training and testing sets.
-   Train an XGBoost regression model.
-   Evaluate model performance using R² Score and MAE.
-   Visualize actual versus predicted house prices.

------------------------------------------------------------------------

## 📊 Dataset

The project uses the **California Housing dataset** available through:

``` python
from sklearn.datasets import fetch_california_housing
```

The dataset is loaded using:

``` python
housing = fetch_california_housing()
```

The feature data is converted into a Pandas DataFrame, and the target
values are added as the `Price` column.

------------------------------------------------------------------------

## 🔄 Project Workflow

``` text
California Housing Dataset
          ↓
     Load Dataset
          ↓
   Create DataFrame
          ↓
 Exploratory Data Analysis
          ↓
 Check Null Values
          ↓
 Descriptive Statistics
          ↓
 Correlation Analysis
          ↓
    Heatmap
          ↓
 Separate Features (X)
 and Target (Y)
          ↓
 Train-Test Split
          ↓
   XGBoost Regressor
          ↓
      Training
          ↓
 Predictions
          ↓
 R² Score + MAE
          ↓
 Actual vs Predicted Visualization
```

------------------------------------------------------------------------

## 🛠️ Technologies Used

  Python           
  NumPy              
  Pandas             
  Matplotlib       
  Seaborn            
  Scikit-learn       
  XGBoost            
  Jupyter Notebook   

------------------------------------------------------------------------

## 📁 Project Structure

``` text
House-Price-Prediction/
│
├── House_Price_Prediction.ipynb
├── README.md
```

------------------------------------------------------------------------

## ⚙️ Installation

### 1. Clone the Repository

``` bash
git clone https://github.com/YandapalliTejaswini/House-Price-Prediction.git
cd House-Price-Prediction
```



### 2. Install Dependencies

``` bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

Or use the included requirements file:

``` bash
pip install -r requirements.txt
```

### 3. Start Jupyter Notebook

``` bash
jupyter notebook
```

Open:

``` text
House_Price_Prediction.ipynb
```

Run the cells from top to bottom.

------------------------------------------------------------------------

## 🔍 Exploratory Data Analysis

The notebook performs the following analysis.

### Dataset Creation

``` python
house_price_dataframe = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)
```

The target is then added:

``` python
house_price_dataframe['Price'] = housing.target
```

### First Five Records

The project uses:

``` python
house_price_dataframe.head()
```

to inspect the first records.

### Dataset Shape

``` python
house_price_dataframe.shape
```

is used to identify the number of rows and columns.

### Missing Value Check

``` python
house_price_dataframe.isnull().sum()
```

is used to check for null values.

### Statistical Summary

``` python
house_price_dataframe.describe()
```

is used to examine statistical properties of the dataset.

------------------------------------------------------------------------

## 📈 Correlation Analysis

The project calculates feature correlations using:

``` python
correlation = house_price_dataframe.corr()
```

A heatmap is then constructed using Seaborn:

``` python
plt.figure(figsize=(10,10))

sns.heatmap(
    correlation,
    cbar=True,
    square=True,
    fmt='.1f',
    annot=True,
    annot_kws={'size':8},
    cmap='Blues'
)
```

This helps understand relationships between the housing variables and
the target price.

------------------------------------------------------------------------

## ✂️ Train-Test Split

The target variable is separated from the input features:

``` python
X = house_price_dataframe.drop(['Price'], axis=1)
Y = house_price_dataframe['Price']
```

The dataset is split into training and testing data using:

``` python
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=2
)
```

### Split Configuration

-   **Testing data:** 20%
-   **Training data:** 80%
-   **Random state:** 2

------------------------------------------------------------------------

## 🤖 Model Training

The project uses an **XGBoost Regressor**:

``` python
model = XGBRegressor()
```

The model is trained using:

``` python
model.fit(X_train, Y_train)
```

XGBoost is a gradient-boosted decision-tree algorithm that is commonly
used for structured/tabular regression and classification tasks.

------------------------------------------------------------------------

## 🧪 Model Evaluation

The project evaluates predictions using two metrics.

### 1. R² Score

``` python
metrics.r2_score(Y_train, training_data_prediction)
```

and:

``` python
metrics.r2_score(Y_test, test_data_prediction)
```

R² indicates how well the model explains the variation in the target
variable.

A value closer to **1** generally indicates stronger predictive fit.

### 2. Mean Absolute Error

``` python
metrics.mean_absolute_error(
    Y_train,
    training_data_prediction
)
```

and:

``` python
metrics.mean_absolute_error(
    Y_test,
    test_data_prediction
)
```

MAE measures the average absolute difference between actual and
predicted values.

Lower MAE indicates better prediction accuracy.



------------------------------------------------------------------------

## 📊 Actual vs Predicted Visualization


``` python
plt.scatter(Y_train, training_data_prediction)

plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')

plt.title('Actual Prices vs Predicted Prices')

plt.show()
```

This visualization helps assess how closely the predicted values follow
the actual values.

------------------------------------------------------------------------

## 💡 Key Learning Outcomes

Through this project, the following machine learning concepts are
demonstrated:

-   Regression
-   Exploratory Data Analysis
-   Feature-target separation
-   Correlation analysis
-   Data visualization
-   Train-test splitting
-   XGBoost regression
-   Model prediction
-   R² evaluation
-   Mean Absolute Error
-   Actual vs predicted visualization

------------------------------------------------------------------------

## 👤 Author

**Yandapalli Tejaswini**

GitHub: `https://github.com/YandapalliTejaswini`

------------------------------------------------------------------------


