# Household Energy Consumption Prediction

## 📌 Project Overview

This project predicts **household energy consumption** using Machine Learning and **Polynomial Regression**.

The model uses three input features:

* Household Size
* Average Temperature
* Peak Hours Usage

The target variable is:

* Energy Consumption

The project also evaluates the model using MAE, MSE, RMSE, and R-squared metrics and visualizes the actual and predicted energy consumption.

---

## 🎯 Objectives

The main objectives of this project are:

* Load and explore household energy consumption data.
* Understand the structure and statistical properties of the dataset.
* Check for missing values.
* Select important input and target variables.
* Split the dataset into training and testing sets.
* Apply Polynomial Features with degree 2.
* Train a Linear Regression model using the polynomial features.
* Predict household energy consumption.
* Compare actual and predicted values.
* Evaluate the model using regression metrics.
* Visualize actual vs predicted energy consumption.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Jupyter Notebook**

---

## 📂 Dataset

The project uses the following dataset:

```text
household_energy_consumption - household_energy_consumption.csv
```

The dataset contains household energy-related information.

### Features Used

| Feature                | Description                            |
| ---------------------- | -------------------------------------- |
| `Household_Size`       | Number/size of people in the household |
| `Avg_Temperature_C`    | Average temperature in Celsius         |
| `Peak_Hours_Usage_kWh` | Energy usage during peak hours         |

### Target Variable

| Target                   | Description                         |
| ------------------------ | ----------------------------------- |
| `Energy_Consumption_kWh` | Household energy consumption in kWh |

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Exploration
   ↓
Missing Value Check
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Polynomial Feature Transformation
   ↓
Linear Regression
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Visualization
```

---

## 1. Import Libraries

The project uses Pandas, NumPy, Matplotlib, and Seaborn:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

Scikit-learn is used for splitting the data, creating polynomial features, building the regression model, and evaluating the model.

---

## 2. Load the Dataset

The dataset is loaded using Pandas:

```python
df = pd.read_csv(
    "/content/household_energy_consumption - household_energy_consumption.csv"
)
```

The first few records are displayed using:

```python
df.head()
```

---

## 3. Explore the Dataset

The notebook examines the dataset using:

```python
df.info()
```

```python
df.describe()
```

```python
df.shape
```

These functions provide information about:

* Dataset structure
* Data types
* Statistical summary
* Number of rows and columns

---

## 4. Check Missing Values

Missing values are checked using:

```python
df.isnull().sum()
```

The notebook also contains:

```python
df.dropna()
```

for removing rows containing missing values.

---

## 5. Select Features and Target

The following three variables are selected as input features:

```python
X = df[
    [
        "Household_Size",
        "Avg_Temperature_C",
        "Peak_Hours_Usage_kWh"
    ]
]
```

The target variable is:

```python
y = df["Energy_Consumption_kWh"]
```

Therefore:

```text
X → Input Features
y → Energy Consumption
```

---

## 6. Train-Test Split

The dataset is divided into training and testing data:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Here:

* **80%** of the data is used for training.
* **20%** of the data is used for testing.
* `random_state=42` makes the split reproducible.

---

## 7. Polynomial Feature Transformation

Polynomial features of degree 2 are created:

```python
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
```

Polynomial transformation allows the model to capture relationships between the input variables beyond a simple straight-line relationship.

The training data is used with `fit_transform()`, while the test data uses only `transform()`.

---

## 8. Train the Regression Model

A Linear Regression model is created:

```python
model = LinearRegression()
```

The model is trained using the polynomial features:

```python
model.fit(X_train_poly, y_train)
```

The target variable remains `y_train`; it is **not** transformed into `y_train_poly`.

---

## 9. Make Predictions

Predictions are generated using the transformed test data:

```python
y_pred = model.predict(X_test_poly)
```

The predicted values represent the model's estimated household energy consumption.

---

## 10. Actual vs Predicted Values

The actual and predicted values are combined into a DataFrame:

```python
result = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})
```

The first 10 results are displayed:

```python
print(result.head(10))
```

This allows the actual energy consumption to be compared with the model's predictions.

---

## 11. Model Evaluation

The notebook evaluates the model using four regression metrics:

### Mean Absolute Error (MAE)

```python
mae = mean_absolute_error(y_test, y_pred)
```

MAE measures the average absolute difference between actual and predicted values.

### Mean Squared Error (MSE)

```python
mse = mean_squared_error(y_test, y_pred)
```

MSE calculates the average squared difference between actual and predicted values.

### Root Mean Squared Error (RMSE)

```python
rmse = np.sqrt(mse)
```

RMSE is the square root of MSE and represents prediction error in the same unit as the target variable.

### R-squared

```python
r2 = r2_score(y_test, y_pred)
```

R-squared indicates how well the model explains the variation in the target variable.

The results are displayed using:

```python
print("\nModel Evaluation")
print("-----------------")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R-squared:", r2)
```

---

## 📊 Visualization

The project creates an **Actual vs Predicted Energy Consumption** scatter plot:

```python
plt.figure(figsize=(10, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Energy Consumption (KWh)")
plt.ylabel("Predicted Energy Consumption (KWh)")

plt.title("Actual vs Predicted Energy Consumption")

plt.show()
```

This visualization helps compare the model's predictions with the actual energy consumption values.

---

## 📁 Project Structure

```text
Household-Energy-Consumption/
│
├── household_energy_consumption.csv
├── Energy_Consumption.ipynb
└── README.md
```

---

## 🚀 How to Run the Project

### Step 1: Install Python

Install Python on your system.

### Step 2: Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Step 3: Open Jupyter Notebook

```bash
jupyter notebook
```

### Step 4: Open the Notebook

Open the project `.ipynb` file.

### Step 5: Add the Dataset

Place the household energy consumption CSV file in the appropriate directory and update the file path if necessary.

### Step 6: Run the Cells

Run the notebook cells from top to bottom to:

1. Load the dataset
2. Explore the data
3. Check missing values
4. Select features
5. Split the dataset
6. Create polynomial features
7. Train the model
8. Generate predictions
9. Evaluate the model
10. Display the visualization

---

## 📌 Key Machine Learning Concepts

This project demonstrates:

* Data preprocessing
* Exploratory Data Analysis
* Feature selection
* Train-test splitting
* Polynomial feature engineering
* Linear Regression
* Prediction
* Regression evaluation
* Data visualization

---

## 👩‍💻 Author

**Madhumidha**

BCA Student
