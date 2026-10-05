# Used Car Price Prediction

Predicting the resale price of used cars from their specifications, comparing **Linear Regression**, **K-Nearest Neighbours** and **XGBoost**. The best model, a tuned XGBoost regressor, explains **~93% of the variance** in (log) price on held-out data.

Undergraduate dissertation project, University of Leeds (2024).

---

## Results

R² on a 30% held-out test set, with price log-transformed.

| Model | Full dataset (20 features) | Reduced dataset (4 features) |
|---|---|---|
| Linear Regression | 0.872 | 0.863 |
| KNN (k = 10, baseline) | 0.907 | 0.874 |
| KNN (tuned with GridSearchCV) | 0.921 (k = 3) | 0.902 (k = 2) |
| XGBoost (baseline, linear booster) | 0.699 | 0.357 |
| **XGBoost (tuned with GridSearchCV)** | **0.933** | **0.930** |

**Key findings**
- **Tuned XGBoost performed best on both datasets**, and kept ~93% R² even with only 4 features, showing that a handful of variables carry most of the predictive signal.
- **Hyperparameter tuning mattered most for XGBoost**: R² rose from 0.70 to 0.93 on the full dataset, and from 0.36 to 0.93 on the reduced one.
- **Linear Regression was a strong, interpretable baseline** at ~0.87, which suggests the log-price relationship is largely linear.

---

## Data

| Dataset | Rows | Features | Description |
|---|---|---|---|
| `Cars.csv` | 2,059 | 20 | Indian used-car listings: make, model, year, kilometres, fuel type, transmission, engine, power, torque, dimensions, owner and seller type, etc. |
| `Validation.csv` | 976 | 4 | Reduced set: year, seating capacity, transmission, max power |

Cars range from 1988 to 2022, across 33 manufacturers and 77 locations, with a median price of ₹8.25 lakh.

---

## Approach

1. **Cleaning:** removed missing values and treated outliers.
2. **Target transformation:** log-transformed price to reduce strong right skew (prices range from ₹49k to ₹3.5 crore).
3. **Feature engineering:** extracted numeric values from text fields (e.g. `"87 bhp @ 6000 rpm"` → `87`) and label-encoded categorical variables.
4. **Exploratory analysis:** price distributions before and after transformation, and correlation heatmaps between predictors and price.
5. **Modelling:** 70/30 train–test split, feature scaling with `StandardScaler`, then three model families.
6. **Tuning:** `GridSearchCV` for the number of neighbours (KNN) and the boosting parameters (XGBoost).
7. **Evaluation:** R², MSE, RMSE and MAE, plus visual comparison of predicted vs actual prices.

**Stack:** Python · pandas · NumPy · scikit-learn · XGBoost · Matplotlib · Seaborn

---

## Repository structure

| File | Description |
|---|---|
| `Linear_Regression_Model(O).ipynb` | Full pipeline (cleaning, EDA, encoding) and Linear Regression on the full dataset |
| `KNN_Model(O).ipynb` | KNN on the full dataset, with tuning |
| `XGB_Model(O).ipynb` | XGBoost on the full dataset, with tuning |
| `*_Model(V).ipynb` | The same three models on the reduced dataset |
| `Cars.csv` | Full dataset |
| `Validation.csv` | Reduced dataset |

`(O)` = original (full) dataset, `(V)` = validation (reduced) dataset.

**To run:** open any notebook in Google Colab, upload the matching CSV (`Cars.csv` for `(O)`, `Validation.csv` for `(V)`) to the Files panel, and run all cells.

---

## Limitations and next steps

- Results come from a single train–test split; **k-fold cross-validation** would give more robust estimates.
- R² is measured on log-price. Reporting errors in **actual currency** would make results easier to interpret for non-technical users.
- Label encoding imposes an artificial order on categories such as make and location; **one-hot or target encoding** may improve the linear and KNN models.
- **Feature importance analysis** of the XGBoost model would show which car attributes drive price the most.
