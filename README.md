# Flight-Price-Prediction
Flight Price Prediction using Machine Learning | EDA, Feature Engineering, and Regression Model Comparison with Tuned XGBoost to estimate flight ticket fares.

# Flight Price Prediction

A machine learning regression project that analyzes flight-fare data and
predicts ticket prices using journey details such as airline, source,
destination, stops, date, departure and arrival times, and flight
duration.

## Project Overview

Flight fares can vary considerably based on airline, route, journey
date, number of stops, and duration. This project uses exploratory data
analysis (EDA), feature engineering, and regression models to understand
fare patterns and estimate ticket prices.

The project is implemented in a Jupyter Notebook:
`Flight Price Prediction.ipynb`.

## Objectives

-   Explore and analyze flight-fare data using visualizations and
    descriptive statistics.
-   Clean the data and engineer features suitable for machine learning.
-   Compare multiple regression algorithms using cross-validation and a
    held-out test set.
-   Tune Random Forest and XGBoost models and select a model based on
    cross-validation performance.
-   Save the final preprocessing-and-model pipeline with Joblib.

## Dataset

The project brief provides the dataset here:

[Flight Fare Dataset
(ZIP)](https://d3ilbtxij3aepc.cloudfront.net/projects/CDS-Capstone-Projects/flight-fare.zip)

The notebook reports 10,683 original rows and 11 columns. After removing
two rows with missing values and 220 duplicate rows (after normalizing
labels), 10,462 rows remain.

Important original columns include:

-   `Airline`
-   `Date_of_Journey`
-   `Source`
-   `Destination`
-   `Route`
-   `Arrival_Time`
-   `Duration`
-   `Total_Stops`
-   `Additional_Info`
-   `Price` --- prediction target

The notebook engineers numeric date/time and duration features and drops
`Route` because its information overlaps with source, destination, and
number of stops. The dataset covers March--June 2019, so the model
should not be assumed to generalize to other periods or current market
fares.

## Workflow

1.  **Data inspection:** examine shape, data types, missing values,
    duplicates, and fare distribution.
2.  **Data cleaning:** remove rows with missing values, remove duplicate
    records, and standardize inconsistent category labels.
3.  **Feature engineering:** derive journey day/month/weekday, departure
    and arrival hour/minute, duration in minutes, and numeric stop
    counts.
4.  **EDA:** examine fares by airline, stops, route, journey month, and
    other features.
5.  **Preprocessing:** one-hot encode categorical features within a
    scikit-learn pipeline; optionally scale numeric features for
    scale-sensitive models.
6.  **Model comparison:** evaluate Linear Regression, Ridge, Lasso, KNN,
    Decision Tree, Random Forest, Gradient Boosting, and XGBoost.
7.  **Model tuning:** use `RandomizedSearchCV` with 5-fold
    cross-validation for Random Forest and XGBoost.
8.  **Evaluation:** compare cross-validation R² with test-set R², MAE,
    RMSE, MAPE, and the train/test R² gap.
9.  **Final pipeline:** refit the selected tuned XGBoost pipeline on the
    full cleaned dataset and save it as `flight_price_model.pkl`.

## Model Results

The notebook uses an 80/20 train/test split (8,369 training rows and
2,093 test rows) and 5-fold cross-validation on the training set.

  Model                          CV R²      Test R²    MAE (₹)     RMSE (₹)        MAPE
  ----------------------- ------------ ------------ ---------- ------------ -----------
  **XGBoost (Tuned)**       **0.9068**       0.8925       ₹681       ₹1,497       7.92%
  XGBoost                       0.8910   **0.8945**       ₹765   **₹1,483**       8.99%
  Random Forest (Tuned)         0.8864       0.8908       ₹726       ₹1,509       8.60%
  Random Forest                 0.8812       0.8861   **₹674**       ₹1,541   **7.68%**

**Model selected:** Tuned XGBoost, selected using the highest
cross-validation R² (0.9068). Its held-out test R² is 0.8925, with an
MAE of approximately ₹681. The default XGBoost and Random Forest have
slightly better results on some individual test metrics, so the choice
prioritizes cross-validation performance rather than claiming it wins
every metric.

*Results above are transcribed from the project notebook and may vary if
the dataset, dependencies, or random settings change.*

## Tech Stack

-   Python
-   Jupyter Notebook
-   pandas and NumPy
-   Matplotlib and Seaborn
-   scikit-learn
-   XGBoost
-   Joblib

## Getting Started

### 1. Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

### 2. Install dependencies

Use a Python environment and install the packages used by the notebook:

``` bash
pip install jupyter pandas numpy matplotlib seaborn scikit-learn xgboost openpyxl joblib
```

### 3. Add the dataset

Download the dataset from the link above and place the extracted Excel
file in your project folder.

**Important:** the notebook currently reads the dataset from a
machine-specific Windows path:

``` python
df = pd.read_excel(r"C:\Users\DELL\Flight_Fare.xlsx")
```

Change this to the actual location of your downloaded file, for example:

``` python
df = pd.read_excel("Flight_Fare.xlsx")
```

The filename and extension must match the file you downloaded.

### 4. Run the notebook

``` bash
jupyter notebook
```

Open `Flight Price Prediction.ipynb` and run the cells in order. The
notebook performs EDA, trains and compares models, tunes the ensemble
models, evaluates the selected model, and saves `flight_price_model.pkl`
when the relevant cells complete successfully.

## Predicting a Fare

The notebook defines a helper function named `predict_price` with this
interface:

``` python
predict_price(
    airline,
    source,
    destination,
    total_stops,
    journey_date,   # "dd/mm/yyyy"
    dep_time,       # "HH:MM"
    arr_time,       # "HH:MM"
    duration_mins,
    additional_info="No info"
)
```

Example from the notebook:

``` python
predict_price(
    "IndiGo",
    "Bangalore",
    "Delhi",
    0,
    "24/03/2019",
    "22:20",
    "01:10",
    170
)
```

This function is defined inside the notebook and uses the final model
trained there. It is not a standalone web application or API.

## Challenges Addressed

-   **Missing values:** removed two affected rows because the number was
    negligible.
-   **Duplicate rows:** removed 220 duplicates to reduce the risk of
    overly optimistic evaluation.
-   **Inconsistent labels:** standardized spelling/capitalization
    variants in categorical fields.
-   **Text-based fields:** converted journey dates, times, duration, and
    stop counts into model-ready features.
-   **Categorical variables:** applied one-hot encoding within a
    pipeline to avoid leakage across cross-validation folds and to
    handle unseen categories.
-   **Skewed fare distribution and nonlinear relationships:** compared
    several model families, including tree-based ensembles.
-   **Model selection:** used cross-validation to select the final model
    while retaining the test set as an independent check.

## Limitations

-   The dataset only covers March to June 2019; it does not represent
    current fare levels or all seasons.
-   Very expensive and unusual fares are less typical and may be harder
    to predict.
-   The model is a learning project, not a live airline pricing service.
    Predictions are estimates, not guaranteed booking prices.
-   The notebook's dataset path must be updated for your local
    environment.
-   The saved `.pkl` model is generated by running the notebook; it may
    not be present in the repository unless you add it.

## Repository Contents

``` text
.
├── Flight Price Prediction.ipynb
├── PRCP-1025-FlightPricePrediction.docx
└── README.md
```

Add the dataset only if its license and redistribution terms permit it.
Avoid committing large or restricted datasets unnecessarily.

## Future Improvements

-   Add a command-line interface, web app, or REST API for interactive
    predictions.
-   Validate on data from additional months and more recent years.
-   Track model performance on new data and retrain when needed.
-   Add automated tests and a reproducible dependency file such as
    `requirements.txt`.

## Author

Add your name, GitHub profile, and contact links here.

## License

Add a license if you intend to publish or distribute this project. Until
a license is added, do not assume others have permission to reuse the
repository's contents.
