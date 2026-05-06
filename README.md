# Hotel Booking Cancellation Risk Prediction

This project predicts whether a hotel reservation will be canceled using machine learning, with the goal of helping hotels reduce revenue loss and improve booking decisions.

## Project Goal
The main question we address is:

**Which reservations are most likely to cancel, and what factors drive that risk?**

## Dataset
We use the **Hotel Booking Demand** dataset, which combines:
- `H1.csv` for Resort Hotel bookings
- `H2.csv` for City Hotel bookings

After preprocessing:
- Total bookings: **119,390**
- Cleaned dataset: **48 columns**
- Model-ready dataset: **45 columns**

## What This Repo Contains
- `preprocess.py` — data cleaning, merging, feature engineering, and model-ready dataset creation
- `plotlyImpl.py` — Plotly-based EDA visualizations / dashboard work
- `streamlit_ml_dashboard.py` — Streamlit dashboard for ML results
- `DV_Final_Project_Up.pbix` — Power BI dashboard file

## Preprocessing Summary
Main preprocessing work included:
- merging H1 and H2 into one dataset
- cleaning nulls, whitespace, and data types
- removing leakage-risk columns for ML:
  - `ReservationStatus`
  - `ReservationStatusDate`
- creating engineered features such as:
  - `total_nights`
  - `total_guests`
  - `season`
  - `arrival_month_num`
  - `has_special_requests`
  - `has_booking_changes`
  - `extreme_adr_flag`
  - `extreme_lead_time_flag`

## Modeling Approach
This project is framed as a **binary classification problem**:
- `0` = Not Canceled
- `1` = Canceled

Models tested:
- Logistic Regression
- Decision Tree
- XGBoost

Feature sets compared:
- **Core**
- **Core + Behavioral**
- **Full Engineered**

We evaluated model performance using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

## Final Model
**XGBoost** performed best overall and was selected as the final model.

Final XGBoost results:
- Accuracy: **0.8278**
- Precision: **0.8543**
- Recall: **0.6451**
- F1-score: **0.7351**
- ROC-AUC: **0.8961**
- PR-AUC: **0.8673**

## Dashboards
### Power BI
The Power BI dashboard focuses on:
- feature set progression
- predicted vs actual cancellation risk by market segment

### Streamlit
The Streamlit dashboard includes ML evaluation visuals such as:
- confusion matrix
- ROC curve
- precision-recall curve
- threshold tradeoff

## Key Findings
- Behavioral booking features gave the biggest model improvement
- Longer lead times are strongly associated with higher cancellation risk
- Market segment and deposit type are important predictors
- XGBoost performed better than Logistic Regression and Decision Tree

## How to Run
### Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost matplotlib plotly streamlit
