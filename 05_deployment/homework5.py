# Machine Learning Zoomcamp 2026 - Homework 5
# Note: Local execution was blocked by Windows Application Control policies.
# This script contains the logic used to evaluate the pipeline.bin model.

import pickle

# --- Question 3: Load the Model ---
with open('pipeline.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

customer_q3 = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41
}

X = dv.transform([customer_q3])
y_pred_q3 = model.predict_proba(X)[0, 1]
print(f"Q3 Conversion probability: {y_pred_q3:.3f}")


# --- Question 4 & 6: API Request Logic ---
# Note: Executed via Invoke-RestMethod in PowerShell to bypass local Python blocks.
# The equivalent Python requests logic:
"""
import requests

url = 'http://localhost:9696/predict'
customer_q4 = {
    "lead_source": "organic_search",
    "industry": "technology",
    "employment_status": "employed",
    "location": "europe",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0,
    "interaction_count": 7,
    "lead_score": 0.74
}

response = requests.post(url, json=customer_q4).json()
print(f"Q4/Q6 API Response: {response['churn_probability']:.3f}")
"""

#answer 6- 0.918
