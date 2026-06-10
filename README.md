# # NomadiQ Smart Flight Fare Predictor

A machine learning-powered flight fare prediction system that estimates future airfare based on route, airline, departure date, cabin class, and current ticket price.

The application combines an XGBoost prediction model with a rule-based fallback engine to provide fare estimates for both supported and unsupported routes.

---

## Features

* Predict future flight fares using Machine Learning
* XGBoost-based prediction engine
* Rule-based fallback for unsupported routes
* Airport code validation with suggestions
* Interactive Streamlit interface
* Support for major Indian domestic routes
* Real-time fare estimation
* ML-ready feature engineering pipeline

---

## Project Structure

```text
nomadiq-flight-predictor/
│
├── app.py
├── xgb_flight_model.json
├── model_artifacts.pkl
│
├── data/
│
├── requirements.txt
│
└── README.md
```

---

## How It Works

### Supported Routes

For routes included in the training dataset:

```text
User Input
    ↓
Feature Engineering
    ↓
XGBoost Model
    ↓
Predicted Future Fare
```

### Unsupported Routes

For routes not available in the training dataset:

```text
User Input
    ↓
Rule-Based Pricing Logic
    ↓
Predicted Future Fare
```

This ensures the application can provide predictions even when a route was not part of the original training data.

---

## Model Features

The model uses features such as:

```text
Origin Airport
Destination Airport
Airline
Cabin Class
Current Fare
Days Before Departure
Departure Month
Departure Weekday
Departure Time
Arrival Time
Baggage Information
```

---

## Supported Airlines

* IndiGo
* Air India
* Vistara
* SpiceJet
* Akasa Air

---

## Validation

The application performs:

* Airport code validation
* Route validation
* Departure date validation
* Minimum booking window checks
* Airport suggestion correction

Example:

```text
Input: DLE

Suggestion: DEL
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/nomadiq-flight-predictor.git

cd nomadiq-flight-predictor
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

The application will launch locally in your browser.

---

## Example Prediction Flow

Input:

```text
Origin: DEL
Destination: BOM
Airline: IndiGo
Current Fare: ₹5,500
Days Before Departure: 20
Cabin Class: Economy
```

Output:

```text
Current Fare: ₹5,500

Predicted Future Fare: ₹5,120
```

---

## Tech Stack

### Frontend

* Streamlit

### Machine Learning

* XGBoost
* Pandas
* NumPy

### Model Artifacts

* Pickle
* XGBoost Booster

---

## Use Cases

* Flight fare forecasting
* Travel planning
* Dynamic pricing analysis
* Airline trend analysis
* Machine learning experimentation

---

## Future Improvements

* Multi-city route support
* Confidence score generation
* Historical fare visualization
* Route demand forecasting
* API deployment
* Real-time flight data integration

---

## Notes

* Predictions are generated using a trained XGBoost model.
* Unsupported routes automatically use a fallback pricing engine.
* Airport codes follow IATA standards.
* The application currently focuses on domestic Indian routes.
