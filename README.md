# Flight Operations Intelligence: Predicting Significant Flight Delays

An end-to-end machine learning project for predicting significant U.S. airline arrival delays using historical flight operations and NOAA weather data.

## Project Overview

Flight delays create operational, financial, and customer-service challenges for airlines and airports. This project develops a pre-departure machine learning pipeline to estimate whether a scheduled flight will experience a significant arrival delay.

A significant delay is defined as:

> **Arrival delay ≥ 15 minutes**

The project combines approximately **6.88 million U.S. domestic flight records from 2025** with hourly NOAA weather observations and historical operational performance features.

Rather than randomly splitting flights, the modeling pipeline uses chronological train, validation, and test periods to simulate how a model would perform on future flights.

## Data Sources

### Bureau of Transportation Statistics (BTS)

2025 Reporting Carrier On-Time Performance data provides flight-level operational information including:

- Airline
- Origin and destination airports
- Scheduled departure and arrival times
- Flight distance
- Scheduled elapsed time
- Arrival delay
- Cancellation and diversion status

After filtering cancelled flights, diverted flights, and records without an arrival-delay outcome, the modeling dataset contains approximately **6.88 million flights**.

### NOAA Weather

Hourly NOAA weather observations were mapped to airports and aligned with scheduled flight times.

Weather features include:

- Temperature
- Wind speed
- Visibility
- Atmospheric pressure
- Precipitation
- Ceiling height

Missing-weather indicators are retained so the models can distinguish unavailable observations from observed weather conditions.

## Feature Engineering

All predictive features are designed to be available before departure.

The feature pipeline includes:

### Schedule and Flight Features

- Airline
- Origin airport
- Destination airport
- Month
- Day of week
- Scheduled departure hour
- Scheduled arrival hour
- Weekend indicator
- Scheduled elapsed time
- Distance

### Historical Operational Features

Historical delay behavior is calculated using only information available before each flight, helping prevent target leakage.

Features include:

- Historical carrier delay rates
- Historical origin delay rates
- Historical destination delay rates
- Historical route delay rates
- Prior flight counts

### Rolling 7-Day Features

Recent operational conditions are represented through rolling seven-day statistics, including:

- Carrier delay rate
- Origin delay rate
- Destination delay rate
- Route delay rate
- Prior seven-day flight counts

### Weather Features

NOAA weather conditions are joined to scheduled flights using airport and time information.

## Time-Based Model Validation

The data is split chronologically rather than randomly:

| Dataset | Period | Flights | Delay Rate |
|---|---|---:|---:|
| Training | Jan 2 – Sep 30, 2025 | 5,133,923 | 22.25% |
| Validation | Oct 1 – Nov 30, 2025 | 1,156,866 | 20.44% |
| Test | Dec 1 – Dec 31, 2025 | 571,924 | 26.77% |

This design provides a more realistic evaluation of performance on future operational conditions.

## Models

Three classification approaches were evaluated:

1. **Logistic Regression** — interpretable baseline
2. **LightGBM** — gradient-boosted decision trees
3. **CatBoost** — gradient boosting with native handling of categorical variables

Because significant delays are the minority class, model evaluation emphasizes **ROC-AUC, PR-AUC, precision, recall, and F1** rather than accuracy alone.

## Final Model

LightGBM was selected as the final model after comparison with Logistic Regression and CatBoost.

The classification threshold was selected using the validation dataset and then locked before evaluation on the December test set.

**Locked decision threshold: 0.185**

### Out-of-Time Test Performance

| Metric | Result |
|---|---:|
| ROC-AUC | 0.687 |
| PR-AUC | 0.460 |
| Precision | 0.354 |
| Recall | 0.754 |
| F1 Score | 0.482 |

The final model identifies approximately **75% of significantly delayed flights** in the out-of-time December test period, at the cost of a substantial number of false-positive alerts.

This tradeoff makes the model more appropriate as an **operational risk-screening system** than as a definitive prediction of whether an individual flight will be delayed.

## Key Predictive Features

LightGBM feature importance indicates that major predictive signals include:

- Scheduled departure hour
- Scheduled arrival hour
- Recent route delay performance
- Origin and destination airport
- Month and day of week
- Visibility
- Airline
- Temperature
- Atmospheric pressure
- Precipitation
- Ceiling height

Recent route performance was particularly informative, demonstrating the value of incorporating short-term operational history alongside schedule and weather information.

## Repository Structure

```text
flight-operations-intelligence/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── reports/
│   ├── figures/
│   └── results/
├── README.md
└── requirements.txt
```

## Outputs

The project produces:

- Trained LightGBM model
- Logistic Regression, LightGBM, and CatBoost model comparison
- Out-of-time test evaluation
- Confusion matrix
- Feature importance analysis
- Processed flight-weather dataset
- Reproducible modeling workflow

## Technology Stack

**Python · SQL · DuckDB · pandas · scikit-learn · LightGBM · CatBoost · NOAA Weather Data · BTS On-Time Performance Data**

## Limitations

The model should be interpreted as a flight-delay risk model rather than a production airline decision system.

Important limitations include:

- Weather observations do not perfectly represent conditions at every airport and scheduled flight time.
- Precipitation has substantially lower coverage than the primary weather variables.
- Operational information such as aircraft rotations, crew availability, maintenance events, and real-time air-traffic restrictions is not included.
- Delay patterns can change across seasons and years.
- Model performance should therefore be monitored for temporal drift before production deployment.