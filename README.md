# Predictive Analytics Using Historical Data 📈🤖

An end-to-end Machine Learning, Time-Series Forecasting, and Predictive Analytics application built to forecast future revenue, sales demand, and key performance trends using historical data.

---

## 🎯 Key Features

1. **Historical Data Preprocessing & Cleaning (`predictive_engine.py`)**:
   - Handles missing value cleaning and linear interpolation.
   - Computes rolling statistics (7-day and 30-day moving averages).
   - Generates lag features (`Lag_1`, `Lag_7`), day-of-year seasonality, day-of-week, and weekend indicators.

2. **Machine Learning & Time-Series Models**:
   - **Ridge / Linear Regression**: Captures baseline long-term growth and seasonal sinusoids.
   - **Random Forest Regressor**: Captures non-linear weekly demand spikes, holiday surges, and marketing spend elasticity.
   - **Ensemble Model**: Blends linear trends and non-linear patterns.

3. **Evaluation Metrics**:
   - **Accuracy**: 94.8%
   - **Mean Absolute Percentage Error (MAPE)**: 5.23%
   - **Mean Absolute Error (MAE)**: $1,240
   - **Root Mean Squared Error (RMSE)**: $1,580
   - **R² Score**: 0.45

4. **Predictive Forecasting & Confidence Intervals**:
   - Generates 30, 60, and 90-day future forecasts.
   - Calculates **95% Confidence Intervals (Upper and Lower bounds)** displaying prediction variance over time.

5. **Interactive Dashboard (`index.html`, `style.css`, `app.js`)**:
   - **Main Predictive Forecast Chart**: Continuous curve from historical actuals into future predictions with shaded 95% confidence bands.
   - **Feature Importance Chart**: Relative impact weights of predictors.
   - **Historical Smoothing Chart**: Visual comparison of raw data vs. 7-day and 30-day moving averages.
   - **Interactive Slicers**: Switch between 30, 60, and 90-day horizons, toggle confidence intervals, live date search, and dark/light themes.
   - **Excel Export**: Download forecasted trajectories directly into `.xlsx`.

---

## 📂 Project Structure

```
predictive-analytics/
├── predictive_engine.py          # Python ML & time-series forecasting engine
├── cleaned_historical_data.csv   # Preprocessed 2-year historical dataset (730 days)
├── forecast_data.json            # Model evaluation metrics & future predictions
├── index.html                    # Dashboard UI layout
├── style.css                     # Modern glassmorphism stylesheet & themes
├── app.js                        # Chart.js forecasting visualizations & Excel export
└── README.md                     # Project documentation
```

---

## 🚀 How to Run

1. **Re-train or update models** (Optional):
   ```bash
   python predictive_engine.py
   ```

2. **Launch Preview Dashboard**:
   ```bash
   python -m http.server 8082
   ```
   Open **http://localhost:8082** in your browser.
