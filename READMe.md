# Oil Well AI Monitoring Dashboard

## Overview

This Streamlit application provides an interactive dashboard for monitoring oil well production and equipment health using AI-driven analytics. Features include:

* **Production Metrics**: Daily oil, gas, and water-sediment production charts.
* **Pump Health**: Real-time monitoring of pump speed, current draw, intake pressure, and motor temperature.
* **Anomaly Detection**: Combines Isolation Forest, Local Outlier Factor, and One-Class SVM to flag potential issues.
* **Predictive Maintenance**: Estimates remaining pump life using a Cox Proportional Hazards survival model.
* **Production Forecasting**: Short-term (ARIMA) and long-term (Prophet) production forecasts with uncertainty intervals.
* **Equipment Clustering**: K-Means clustering of operating modes.
* **Root Cause Analysis**: Permutation importance from Logistic Regression to rank anomaly drivers.
* **Automated Insights**: Decision-tree rules and treemaps for feature importance.

## Repository Structure

```plaintext
├── app.py                # Main Streamlit application script
├── requirements.txt      # Python dependencies
├── streamlit.toml        # Streamlit configuration
└── README.md             # This documentation file
```

## Requirements

* Python 3.8+
* Streamlit
* pandas
* numpy
* plotly
* scikit-learn
* lifelines
* statsmodels
* prophet

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Installation & Local Run

1. Clone the repository:

   ```bash
   ```

git clone [https://github.com/your-username/oil-well-dashboard.git](https://github.com/your-username/oil-well-dashboard.git)
cd oil-well-dashboard

````
2. Install dependencies:
   ```bash
pip install -r requirements.txt
````

3. Run the app locally:

   ```bash
   ```

streamlit run app.py

````
4. Open your browser at `http://localhost:8501`.

## Configuration
- **streamlit.toml**: Customize theme and server settings.
- **Environment Variables / Secrets**: (Optional) If connecting to external data sources, add credentials via Streamlit Cloud secrets or local `.env`:
  ```dotenv
  DB_USER=your_user
  DB_PASS=your_pass
  API_ENDPOINT=https://...
````

Access in Python with:

```python
import os
user = os.getenv("DB_USER")
```

## Deployment on Streamlit Cloud

1. Push your code to a GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://streamlit.io/cloud) with GitHub.
3. Click **New app**, select your repo and branch, and set `app.py` as the entry point.
4. Add any secrets under **Settings → Secrets**.
5. Click **Deploy**. Your dashboard will be live at `https://<your-app>.streamlitapp.com`.

## Usage

* **Production Tab**: View historical and current production metrics.
* **Pump Health Tab**: Monitor ESP vitals with safety thresholds.
* **Alerts Tab**: Inspect anomaly counts, timeline, and recent alerts.
* **Predictions Tab**: Check pump RUL and production forecasts.
* **Insights Tab**: Explore clustering, root causes, and AI-generated rules.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for enhancements and bug fixes.

## License

This project is licensed under the MIT License.
