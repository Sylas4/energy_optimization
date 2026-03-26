# 🔋 Energy Optimization for Smart Home Fleets

This project simulates and optimizes energy usage across a fleet of residential homes equipped with:
	•	Heat pumps
	•	Batteries
	•	Solar panels

## 🚀 Objective

Minimize energy costs and support the grid by:
	•	Scheduling heat pump operation
	•	Charging/discharging batteries
	•	Exporting energy during high-demand events

⸻

## 🧠 Approach
	1.	Synthetic data generation
	•	180 homes
	•	Weather-driven load + solar
	•	Dynamic electricity pricing
	2.	Forecasting (ML)
	•	Predict hourly load and solar generation
	•	Models: Random Forest
	3.	Optimization (Linear Programming)
	•	Battery constraints
	•	Comfort constraints (heating)
	•	Grid event participation

⸻

## 📊 Results
	•	Load Forecast MAE: 0.86 kW
	•	Solar Forecast MAE: 0.09 kW
	•	Cost reduction: ~4.7%
	•	Grid import reduction during events: ~33%

⸻

## 🛠 Tech Stack
	•	Python
	•	pandas, numpy
	•	scikit-learn
	•	scipy (optimization)
	•	matplotlib

## Project structure

```text
energy_optimization/
├── outputs/
├── src/
    ── src_generate_dataset.py
    ── src_train_optimize.py
├── requirements.txt
├── setup_venv.sh
└── README.md
```

## Dataset

The dataset is synthetic so the project is easy to run anywhere, but it is inspired by real energy-data sources such as:
- weather-driven hourly context,
- residential household electricity patterns,
- dynamic day-ahead style pricing,
- flexibility-event windows in winter peak hours.

Generated fields include:
- outdoor temperature,
- solar irradiance,
- retail import price,
- feed-in tariff,
- appliance load,
- solar generation,
- heat pump load,
- net load,
- per-home battery and heat pump capacities.

## Modeling approach

### 1) Forecasting
Two gradient boosting models forecast:
- fleet appliance load,
- fleet solar generation.

Features:
- hour of day,
- day of week,
- month,
- outdoor temperature,
- solar irradiance,
- price,
- flexibility event flag.

### 2) Optimization
The optimizer controls, hour by hour:
- heat pump power,
- battery charging,
- battery discharging,
- grid import,
- grid export,
- battery state of charge,
- indoor temperature.

Objective:
- minimize energy cost,
- earn export revenue,
- keep a comfort band,
- reduce import during flexibility events,
- hit a fleet export target during grid-support hours.

### 3) Comparison
A rule-based baseline is compared against optimized dispatch over the final evaluation week.

## How to run

```bash
cd energy_optimization_portfolio
bash setup_venv.sh
source .venv/bin/activate
python src/src_generate_dataset.py
python src/src_train_optimize.py
```

## What to show on GitHub

In your repo description and README, highlight:
- **forecast + optimization** in one pipeline,
- **heat pump + battery + export control**,
- **fleet-level flexibility events**,
- **clear business KPIs** like cost reduction and event-hour import reduction.

## Ideas for a stronger v2

- replace synthetic price with real Nord Pool day-ahead data,
- replace synthetic weather with Meteostat historical weather,
- train a 24-hour ahead forecast instead of same-hour forecast,
- optimize per home, then aggregate into a fleet market bid,
- add reinforcement learning or MPC for rolling control.
