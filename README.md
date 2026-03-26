# FleetFlex: Energy Optimization for Heat Pumps, Batteries, and Grid Support

This portfolio project simulates a fleet of electrically heated homes and shows how machine learning plus optimization can reduce energy cost while delivering grid flexibility.

## Why this matches the role

The job description focuses on three things:
1. deciding **when a heat pump should run**,
2. deciding **when a battery should charge or discharge**,
3. deciding **when to export to the grid**, including at a **fleet level**.

This project reproduces that logic in a portfolio-friendly way:
- a **synthetic but realistic hourly dataset** for 180 homes,
- ML models that forecast **uncontrollable load** and **solar generation**,
- a **linear optimization model** that schedules heat pump and battery dispatch,
- a **flexibility-event constraint** that asks the fleet to support the grid during critical hours.

## Project structure

```text
energy_optimization_portfolio/
├── data/
├── outputs/
├── requirements.txt
├── setup_venv.sh
├── src_generate_dataset.py
├── src_train_optimize.py
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
python src_generate_dataset.py
python src_train_optimize.py
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
