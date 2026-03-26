from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd


OUTPUT_DIR = Path('../data/energy_optimization_portfolio/data')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def seasonal_daily_profile(hour: np.ndarray, morning_peak: float, evening_peak: float) -> np.ndarray:
    """Smooth residential demand shape with morning and evening peaks."""
    morning = np.exp(-0.5 * ((hour - 7) / 2.0) ** 2) * morning_peak
    evening = np.exp(-0.5 * ((hour - 19) / 3.0) ** 2) * evening_peak
    night = 0.2 + 0.15 * np.exp(-0.5 * ((hour - 1) / 2.5) ** 2)
    return night + morning + evening


rng = np.random.default_rng(42)
start = '2024-01-01 00:00:00'
end = '2024-12-31 23:00:00'
idx = pd.date_range(start, end, freq='h')

hours = idx.hour.to_numpy()
dayofyear = idx.dayofyear.to_numpy()
weekday = idx.dayofweek.to_numpy()
month = idx.month.to_numpy()

# Synthetic weather inspired by Nordic seasonality.
annual_temp = 7 + 12 * np.sin(2 * np.pi * (dayofyear - 110) / 366)
daily_temp = 2 * np.sin(2 * np.pi * (hours - 14) / 24)
outdoor_temp = annual_temp + daily_temp + rng.normal(0, 1.8, len(idx))

# Approximate solar irradiance pattern for a northern latitude.
daylight_hours = 8 + 8 * np.sin(2 * np.pi * (dayofyear - 80) / 366)
sunrise = 12 - daylight_hours / 2
sunset = 12 + daylight_hours / 2
sun_position = np.clip((hours - sunrise) / np.maximum(daylight_hours, 1e-3), 0, 1)
clear_sky = np.sin(np.pi * sun_position)
cloud_factor = np.clip(0.75 + 0.25 * np.sin(2 * np.pi * dayofyear / 7) + rng.normal(0, 0.12, len(idx)), 0.15, 1.0)
solar_irradiance = 850 * clear_sky * cloud_factor

# Dynamic retail price proxy inspired by day-ahead price variation.
base_price = 40 + 18 * np.exp(-0.5 * ((hours - 8) / 2.8) ** 2) + 28 * np.exp(-0.5 * ((hours - 18) / 3.5) ** 2)
seasonal_markup = np.clip((12 - outdoor_temp) * 1.8, 0, None)
weekend_discount = np.where(weekday >= 5, -8, 0)
price_eur_mwh = np.clip(base_price + seasonal_markup + weekend_discount + rng.normal(0, 6, len(idx)), 5, None)
feed_in_tariff = np.clip(price_eur_mwh * 0.72 - 3, 0, None)

# Flexibility events emulate grid support windows.
event_signal = (((month <= 3) | (month >= 11)) & (hours >= 17) & (hours <= 20) & (weekday < 5)).astype(int)

weather = pd.DataFrame(
    {
        'timestamp': idx,
        'outdoor_temp_c': outdoor_temp,
        'solar_irradiance_wm2': solar_irradiance,
        'price_eur_mwh': price_eur_mwh,
        'feed_in_tariff_eur_mwh': feed_in_tariff,
        'flex_event': event_signal,
    }
)
weather.to_csv(OUTPUT_DIR / 'hourly_context.csv', index=False)

n_homes = 180
homes = []
records: list[pd.DataFrame] = []

for home_id in range(n_homes):
    heat_loss = rng.uniform(0.12, 0.24)
    thermal_gain = rng.uniform(0.30, 0.55)
    base_load_kw = rng.uniform(0.25, 0.55)
    morning_peak = rng.uniform(0.45, 0.9)
    evening_peak = rng.uniform(0.8, 1.4)
    solar_kwp = rng.choice([0, 2.5, 4.0, 6.0], p=[0.20, 0.30, 0.30, 0.20])
    battery_kwh = rng.choice([0, 5, 10, 13.5], p=[0.18, 0.22, 0.40, 0.20])
    hp_max_kw = rng.uniform(2.5, 5.0)
    battery_power_kw = 0.0 if battery_kwh == 0 else min(4.5, battery_kwh / 2.5)
    comfort = rng.uniform(20.4, 21.5)

    occupancy_boost = np.where(weekday >= 5, 1.08, 1.0)
    profile = seasonal_daily_profile(hours, morning_peak, evening_peak)
    appliance_load = (base_load_kw + profile) * occupancy_boost
    appliance_load += rng.normal(0, 0.08, len(idx))
    appliance_load = np.clip(appliance_load, 0.15, None)

    solar_gen_kw = solar_kwp * solar_irradiance / 1000 * rng.uniform(0.78, 0.9)
    solar_gen_kw = np.clip(solar_gen_kw, 0, None)

    heating_need = np.clip(comfort - outdoor_temp, 0, None)
    heat_pump_kw = np.clip(0.12 * heating_need / thermal_gain, 0, hp_max_kw)
    heat_pump_kw += rng.normal(0, 0.12, len(idx))
    heat_pump_kw = np.clip(heat_pump_kw, 0, hp_max_kw)

    net_load_kw = appliance_load + heat_pump_kw - solar_gen_kw

    home_df = pd.DataFrame(
        {
            'timestamp': idx,
            'home_id': home_id,
            'appliance_load_kw': appliance_load,
            'heat_pump_kw': heat_pump_kw,
            'solar_gen_kw': solar_gen_kw,
            'net_load_kw': net_load_kw,
            'thermal_gain_coeff': thermal_gain,
            'heat_loss_coeff': heat_loss,
            'battery_kwh': battery_kwh,
            'battery_power_kw': battery_power_kw,
            'hp_max_kw': hp_max_kw,
            'comfort_setpoint_c': comfort,
        }
    )
    records.append(home_df)
    homes.append(
        {
            'home_id': home_id,
            'thermal_gain_coeff': round(thermal_gain, 4),
            'heat_loss_coeff': round(heat_loss, 4),
            'battery_kwh': float(battery_kwh),
            'battery_power_kw': float(battery_power_kw),
            'solar_kwp': float(solar_kwp),
            'hp_max_kw': round(hp_max_kw, 3),
            'comfort_setpoint_c': round(comfort, 3),
        }
    )

fleet = pd.concat(records, ignore_index=True)
fleet.to_csv(OUTPUT_DIR / 'synthetic_fleet.csv', index=False)
pd.DataFrame(homes).to_csv(OUTPUT_DIR / 'home_metadata.csv', index=False)

agg = (
    fleet.groupby('timestamp', as_index=False)[['appliance_load_kw', 'heat_pump_kw', 'solar_gen_kw', 'net_load_kw']]
    .sum()
    .merge(weather, on='timestamp', how='left')
)
agg['month'] = agg['timestamp'].dt.month
agg['hour'] = agg['timestamp'].dt.hour
agg['dow'] = agg['timestamp'].dt.dayofweek
agg.to_csv(OUTPUT_DIR / 'fleet_hourly.csv', index=False)

summary = {
    'rows_fleet_hourly': int(len(agg)),
    'rows_synthetic_fleet': int(len(fleet)),
    'homes': n_homes,
    'avg_daily_load_mwh': float(agg['net_load_kw'].clip(lower=0).sum() / 1000 / 366),
    'avg_daily_solar_mwh': float(agg['solar_gen_kw'].sum() / 1000 / 366),
    'event_hours': int(agg['flex_event'].sum()),
}
with open(OUTPUT_DIR / 'dataset_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

print('Saved dataset to', OUTPUT_DIR)
print(json.dumps(summary, indent=2))
