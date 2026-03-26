from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

BASE_DIR = Path('./')
DATA_DIR = BASE_DIR / 'data'
OUT_DIR = BASE_DIR / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    feat['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    feat['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    feat['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    feat['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    feat['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    feat['outdoor_temp_c'] = df['outdoor_temp_c']
    feat['solar_irradiance_wm2'] = df['solar_irradiance_wm2']
    feat['price_eur_mwh'] = df['price_eur_mwh']
    feat['flex_event'] = df['flex_event']
    return feat


def fit_models(df: pd.DataFrame):
    split = int(len(df) * 0.8)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    X_train = make_features(train)
    X_test = make_features(test)

    y_load_train = train['appliance_load_kw']
    y_load_test = test['appliance_load_kw']
    y_solar_train = train['solar_gen_kw']
    y_solar_test = test['solar_gen_kw']

    load_model = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.06, max_iter=350, random_state=42)
    solar_model = HistGradientBoostingRegressor(max_depth=5, learning_rate=0.05, max_iter=320, random_state=42)

    load_model.fit(X_train, y_load_train)
    solar_model.fit(X_train, y_solar_train)

    test = test.copy()
    test['pred_appliance_load_kw'] = load_model.predict(X_test)
    test['pred_solar_gen_kw'] = np.clip(solar_model.predict(X_test), 0, None)

    metrics = {
        'appliance_load_mae_kw': float(mean_absolute_error(y_load_test, test['pred_appliance_load_kw'])),
        'appliance_load_rmse_kw': float(root_mean_squared_error(y_load_test, test['pred_appliance_load_kw'])),
        'solar_mae_kw': float(mean_absolute_error(y_solar_test, test['pred_solar_gen_kw'])),
        'solar_rmse_kw': float(root_mean_squared_error(y_solar_test, test['pred_solar_gen_kw'])),
    }
    return test, metrics


def run_baseline(horizon: pd.DataFrame, params: dict) -> pd.DataFrame:
    n = len(horizon)
    soc = np.zeros(n + 1)
    soc[0] = 0.45 * params['battery_kwh']
    indoor = np.zeros(n + 1)
    indoor[0] = 21.0

    hp = np.zeros(n)
    charge = np.zeros(n)
    discharge = np.zeros(n)
    imp = np.zeros(n)
    exp = np.zeros(n)

    for t in range(n):
        temp_out = horizon.iloc[t]['outdoor_temp_c']
        comfort = 21.0
        required_hp = (comfort - indoor[t] - params['temp_loss'] * (temp_out - indoor[t])) / max(params['temp_gain'], 1e-6)
        hp[t] = np.clip(required_hp, 0, params['hp_max_kw'])
        indoor[t + 1] = indoor[t] + params['temp_loss'] * (temp_out - indoor[t]) + params['temp_gain'] * hp[t]

        raw_net = horizon.iloc[t]['pred_appliance_load_kw'] + hp[t] - horizon.iloc[t]['pred_solar_gen_kw']
        price = horizon.iloc[t]['price_eur_mwh']
        solar_surplus = max(-raw_net, 0)
        demand = max(raw_net, 0)

        if solar_surplus > 0 and params['battery_kwh'] > 0:
            charge[t] = min(solar_surplus, params['battery_power_kw'], params['battery_kwh'] - soc[t])
        elif price >= np.quantile(horizon['price_eur_mwh'], 0.75) and params['battery_kwh'] > 0:
            discharge[t] = min(demand, params['battery_power_kw'], soc[t])

        soc[t + 1] = soc[t] + 0.95 * charge[t] - discharge[t] / 0.95
        net = raw_net + charge[t] - discharge[t]
        imp[t] = max(net, 0)
        exp[t] = max(-net, 0)

    return pd.DataFrame(
        {
            'timestamp': horizon['timestamp'].to_numpy(),
            'hp_kw': hp,
            'charge_kw': charge,
            'discharge_kw': discharge,
            'soc_kwh': soc[1:],
            'indoor_temp_c': indoor[1:],
            'grid_import_kw': imp,
            'grid_export_kw': exp,
        }
    )


def optimize_schedule(horizon: pd.DataFrame, params: dict) -> pd.DataFrame:
    n = len(horizon)
    idx = {}
    cursor = 0
    for name, size in [('hp', n), ('charge', n), ('discharge', n), ('soc', n + 1), ('temp', n + 1), ('imp', n), ('exp', n), ('shortfall', n)]:
        idx[name] = np.arange(cursor, cursor + size)
        cursor += size
    m = cursor

    c = np.zeros(m)
    c[idx['imp']] = horizon['price_eur_mwh'].to_numpy() / 1000
    c[idx['exp']] = -horizon['feed_in_tariff_eur_mwh'].to_numpy() / 1000
    c[idx['hp']] = 0.002
    c[idx['shortfall']] = horizon['flex_event'].to_numpy() * 0.9

    bounds = [(0, None)] * m
    for i in idx['hp']:
        bounds[i] = (0, params['hp_max_kw'])
    for i in idx['charge']:
        bounds[i] = (0, params['battery_power_kw'])
    for i in idx['discharge']:
        bounds[i] = (0, params['battery_power_kw'])
    for i in idx['soc']:
        bounds[i] = (0, params['battery_kwh'])
    for i in idx['temp']:
        bounds[i] = (19.5, 23.8)
    for i in idx['imp']:
        bounds[i] = (0, None)
    for i in idx['exp']:
        bounds[i] = (0, None)
    for i in idx['shortfall']:
        bounds[i] = (0, None)

    A_eq = []
    b_eq = []

    row = np.zeros(m)
    row[idx['soc'][0]] = 1
    A_eq.append(row)
    b_eq.append(0.45 * params['battery_kwh'])

    row = np.zeros(m)
    row[idx['temp'][0]] = 1
    A_eq.append(row)
    b_eq.append(21.0)

    for t in range(n):
        row = np.zeros(m)
        row[idx['soc'][t + 1]] = 1
        row[idx['soc'][t]] = -1
        row[idx['charge'][t]] = -0.95
        row[idx['discharge'][t]] = 1 / 0.95
        A_eq.append(row)
        b_eq.append(0)

        row = np.zeros(m)
        row[idx['temp'][t + 1]] = 1
        row[idx['temp'][t]] = -(1 - params['temp_loss'])
        row[idx['hp'][t]] = -params['temp_gain']
        A_eq.append(row)
        b_eq.append(params['temp_loss'] * horizon.iloc[t]['outdoor_temp_c'])

        row = np.zeros(m)
        row[idx['imp'][t]] = 1
        row[idx['exp'][t]] = -1
        row[idx['charge'][t]] = -1
        row[idx['discharge'][t]] = 1
        row[idx['hp'][t]] = -1
        A_eq.append(row)
        b_eq.append(horizon.iloc[t]['pred_appliance_load_kw'] - horizon.iloc[t]['pred_solar_gen_kw'])

    A_ub = []
    b_ub = []
    event_target = params['event_export_target_kw']
    for t in range(n):
        if horizon.iloc[t]['flex_event'] == 1:
            row = np.zeros(m)
            row[idx['shortfall'][t]] = -1
            row[idx['exp'][t]] = -1
            A_ub.append(row)
            b_ub.append(-event_target)

            row = np.zeros(m)
            row[idx['imp'][t]] = 1
            A_ub.append(row)
            b_ub.append(params['max_event_import_kw'])

    res = linprog(
        c=c,
        A_ub=np.array(A_ub) if A_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=np.array(A_eq),
        b_eq=np.array(b_eq),
        bounds=bounds,
        method='highs',
    )
    if not res.success:
        raise RuntimeError(f'Optimization failed: {res.message}')

    x = res.x
    return pd.DataFrame(
        {
            'timestamp': horizon['timestamp'].to_numpy(),
            'hp_kw': x[idx['hp']],
            'charge_kw': x[idx['charge']],
            'discharge_kw': x[idx['discharge']],
            'soc_kwh': x[idx['soc']][1:],
            'indoor_temp_c': x[idx['temp']][1:],
            'grid_import_kw': x[idx['imp']],
            'grid_export_kw': x[idx['exp']],
            'event_shortfall_kw': x[idx['shortfall']],
        }
    )


def summarize_dispatch(label: str, dispatch: pd.DataFrame, horizon: pd.DataFrame) -> dict:
    merged = dispatch.merge(
        horizon[['timestamp', 'price_eur_mwh', 'feed_in_tariff_eur_mwh', 'flex_event', 'pred_solar_gen_kw']],
        on='timestamp',
        how='left',
    )
    import_cost = (merged['grid_import_kw'] * merged['price_eur_mwh'] / 1000).sum()
    export_revenue = (merged['grid_export_kw'] * merged['feed_in_tariff_eur_mwh'] / 1000).sum()
    event_mask = merged['flex_event'] == 1
    return {
        'strategy': label,
        'total_cost_eur': float(import_cost - export_revenue),
        'grid_import_kwh': float(merged['grid_import_kw'].sum()),
        'grid_export_kwh': float(merged['grid_export_kw'].sum()),
        'event_import_kwh': float(merged.loc[event_mask, 'grid_import_kw'].sum()),
        'event_export_kwh': float(merged.loc[event_mask, 'grid_export_kw'].sum()),
        'avg_indoor_temp_c': float(dispatch['indoor_temp_c'].mean()),
    }


def main() -> None:
    df = pd.read_csv(DATA_DIR / 'fleet_hourly.csv', parse_dates=['timestamp'])
    test, metrics = fit_models(df)
    test.to_csv(OUT_DIR / 'test_predictions.csv', index=False)

    horizon = test.tail(24 * 7).copy().reset_index(drop=True)
    params = {
        'battery_kwh': 820.0,
        'battery_power_kw': 240.0,
        'hp_max_kw': float(df['heat_pump_kw'].quantile(0.98) * 1.05),
        'temp_loss': 0.035,
        'temp_gain': 0.022,
        'event_export_target_kw': 80.0,
        'max_event_import_kw': 260.0,
    }

    baseline = run_baseline(horizon, params)
    optimized = optimize_schedule(horizon, params)

    baseline_summary = summarize_dispatch('baseline_rules', baseline, horizon)
    optimized_summary = summarize_dispatch('optimized_lp', optimized, horizon)
    comparison = pd.DataFrame([baseline_summary, optimized_summary])
    comparison['cost_savings_vs_baseline_eur'] = comparison['total_cost_eur'].iloc[0] - comparison['total_cost_eur']
    comparison.to_csv(OUT_DIR / 'strategy_comparison.csv', index=False)

    report = {
        'forecast_metrics': metrics,
        'baseline': baseline_summary,
        'optimized': optimized_summary,
        'cost_reduction_pct': float((baseline_summary['total_cost_eur'] - optimized_summary['total_cost_eur']) / baseline_summary['total_cost_eur'] * 100),
        'event_import_reduction_pct': float((baseline_summary['event_import_kwh'] - optimized_summary['event_import_kwh']) / max(baseline_summary['event_import_kwh'], 1e-6) * 100),
    }
    with open(OUT_DIR / 'report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    plt.figure(figsize=(12, 5))
    plt.plot(horizon['timestamp'], horizon['pred_appliance_load_kw'] + baseline['hp_kw'] - horizon['pred_solar_gen_kw'], label='Baseline net load')
    plt.plot(horizon['timestamp'], optimized['grid_import_kw'] - optimized['grid_export_kw'], label='Optimized net grid power')
    plt.plot(horizon['timestamp'], horizon['flex_event'] * 140, label='Flex event signal')
    plt.xticks(rotation=45)
    plt.ylabel('kW')
    plt.title('Fleet dispatch over the evaluation week')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'dispatch_week.png', dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(horizon['timestamp'], baseline['soc_kwh'], label='Baseline battery SoC')
    plt.plot(horizon['timestamp'], optimized['soc_kwh'], label='Optimized battery SoC')
    plt.xticks(rotation=45)
    plt.ylabel('kWh')
    plt.title('Battery state of charge')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'soc_week.png', dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.bar(comparison['strategy'], comparison['total_cost_eur'])
    plt.ylabel('EUR over 7 days')
    plt.title('Cost comparison')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'cost_comparison.png', dpi=150)
    plt.close()

    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
