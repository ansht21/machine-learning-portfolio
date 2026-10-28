"""Fair SARIMA vs seasonal-naive comparison (item-3 fix).

Run:  <venv> python compare_baselines.py
Both forecasts are FIXED multi-step over the identical test window:
  SARIMA          - single fit on train, one get_forecast(len(test))
  seasonal-naive  - repeat the last 7 TRAIN days across the test window
                    (no within-test actuals are revealed to the baseline)
Zero-demand days are excluded from MAPE denominators; WAPE reported too.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

df = pd.read_csv("data/Uber_Cleaned_data.csv")
if "request_date" in df.columns:
    datecol = "request_date"
elif "date" in df.columns:
    datecol = "date"
else:
    datecol = [c for c in df.columns if "date" in c.lower()][0]
daily = df.groupby(datecol).size().rename("trips").reset_index()
daily[datecol] = pd.to_datetime(daily[datecol])
daily = daily.sort_values(datecol).reset_index(drop=True)
split = int(len(daily) * 0.8333)  # Jan-Oct train, Nov-Dec test (as notebook)
train, test = daily.trips.iloc[:split], daily.trips.iloc[split:]

model = SARIMAX(train.values, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7),
                enforce_stationarity=False, enforce_invertibility=False)
res = model.fit(disp=False)
pred = np.clip(res.get_forecast(len(test)).predicted_mean, 0, None)

# FAIR baseline: fixed multi-step, last training week repeated over the horizon
naive = np.tile(train.values[-7:], int(np.ceil(len(test) / 7)))[:len(test)]

def metrics(actual, estimate, label):
    nz = actual > 0
    mae = float(np.mean(np.abs(actual - estimate)))
    rmse = float(np.sqrt(np.mean((actual - estimate) ** 2)))
    mape = float(np.mean(np.abs(actual - estimate)[nz] / actual[nz]) * 100)
    wape = float(np.abs(actual - estimate).sum() / actual.sum() * 100)
    print(f"{label:<22} MAE {mae:.1f} | RMSE {rmse:.1f} | MAPE {mape:.1f}% | WAPE {wape:.1f}%")
    return mape, wape

print(f"test window: {len(test)} days; zero-demand days: {int((test == 0).sum())}")
m_sar, w_sar = metrics(test.values, pred, "SARIMA (fixed 60d)")
m_nav, w_nav = metrics(test.values, naive, "seasonal-naive (fixed)")
print(f"MAPE improvement vs fair baseline: {(m_nav - m_sar) / m_nav * 100:.1f}%")
print(f"WAPE improvement vs fair baseline: {(w_nav - w_sar) / w_nav * 100:.1f}%")
