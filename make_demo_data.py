"""Generate a FAKE demo CSV (3 bananas + an empty-box baseline, 5 days) into data/demo/.
Every row is marked synthetic. Never use this for training a real model."""
import numpy as np, pandas as pd
from src.data_manager import mark_as_synthetic, save_dataset
from src.data_validator import validate_sensor_data

rng = np.random.default_rng(2026)
ts = pd.date_range("2026-11-01 08:00:00", periods=5 * 24 * 2, freq="30min")   # 5 days, every 30 min
day = (ts - ts[0]).total_seconds().to_numpy() / 86400
frames = []
for name, speed in [("BANANA-A", 1.0), ("BANANA-B", 1.25), ("BANANA-C", 0.8), ("EMPTY-BOX", 0.0)]:
    ripe = 1 / (1 + np.exp(-(day * speed - 2.6) * 1.7))                  # S-curve: fresh -> spoiled
    gas = 1500 + 520 * ripe + 35 * np.sin(day * 2 * np.pi) + rng.normal(0, 14, len(ts))   # daily wobble + noise
    temp = 26 + 1.8 * np.sin((day - 0.3) * 2 * np.pi) + rng.normal(0, .15, len(ts))
    hum = 62 + 5 * ripe * (name != "EMPTY-BOX") + rng.normal(0, .6, len(ts))
    label = np.where(name == "EMPTY-BOX", "no_banana", np.where(ripe < .25, "fresh", np.where(ripe < .75, "ripe", "spoiled")))
    frames.append(pd.DataFrame({"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "sample_id": [f"{name}-{i:04d}" for i in range(len(ts))], "specimen_id": name,
        "gas_raw": np.rint(gas).astype(int), "temperature_c": temp.round(1), "humidity_percent": hum.clip(0, 100).round(1),
        "spoilage_label": label}))
df = mark_as_synthetic(pd.concat(frames, ignore_index=True))
res = save_dataset(df, "data/demo/demo_5day_3bananas.csv", origin="synthetic", overwrite=True)
print("saved" if res.saved else "NOT saved", len(df), "rows | valid:", validate_sensor_data(df, expected_origin="synthetic").is_valid)
