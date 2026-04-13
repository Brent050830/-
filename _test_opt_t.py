import pandas as pd
import numpy as np

def test_optimal_torque():
    df = pd.read_csv("sys_eff_pivot.csv", index_col=0)
    torques = df.index.values.astype(float)
    speeds = df.columns.values.astype(float)
    eff = df.values.astype(float) / 100.0
    
    s = 4000
    diffs = np.abs(speeds - s)
    closest_si = np.argmin(diffs)
    col_eff = np.nan_to_num(eff[:, closest_si], nan=0.0)
    
    mask = torques >= 0.0
    valid_effs = np.where(mask, col_eff, -1.0)
    best_idx = np.argmax(valid_effs)
    opt_t = torques[best_idx]
    print(f"At {s} rpm, positive optimal torque is {opt_t} Nm with eff {valid_effs[best_idx]}")
    
    mask = torques <= 0.0
    valid_effs = np.where(mask, col_eff, -1.0)
    best_idx = np.argmax(valid_effs)
    opt_t = torques[best_idx]
    print(f"At {s} rpm, negative optimal torque is {opt_t} Nm with eff {valid_effs[best_idx]}")

test_optimal_torque()
