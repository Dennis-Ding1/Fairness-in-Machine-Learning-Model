import pandas as pd
import numpy as np
from pathlib import Path

def sim_data_preprocess(csv_path="../data/data_I.csv"):
    """
    Preprocess custom simulated survival data to match the FCPH/FDCPH method input.
    This version assumes data generated from the R function `generate_data()`.
    
    Expected columns:
        id, A, X1, X2, ..., Y, Delta
    """
    csv_path = Path(__file__).resolve().parent.joinpath(csv_path).resolve()
    df = pd.read_csv(csv_path)

    # ---- Sensitive attribute(s)
    protect_attr = df[["A"]].values  # (n, 1)

    # ---- Features (auto-detect X columns)
    x_cols = [col for col in df.columns if col.startswith("X")]
    data_x = df[x_cols].copy()

    # ---- Survival outcomes
    data_event =df["Delta"].astype(bool).values  # Opposite from data, 1 is censored here
    data_time = df["Y"].astype(float).values

    # ---- Convert to sksurv-compatible structured array
    data_y = np.empty(len(data_event), dtype=[("censor", bool), ("time", float)])
    data_y["censor"] = data_event
    data_y["time"] = data_time

    return data_x, data_y, protect_attr
