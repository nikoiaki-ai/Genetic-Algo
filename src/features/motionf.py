from __future__ import annotations

import numpy as np
import pandas as pd


def _mag(x: pd.Series, y: pd.Series, z: pd.Series) -> pd.Series:
    return np.sqrt(x**2 + y**2 + z**2)


def _rms(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(s.to_numpy() ** 2)))


def _std(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.std(ddof=1)) if len(s) > 1 else float("nan")


def _mean(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.mean()) if len(s) else float("nan")


def _jerk_rms(time: pd.Series, signal: pd.Series) -> float:
    """
    Approx jerk = derivative of signal w.r.t time.
    Assumes time is monotonic (in whatever units).
    """
    t = time.to_numpy()
    x = signal.to_numpy()
    if len(t) < 3:
        return float("nan")
    dt = np.diff(t)
    dx = np.diff(x)
    # avoid divide-by-zero
    valid = dt != 0
    if valid.sum() < 2:
        return float("nan")
    deriv = dx[valid] / dt[valid]
    return float(np.sqrt(np.mean(deriv**2)))


def compute_imu_features(df: pd.DataFrame, side: str) -> dict:
    """
    Compute trial-level features for one side (RH or LH).
    Expects columns like: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, time
    """
    if df is None or df.empty or "side" not in df.columns:
        prefix = f"imu_{side.lower()}"
        return {
            f"{prefix}_n": 0,
            f"{prefix}_acc_mag_mean": np.nan,
            f"{prefix}_acc_mag_std": np.nan,
            f"{prefix}_acc_mag_rms": np.nan,
            f"{prefix}_acc_mag_jerk_rms": np.nan,
            f"{prefix}_gyro_mag_mean": np.nan,
            f"{prefix}_gyro_mag_std": np.nan,
            f"{prefix}_gyro_mag_rms": np.nan,
            f"{prefix}_gyro_mag_jerk_rms": np.nan,
        }
    # ---- END GUARD ----

    d = df[df["side"] == side].copy()
    if d.empty:
        return {f"imu_{side.lower()}_n": 0}

    feats = {f"imu_{side.lower()}_n": int(len(d))}

    # accel magnitude
    if {"acc_x", "acc_y", "acc_z"}.issubset(d.columns):
        acc_mag = _mag(d["acc_x"], d["acc_y"], d["acc_z"])
        feats[f"imu_{side.lower()}_acc_mag_mean"] = _mean(acc_mag)
        feats[f"imu_{side.lower()}_acc_mag_std"] = _std(acc_mag)
        feats[f"imu_{side.lower()}_acc_mag_rms"] = _rms(acc_mag)

        if "time" in d.columns:
            feats[f"imu_{side.lower()}_acc_mag_jerk_rms"] = _jerk_rms(d["time"], acc_mag)

    # gyro magnitude
    if {"gyro_x", "gyro_y", "gyro_z"}.issubset(d.columns):
        gyro_mag = _mag(d["gyro_x"], d["gyro_y"], d["gyro_z"])
        feats[f"imu_{side.lower()}_gyro_mag_mean"] = _mean(gyro_mag)
        feats[f"imu_{side.lower()}_gyro_mag_std"] = _std(gyro_mag)
        feats[f"imu_{side.lower()}_gyro_mag_rms"] = _rms(gyro_mag)

        if "time" in d.columns:
            feats[f"imu_{side.lower()}_gyro_mag_jerk_rms"] = _jerk_rms(d["time"], gyro_mag)

    # temperature summary (optional)
    if "temp_c" in d.columns:
        feats[f"imu_{side.lower()}_temp_mean"] = _mean(d["temp_c"])
        feats[f"imu_{side.lower()}_temp_std"] = _std(d["temp_c"])

    return feats

def compute_shimmer_features(df: pd.DataFrame, device: str) -> dict:
    """
    Same feature set as IMU, but using:
      device column instead of side
      timestamp_ms instead of time
    """
     # --- guard: missing/empty shimmer ---
    if df is None or df.empty or "device" not in df.columns:
        prefix = f"shim_{device.lower()}"
        return {
            f"{prefix}_n": 0,
            f"{prefix}_acc_mag_mean": np.nan,
            f"{prefix}_acc_mag_std": np.nan,
            f"{prefix}_acc_mag_rms": np.nan,
            f"{prefix}_acc_mag_jerk_rms": np.nan,
            f"{prefix}_gyro_mag_mean": np.nan,
            f"{prefix}_gyro_mag_std": np.nan,
            f"{prefix}_gyro_mag_rms": np.nan,
            f"{prefix}_gyro_mag_jerk_rms": np.nan,
        }

    d = df[df["device"] == device].copy()
    if d.empty:
        return {f"shim_{device.lower()}_n": 0}

    # Rename to reuse compute_imu_features logic:
    d = d.rename(columns={"device": "side", "timestamp_ms": "time"})
    feats = compute_imu_features(d, side=device)  # device label acts like side

    # Rename imu_* -> shim_*
    out = {}
    for k, v in feats.items():
        if k.startswith("imu_"):
            out["shim_" + k[len("imu_"):]] = v
        else:
            out[k] = v

    return out