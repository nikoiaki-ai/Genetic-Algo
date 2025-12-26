from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import pandas as pd


def _to_float(x: str):
    x = x.strip()
    try:
        return float(x)
    except ValueError:
        return x


def parse_shimmer_file(path: str | Path) -> pd.DataFrame:
    """
    Parses lines like:

    COM9,Shimmer_DA3F,
      System Timestamp;CAL;mSecs;1694450710054.46,
      Low Noise Accelerometer X;CAL;m/(sec^2);9.3855,
      ...
      Gyroscope X;CAL;deg/sec;-0.2290,
      ...

    Returns DataFrame with columns:
      device (COM9/COM10), timestamp_ms, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z, vbatt_mv
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    rows: List[Dict[str, Any]] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) < 3:
            continue

        device = parts[0]          # COM9 / COM10
        shimmer_id = parts[1]      # Shimmer_DA3F (optional)

        row: Dict[str, Any] = {
            "device": device,
            "shimmer_id": shimmer_id,
        }

        # Each remaining token looks like: "Name;CAL;units;value"
        for token in parts[2:]:
            if ";" not in token:
                continue
            segs = [s.strip() for s in token.split(";")]

            # We expect at least: Name, CAL, Units, Value
            if len(segs) < 4:
                continue

            name = segs[0]
            value = segs[-1]
            value = _to_float(value)

            name_l = name.lower()

            # Timestamp
            if "system timestamp" in name_l:
                row["timestamp_ms"] = value
                continue

            # Accelerometer (Low Noise Accelerometer X/Y/Z)
            if "low noise accelerometer x" in name_l:
                row["acc_x"] = value
                continue
            if "low noise accelerometer y" in name_l:
                row["acc_y"] = value
                continue
            if "low noise accelerometer z" in name_l:
                row["acc_z"] = value
                continue

            # Gyroscope X/Y/Z
            if name_l == "gyroscope x":
                row["gyro_x"] = value
                continue
            if name_l == "gyroscope y":
                row["gyro_y"] = value
                continue
            if name_l == "gyroscope z":
                row["gyro_z"] = value
                continue

            # Magnetometer X/Y/Z
            if name_l == "magnetometer x":
                row["mag_x"] = value
                continue
            if name_l == "magnetometer y":
                row["mag_y"] = value
                continue
            if name_l == "magnetometer z":
                row["mag_z"] = value
                continue

            # Battery voltage
            if "vsensebatt" in name_l:
                row["vbatt_mv"] = value
                continue

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Sort within device by time if present
    if "timestamp_ms" in df.columns:
        df = df.sort_values(["device", "timestamp_ms"]).reset_index(drop=True)

    return df