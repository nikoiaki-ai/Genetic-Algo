from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import pandas as pd


def _parse_kv_line(line: str) -> Dict[str, Any]:
    """
    Example line:
    RH,Time:681179,Temp(C):21.97,Acc_X:-7.31,Acc_Y:5.91,...

    Returns a dict like:
    {
      "side": "RH",
      "time": 681179,
      "temp_c": 21.97,
      "acc_x": -7.31, ...
    }
    """
    parts = [p.strip() for p in line.strip().split(",") if p.strip()]
    if not parts:
        return {}

    side = parts[0]
    out: Dict[str, Any] = {"side": side}

    for token in parts[1:]:
        if ":" not in token:
            continue
        k, v = token.split(":", 1)
        k = k.strip()

        # normalize keys
        k_norm = (
            k.lower()
             .replace("(c)", "_c")
             .replace("(", "_")
             .replace(")", "")
             .replace("/", "_")
             .replace(" ", "_")
        )

        v = v.strip()
        # numeric cast when possible
        try:
            if "." in v or "e" in v.lower():
                out[k_norm] = float(v)
            else:
                out[k_norm] = int(v)
        except ValueError:
            out[k_norm] = v

    return out


def parse_imu_file(path: str | Path) -> pd.DataFrame:
    """
    Parses IMU text where each line looks like:
      RH,Time:681179,Temp(C):21.97,Acc_X:-7.31,...
      LH,Time:693963,Temp(C):22.34,Acc_X:-7.81,...

    Returns a DataFrame with a required 'side' column.
    """
    path = Path(path)

    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = _parse_kv_line(line)
            if rec:
                rows.append(rec)

    return pd.DataFrame(rows)