from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.parsers.imu import parse_imu_file
from src.features.motionf import compute_imu_features


def imu_path_from_log_path(log_path: str) -> Path:
    """
    Convert ..._log.txt -> ..._IMU.txt
    """
    p = Path(log_path)
    return Path(str(p).replace("_log.txt", "_IMU.txt"))


def append_imu_features(master_csv: str | Path, out_csv: str | Path) -> None:
    master_csv = Path(master_csv)
    df = pd.read_csv(master_csv)

    feat_rows = []
    for _, row in df.iterrows():
        imu_path = imu_path_from_log_path(row["log_path"])
        if not imu_path.exists():
            feat_rows.append({})
            continue

        imu_df = parse_imu_file(imu_path)
        feats = {}
        feats.update(compute_imu_features(imu_df, "RH"))
        feats.update(compute_imu_features(imu_df, "LH"))
        feat_rows.append(feats)

    feat_df = pd.DataFrame(feat_rows)
    out = pd.concat([df, feat_df], axis=1)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Wrote {len(out)} rows with IMU features to {out_csv}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("master_csv")
    p.add_argument("out_csv")
    args = p.parse_args()

    append_imu_features(args.master_csv, args.out_csv)
