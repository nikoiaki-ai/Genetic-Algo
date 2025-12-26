from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.parsers.eye import parse_eye_file
from src.features.eyef import compute_eye_features


def eye_path_from_log_path(log_path: str) -> Path:
    """
    Convert:
      ..._log.txt  ->  ..._eye.txt
    """
    p = Path(log_path)
    return Path(str(p).replace("_log.txt", "_eye.txt"))


def append_eye_features(master_csv: str | Path, out_csv: str | Path) -> None:
    master_csv = Path(master_csv)
    df = pd.read_csv(master_csv)

    eye_feat_rows = []

    for _, row in df.iterrows():
        eye_path = eye_path_from_log_path(row["log_path"])

        if not eye_path.exists():
            # No eye file → empty feature dict (will become NaNs)
            eye_feat_rows.append({})
            continue

        eye_df = parse_eye_file(eye_path)
        feats = compute_eye_features(eye_df)
        eye_feat_rows.append(feats)

    eye_feat_df = pd.DataFrame(eye_feat_rows)

    # Append column-wise
    out = pd.concat([df.reset_index(drop=True),
                     eye_feat_df.reset_index(drop=True)], axis=1)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    print(f"Wrote {len(out)} rows with Eye features to {out_csv}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("master_csv", help="CSV with IMU + Shimmer features")
    p.add_argument("out_csv", help="Output CSV with Eye features appended")
    args = p.parse_args()

    append_eye_features(args.master_csv, args.out_csv)