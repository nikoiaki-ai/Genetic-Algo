from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.parsers.shimmer import parse_shimmer_file
from src.features.motionf import compute_shimmer_features


def shimmer_path_from_log_path(log_path: str) -> Path:
    p = Path(log_path)
    # Match your actual casing:
    return Path(str(p).replace("_log.txt", "_Shimmer.txt"))


def append_shimmer_features(master_csv: str | Path, out_csv: str | Path) -> None:
    df = pd.read_csv(master_csv)

    feat_rows = []
    for _, row in df.iterrows():
        sh_path = shimmer_path_from_log_path(row["log_path"])
        if not sh_path.exists():
            feat_rows.append({})
            continue

        sh_df = parse_shimmer_file(sh_path)

        feats = {}
        feats.update(compute_shimmer_features(sh_df, "COM9"))
        feats.update(compute_shimmer_features(sh_df, "COM10"))

        feat_rows.append(feats)

    out = pd.concat([df, pd.DataFrame(feat_rows)], axis=1)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Wrote {len(out)} rows with Shimmer features to {out_csv}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("master_csv")
    p.add_argument("out_csv")
    args = p.parse_args()

    append_shimmer_features(args.master_csv, args.out_csv)