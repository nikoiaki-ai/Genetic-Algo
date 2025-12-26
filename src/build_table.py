from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.parsers.log import parse_log_file


def build_table(root_dir: str | Path, out_csv: str | Path) -> None:
    root_dir = Path(root_dir)
    log_files = sorted(root_dir.rglob("*_log.txt"))

    rows = []
    for lf in log_files:
        record = parse_log_file(lf)
        rows.append({
            "participant_id": record.participant_id,
            "task": record.task,
            "session_id": record.session_id,
            "trial_id": record.trial_id,
            "completion_time_s": record.completion_time,
            "log_path": str(lf),
            "task_stage": infer_stage(record.task)
        })

    if not rows:
        print(f"No *_log.txt files found under: {root_dir.resolve()}")
        return

    df = pd.DataFrame(rows).sort_values(["participant_id", "task", "session_id", "trial_id"])
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows to {out_csv}")

def infer_stage(task: str) -> str:
    if task.endswith("A_00"):
        return "baseline"
    else:
        return "trained"



if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("root_dir", help="Folder containing extracted participant/task files")
    p.add_argument("out_csv", help="Output CSV path")
    args = p.parse_args()

    build_table(args.root_dir, args.out_csv)