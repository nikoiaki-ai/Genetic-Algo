import pandas as pd
import re

# Load your final dataset
df = pd.read_csv("outputs/master_trials_full_all.csv")

def task_num_from_path(p):
    m = re.search(r"Task A_(\d+)", str(p))
    return int(m.group(1)) if m else None

# Derive task number and stage
df["task_num"] = df["log_path"].apply(task_num_from_path)
df["task_stage"] = df["task_num"].apply(
    lambda x: "baseline" if x == 0 else "trained"
)

# Save corrected dataset
df.to_csv("outputs/master_trials_full_all_fixed.csv", index=False)
print("Wrote outputs/master_trials_full_all_fixed.csv")