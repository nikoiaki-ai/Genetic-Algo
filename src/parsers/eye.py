from __future__ import annotations
from pathlib import Path
import pandas as pd

def parse_eye_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df