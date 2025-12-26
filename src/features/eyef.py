from __future__ import annotations
import numpy as np
import pandas as pd


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _basic_stats(x: pd.Series, prefix: str) -> dict:
    x = x.dropna()
    if len(x) == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_p95": np.nan,
        }
    return {
        f"{prefix}_n": int(len(x)),
        f"{prefix}_mean": float(x.mean()),
        f"{prefix}_std": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
        f"{prefix}_p95": float(np.percentile(x.to_numpy(), 95)) if len(x) > 1 else np.nan,
    }


def compute_eye_features(df: pd.DataFrame) -> dict:
    feats: dict = {"eye_rows": int(len(df))}

    # Column names (exact from your file)
    lg_valid = "LeftEye_GazePoint_Validity"
    lgx = "LeftEye_GazePoint_PositionOnDisplayArea_X"
    lgy = "LeftEye_GazePoint_PositionOnDisplayArea_Y"
    lp_valid = "LeftEye_Pupil_Validity"
    lp = "LeftEye_Pupil_PupilDiameter"

    rg_valid = "RightEye_GazePoint_Validity"
    rgx = "RightEye_GazePoint_PositionOnDisplayArea_X"
    rgy = "RightEye_GazePoint_PositionOnDisplayArea_Y"
    rp_valid = "RightEye_Pupil_Validity"
    rp = "RightEye_Pupil_PupilDiameter"

    for c in [lg_valid, lp_valid, rg_valid, rp_valid]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Missing/invalid rates
    feats["eye_left_gaze_valid_rate"]  = float((df[lg_valid] == "Valid").mean()) if lg_valid in df else np.nan
    feats["eye_right_gaze_valid_rate"] = float((df[rg_valid] == "Valid").mean()) if rg_valid in df else np.nan
    feats["eye_left_pupil_valid_rate"] = float((df[lp_valid] == "Valid").mean()) if lp_valid in df else np.nan
    feats["eye_right_pupil_valid_rate"]= float((df[rp_valid] == "Valid").mean()) if rp_valid in df else np.nan

    # --- Left gaze features (only valid samples) ---
    if all(c in df for c in [lg_valid, lgx, lgy]):
        left_mask = (df[lg_valid] == "Valid")
        left_x = _to_num(df.loc[left_mask, lgx])
        left_y = _to_num(df.loc[left_mask, lgy])

        feats.update(_basic_stats(left_x, "eye_left_gx"))
        feats.update(_basic_stats(left_y, "eye_left_gy"))

        # dispersion (spread)
        if left_x.dropna().shape[0] > 1 and left_y.dropna().shape[0] > 1:
            feats["eye_left_dispersion"] = float(np.sqrt(left_x.var(ddof=1) + left_y.var(ddof=1)))
        else:
            feats["eye_left_dispersion"] = np.nan

    # --- Right gaze features ---
    if all(c in df for c in [rg_valid, rgx, rgy]):
        right_mask = (df[rg_valid] == "Valid")
        right_x = _to_num(df.loc[right_mask, rgx])
        right_y = _to_num(df.loc[right_mask, rgy])

        feats.update(_basic_stats(right_x, "eye_right_gx"))
        feats.update(_basic_stats(right_y, "eye_right_gy"))

        if right_x.dropna().shape[0] > 1 and right_y.dropna().shape[0] > 1:
            feats["eye_right_dispersion"] = float(np.sqrt(right_x.var(ddof=1) + right_y.var(ddof=1)))
        else:
            feats["eye_right_dispersion"] = np.nan

    # --- Pupil features (valid only) ---
    if all(c in df for c in [lp_valid, lp]):
        lp_mask = (df[lp_valid] == "Valid")
        left_p = _to_num(df.loc[lp_mask, lp])
        feats.update(_basic_stats(left_p, "eye_left_pupil"))

    if all(c in df for c in [rp_valid, rp]):
        rp_mask = (df[rp_valid] == "Valid")
        right_p = _to_num(df.loc[rp_mask, rp])
        feats.update(_basic_stats(right_p, "eye_right_pupil"))

    # --- Combined features (average when both valid) ---
    if all(c in df for c in [lp_valid, lp, rp_valid, rp]):
        both_mask = (df[lp_valid] == "Valid") & (df[rp_valid] == "Valid")
        lpv = _to_num(df.loc[both_mask, lp])
        rpv = _to_num(df.loc[both_mask, rp])
        pupil_avg = (lpv + rpv) / 2.0
        feats.update(_basic_stats(pupil_avg, "eye_both_pupil_avg"))

        # pupil asymmetry
        diff = (lpv - rpv).abs()
        feats.update(_basic_stats(diff, "eye_pupil_asym_abs"))

     # --- Combined gaze features (average when both valid) ---
    if all(c in df for c in [lg_valid, lgx, lgy, rg_valid, rgx, rgy]):
        both_gaze = (df[lg_valid] == "Valid") & (df[rg_valid] == "Valid")

        lx = _to_num(df.loc[both_gaze, lgx])
        ly = _to_num(df.loc[both_gaze, lgy])
        rx = _to_num(df.loc[both_gaze, rgx])
        ry = _to_num(df.loc[both_gaze, rgy])

        gx_avg = (lx + rx) / 2.0
        gy_avg = (ly + ry) / 2.0

        feats.update(_basic_stats(gx_avg, "eye_both_gx_avg"))
        feats.update(_basic_stats(gy_avg, "eye_both_gy_avg"))

        # Combined dispersion (avg gaze)
        if gx_avg.dropna().shape[0] > 1 and gy_avg.dropna().shape[0] > 1:
            feats["eye_both_dispersion"] = float(np.sqrt(gx_avg.var(ddof=1) + gy_avg.var(ddof=1)))
        else:
            feats["eye_both_dispersion"] = np.nan

        # Left vs Right gaze disagreement (distance between gaze points)
        # This can spike during loss of tracking / vergence / poor calibration
        dx = (lx - rx).abs()
        dy = (ly - ry).abs()
        dist = np.sqrt((lx - rx) ** 2 + (ly - ry) ** 2)

        feats.update(_basic_stats(dx, "eye_gaze_dx_abs"))
        feats.update(_basic_stats(dy, "eye_gaze_dy_abs"))
        feats.update(_basic_stats(dist, "eye_gaze_lr_dist"))

    # --- Extra: overall validity summaries ---
    # % of rows where BOTH pupils valid / BOTH gaze valid
    if all(c in df for c in [lp_valid, rp_valid]):
        feats["eye_both_pupil_valid_rate"] = float(((df[lp_valid] == "Valid") & (df[rp_valid] == "Valid")).mean())
    else:
        feats["eye_both_pupil_valid_rate"] = np.nan

    if all(c in df for c in [lg_valid, rg_valid]):
        feats["eye_both_gaze_valid_rate"] = float(((df[lg_valid] == "Valid") & (df[rg_valid] == "Valid")).mean())
    else:
        feats["eye_both_gaze_valid_rate"] = np.nan

    return feats