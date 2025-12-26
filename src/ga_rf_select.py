from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_score

import time
start_time = time.time()



# -----------------------------
# Config
# -----------------------------
@dataclass
class GAConfig:
    seed: int = 42
    pop_size: int = 60
    generations: int = 40
    tournament_k: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 0.02  # per-gene
    elitism: int = 4

    # Feature subset constraints (prevents empty/all features)
    min_features: int = 8
    max_features: int = 60  # set None to disable cap

    # CV: use participant_id grouping to avoid leakage
    n_splits: int = 5

    # RF hyperparams (solid default; tune later)
    rf_params: Dict = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# -----------------------------
# Data
# -----------------------------
def load_xy_groups(csv_path: str | Path) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(csv_path)

    # target + group
    y = df["completion_time_s"].astype(float)
    groups = df["participant_id"].astype(str)

    # drop non-features
    drop_cols = {
        "participant_id", "task", "session_id", "trial_id",
        "log_path", "completion_time_s",
        "task_stage", "task_num"
    }
    X = df.drop(columns=[c for c in df.columns if c in drop_cols], errors="ignore")

    # keep numeric only (RF can handle unscaled numerics)
    X = X.apply(pd.to_numeric, errors="coerce")

    # basic missing handling: fill with median
    X = X.fillna(X.median(numeric_only=True))

    return X, y, groups


# -----------------------------
# Fitness (CV MAE)
# -----------------------------
def score_mask(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    mask: np.ndarray,
    cfg: GAConfig
) -> float:
    # Enforce min/max features
    k = int(mask.sum())
    if k < cfg.min_features:
        return 1e9  # big MAE penalty
    if cfg.max_features is not None and k > cfg.max_features:
        return 1e9

    Xsub = X.loc[:, mask]

    gkf = GroupKFold(n_splits=cfg.n_splits)

    model = RandomForestRegressor(**cfg.rf_params)

    # cross_val_score returns NEG MAE if we use neg_mean_absolute_error
    scores = cross_val_score(
        model, Xsub, y,
        cv=gkf,
        groups=groups,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )
    mae = -float(scores.mean())
    return mae


# -----------------------------
# GA ops
# -----------------------------
def init_population(n_features: int, cfg: GAConfig) -> List[np.ndarray]:
    pop = []
    for _ in range(cfg.pop_size):
        # sample subset size in a reasonable range
        hi = cfg.max_features if cfg.max_features is not None else n_features
        size = random.randint(cfg.min_features, min(hi, n_features))
        idx = np.random.choice(n_features, size=size, replace=False)
        mask = np.zeros(n_features, dtype=bool)
        mask[idx] = True
        pop.append(mask)
    return pop


def tournament_select(pop: List[np.ndarray], fitness: List[float], k: int) -> np.ndarray:
    ids = random.sample(range(len(pop)), k)
    best = min(ids, key=lambda i: fitness[i])  # lower MAE is better
    return pop[best].copy()


def crossover(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # uniform crossover
    m = np.random.rand(a.size) < 0.5
    c1 = a.copy()
    c2 = b.copy()
    c1[m] = b[m]
    c2[m] = a[m]
    return c1, c2


def mutate(mask: np.ndarray, cfg: GAConfig) -> np.ndarray:
    flip = np.random.rand(mask.size) < cfg.mutation_rate
    out = mask.copy()
    out[flip] = ~out[flip]
    return out


def repair(mask: np.ndarray, cfg: GAConfig) -> np.ndarray:
    out = mask.copy()
    k = int(out.sum())

    # too few -> add random features
    if k < cfg.min_features:
        zeros = np.where(~out)[0]
        add = np.random.choice(zeros, size=(cfg.min_features - k), replace=False)
        out[add] = True

    # too many -> drop random features
    if cfg.max_features is not None:
        k = int(out.sum())
        if k > cfg.max_features:
            ones = np.where(out)[0]
            drop = np.random.choice(ones, size=(k - cfg.max_features), replace=False)
            out[drop] = False

    return out


# -----------------------------
# Main GA loop
# -----------------------------
def run_ga_rf(csv_path: str | Path, out_dir: str | Path, cfg: GAConfig) -> None:
    set_seed(cfg.seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.rf_params is None:
        cfg.rf_params = dict(
            n_estimators=600,
            random_state=cfg.seed,
            n_jobs=-1,
            max_features="sqrt",
            min_samples_leaf=2,
        )

    X, y, groups = load_xy_groups(csv_path)
    feature_names = X.columns.to_list()
    n_features = X.shape[1]

    print(f"Loaded: rows={len(X)}, features={n_features}, participants={groups.nunique()}")

    pop = init_population(n_features, cfg)

    # cache for speed (mask bytes -> mae)
    cache: Dict[bytes, float] = {}

    def eval_mask(mask: np.ndarray) -> float:
        key = mask.tobytes()
        if key in cache:
            return cache[key]
        mae = score_mask(X, y, groups, mask, cfg)
        cache[key] = mae
        return mae

    fitness = [eval_mask(m) for m in pop]
    best_i = int(np.argmin(fitness))
    best_mask = pop[best_i].copy()
    best_mae = float(fitness[best_i])

    print(f"Gen 0 best MAE: {best_mae:.4f} with {best_mask.sum()} features")

    history = []

    for gen in range(1, cfg.generations + 1):
        # elitism
        elite_idx = np.argsort(fitness)[: cfg.elitism]
        new_pop = [pop[i].copy() for i in elite_idx]

        # breed
        while len(new_pop) < cfg.pop_size:
            p1 = tournament_select(pop, fitness, cfg.tournament_k)
            p2 = tournament_select(pop, fitness, cfg.tournament_k)

            if random.random() < cfg.crossover_rate:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = p1, p2

            c1 = repair(mutate(c1, cfg), cfg)
            c2 = repair(mutate(c2, cfg), cfg)

            new_pop.append(c1)
            if len(new_pop) < cfg.pop_size:
                new_pop.append(c2)

        pop = new_pop
        fitness = [eval_mask(m) for m in pop]

        gen_best_i = int(np.argmin(fitness))
        gen_best_mae = float(fitness[gen_best_i])
        gen_best_mask = pop[gen_best_i].copy()

        if gen_best_mae < best_mae:
            best_mae = gen_best_mae
            best_mask = gen_best_mask.copy()

        history.append((gen, gen_best_mae, int(gen_best_mask.sum()), best_mae, int(best_mask.sum())))

        elapsed = time.time() - start_time
        avg_per_gen = elapsed / gen
        remaining = avg_per_gen * (cfg.generations - gen)

        if gen % 5 == 0 or gen == cfg.generations:
            print(
                f"Gen {gen:02d} | "
                f"best(gen)={gen_best_mae:.4f} k={gen_best_mask.sum():02d} | "
                f"best(ever)={best_mae:.4f} | "
                f"elapsed={elapsed/60:.1f} min | "
                f"ETA={remaining/60:.1f} min"
)

    # Save outputs
    selected = [feature_names[i] for i, on in enumerate(best_mask) if on]
    out_features = pd.DataFrame({"feature": selected})
    out_features.to_csv(out_dir / "ga_rf_selected_features.csv", index=False)

    hist_df = pd.DataFrame(history, columns=["gen", "gen_best_mae", "gen_best_k", "best_mae", "best_k"])
    hist_df.to_csv(out_dir / "ga_rf_history.csv", index=False)

    # quick modality breakdown
    def mod(f: str) -> str:
        if f.startswith("imu_"):
            return "imu"
        if f.startswith("shim_"):
            return "shimmer"
        if f.startswith("eye_"):
            return "eye"
        return "other"

    mod_counts = pd.Series([mod(f) for f in selected]).value_counts().reset_index()
    mod_counts.columns = ["modality", "count"]
    mod_counts.to_csv(out_dir / "ga_rf_modality_counts.csv", index=False)

    print("\n=== DONE ===")
    print(f"Best MAE (GroupKFold by participant): {best_mae:.4f}")
    print(f"Selected {len(selected)} features")
    print(f"Saved: {out_dir/'ga_rf_selected_features.csv'}")
    print(f"Saved: {out_dir/'ga_rf_history.csv'}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("csv_path", help="Path to fixed full dataset CSV")
    p.add_argument("--out_dir", default="outputs/ga_rf", help="Output folder")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = GAConfig(seed=args.seed)
    run_ga_rf(args.csv_path, args.out_dir, cfg)