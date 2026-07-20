import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
import hashlib
from credit_default.eda.static_features import TARGET_COL
import json

MANIFEST_PATH = Path("reports/preprocessing/split_manifest.csv")
LOCK_PATH = Path("artifacts/preprocessing/split_lock.json")
SPLIT_SEED = 42
VAL_SEED = 43

def compute_stable_group_id(row: pd.Series, cols: list) -> str:
    """Compute a stable hash based on sorted predictor columns."""
    # Convert all to float/int strings consistently
    vals = [str(row[c]) for c in cols]
    content = "|".join(vals).encode("utf-8")
    return hashlib.md5(content).hexdigest()

def build_duplicate_groups(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Build predictor-group IDs strictly excluding ID and target.
    Uses a stable hash of the 23 raw predictors.
    """
    exclude_cols = ["ID", TARGET_COL]
    predictor_cols = sorted([c for c in df.columns if c not in exclude_cols])
    
    df_mapped = df.copy()
    
    # Compute stable ID
    # To speed this up, we can drop duplicates on the predictors first
    unique_rows = df_mapped[predictor_cols].drop_duplicates()
    unique_rows["predictor_group_id"] = unique_rows.apply(lambda r: compute_stable_group_id(r, predictor_cols), axis=1)
    
    df_mapped = pd.merge(df_mapped, unique_rows, on=predictor_cols, how="left")
    
    # Audit info
    group_sizes = df_mapped["predictor_group_id"].value_counts()
    unique_groups = len(group_sizes)
    repeated_groups = (group_sizes > 1).sum()
    rows_in_repeated = group_sizes[group_sizes > 1].sum()
    
    # Conflicting target labels in same group
    target_nunique = df_mapped.groupby("predictor_group_id")[TARGET_COL].nunique()
    conflicting_groups = (target_nunique > 1).sum()
    rows_in_conflicting = df_mapped[df_mapped["predictor_group_id"].isin(target_nunique[target_nunique > 1].index)].shape[0]
    
    max_group_size = group_sizes.max()
    
    audit_df = pd.DataFrame([{
        "total_rows": len(df),
        "unique_predictor_groups": unique_groups,
        "repeated_predictor_groups": repeated_groups,
        "rows_in_repeated_predictor_groups": rows_in_repeated,
        "conflicting_target_predictor_groups": conflicting_groups,
        "rows_in_conflicting_target_groups": rows_in_conflicting,
        "maximum_predictor_group_size": max_group_size
    }])
    
    return df_mapped, audit_df

def generate_split(df_mapped: pd.DataFrame, seed: int = SPLIT_SEED, val_seed: int = VAL_SEED) -> pd.DataFrame:
    """
    Generates deterministic leakage-safe train/val/test splits (64/16/20).
    Rows are sorted canonically before splitting to ensure row-order independence.
    """
    df_split = df_mapped.copy()
    
    # Canonical sort to make StratifiedGroupKFold completely row-order independent
    df_split = df_split.sort_values(["predictor_group_id", "ID"]).reset_index(drop=True)
    df_split["split"] = "unassigned"
    
    X = df_split["ID"].values
    y = df_split[TARGET_COL].values
    groups = df_split["predictor_group_id"].values
    
    # Step 1: 5-fold split to isolate 20% TEST
    sgkf_test = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    
    for dev_idx, test_idx in sgkf_test.split(X, y, groups):
        df_split.iloc[test_idx, df_split.columns.get_loc("split")] = "test"
        
        # The remaining is dev (train + val)
        X_dev = X[dev_idx]
        y_dev = y[dev_idx]
        groups_dev = groups[dev_idx]
        
        # Step 2: 5-fold split on Dev to isolate 20% of dev (16% of total) as VAL
        sgkf_val = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=val_seed)
        for train_dev_idx, val_dev_idx in sgkf_val.split(X_dev, y_dev, groups_dev):
            true_train_idx = dev_idx[train_dev_idx]
            true_val_idx = dev_idx[val_dev_idx]
            
            df_split.iloc[true_val_idx, df_split.columns.get_loc("split")] = "validation"
            df_split.iloc[true_train_idx, df_split.columns.get_loc("split")] = "train"
            break # Need inner fold 0
            
        break # Need outer fold 0
        
    return df_split[["ID", "split", TARGET_COL, "predictor_group_id"]]

def get_file_sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def get_id_population_sha256(df: pd.DataFrame) -> str:
    sorted_ids = ",".join(df["ID"].astype(str).sort_values().values)
    return hashlib.sha256(sorted_ids.encode("utf-8")).hexdigest()

def load_or_create_split_manifest(df: pd.DataFrame, force_regenerate: bool = False, raw_sha256: str = "") -> (pd.DataFrame, pd.DataFrame, str):
    """
    Loads existing split manifest or creates it if not exists or forced.
    Enforces the split lock securely during normal reuse.
    Returns: manifest_df, audit_df, manifest_sha256
    """
    df_mapped, audit_df = build_duplicate_groups(df)
    
    if MANIFEST_PATH.exists() and LOCK_PATH.exists() and not force_regenerate:
        with open(LOCK_PATH, "r") as f:
            lock_data = json.load(f)
            
        if lock_data.get("raw_sha256") != raw_sha256:
            raise ValueError("Lock mismatch: raw dataset SHA-256 differs.")
            
        manifest_sha_disk = get_file_sha256(MANIFEST_PATH)
        if lock_data.get("split_manifest_sha256") != manifest_sha_disk:
            raise ValueError("Lock mismatch: manifest file SHA-256 differs from lock.")
            
        manifest = pd.read_csv(MANIFEST_PATH)
        
        # Verify schema exact
        req_cols = ["ID", "split", TARGET_COL, "predictor_group_id"]
        if list(manifest.columns) != req_cols:
            raise ValueError(f"Manifest schema mismatch. Expected {req_cols}, got {list(manifest.columns)}")
            
        # Unique IDs
        if len(manifest["ID"].unique()) != len(manifest):
            raise ValueError("Manifest IDs are not unique.")
            
        # ID Population match
        if len(manifest) != len(df) or set(manifest["ID"]) != set(df["ID"]):
            raise ValueError("Manifest ID population does not match raw data.")
            
        id_pop_sha = get_id_population_sha256(manifest)
        if lock_data.get("id_population_sha256") != id_pop_sha:
            raise ValueError("Lock mismatch: ID population SHA differs.")
            
        # Targets match raw
        df_target_check = pd.merge(manifest[["ID", TARGET_COL]], df[["ID", TARGET_COL]], on="ID", suffixes=('_man', '_raw'))
        if not (df_target_check[f"{TARGET_COL}_man"] == df_target_check[f"{TARGET_COL}_raw"]).all():
            raise ValueError("Manifest target labels do not match raw target labels.")
            
        # Stable IDs match recomputed
        df_group_check = pd.merge(manifest[["ID", "predictor_group_id"]], df_mapped[["ID", "predictor_group_id"]], on="ID", suffixes=('_man', '_raw'))
        if not (df_group_check["predictor_group_id_man"] == df_group_check["predictor_group_id_raw"]).all():
            raise ValueError("Manifest stable group IDs do not match dynamically computed group IDs.")
            
        # Valid splits
        if not manifest["split"].isin(["train", "validation", "test"]).all():
            raise ValueError("Manifest contains invalid split labels.")
            
        # Group overlap
        tr_g = set(manifest[manifest["split"]=="train"]["predictor_group_id"])
        va_g = set(manifest[manifest["split"]=="validation"]["predictor_group_id"])
        te_g = set(manifest[manifest["split"]=="test"]["predictor_group_id"])
        if len(tr_g & va_g) > 0 or len(tr_g & te_g) > 0 or len(va_g & te_g) > 0:
            raise ValueError("Predictor group leakage detected across splits in manifest.")
            
        return manifest, audit_df, manifest_sha_disk
        
    elif (MANIFEST_PATH.exists() or LOCK_PATH.exists()) and not force_regenerate:
        raise ValueError("Partial lock/manifest detected. Refusing to silently regenerate without --regenerate-splits.")
            
    # Need to generate
    manifest = generate_split(df_mapped)
    
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = manifest.sort_values("ID").reset_index(drop=True)
    
    # We don't save here. The main script must save at the very end when all checks pass.
    # To support this, we return the manifest DataFrame directly.
    return manifest, audit_df, ""

def generate_split_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    records = []
    total = len(manifest)
    
    for s in ["train", "validation", "test"]:
        sub = manifest[manifest["split"] == s]
        cnt = len(sub)
        d_cnt = (sub[TARGET_COL] == 1).sum()
        nd_cnt = cnt - d_cnt
        u_groups = sub["predictor_group_id"].nunique()
        
        records.append({
            "split": s,
            "total_count": cnt,
            "non_default_count": nd_cnt,
            "default_count": d_cnt,
            "default_rate": (d_cnt / cnt) if cnt > 0 else 0.0,
            "population_percentage": (cnt / total * 100) if total > 0 else 0.0,
            "unique_predictor_groups": u_groups
        })
    return pd.DataFrame(records)
