# deployment/src/prediction.py
# Stable, terminal-safe inference for LoL win probability models (PIPELINE-FRIENDLY)

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import joblib

import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names"
)

# ============================================================
# 0) PATH BOOTSTRAP (critical for joblib unpickle)
# ============================================================
SRC_DIR = Path(__file__).resolve().parent          # .../deployment/src
DEPLOYMENT_DIR = SRC_DIR.parent                    # .../deployment
REPO_ROOT = DEPLOYMENT_DIR.parent                  # .../<repo root>

for p in (str(REPO_ROOT), str(DEPLOYMENT_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


# ============================================================
# 1) OPTIONAL Streamlit caching (disable with LOL_DISABLE_CACHE=1)
# ============================================================
DISABLE_CACHE = os.getenv("LOL_DISABLE_CACHE", "0") == "1"

try:
    import streamlit as st  # type: ignore
except Exception:
    st = None

if st is not None and not DISABLE_CACHE:
    _cache_resource = st.cache_resource
else:
    def _cache_resource(func):
        return func


# ============================================================
# 2) MODEL PATHS
# ============================================================
MODEL_DIR = DEPLOYMENT_DIR / "models"

MODEL_FILES = {
    "intrinsic_lgbm": "intrinsic_lgbm.joblib",
    "match_logreg": "match_logreg.joblib",
    "match_lgbm_cal": "match_lgbm_cal.joblib",
}


# ============================================================
# 3) RAW COLS REQUIRED BY LolPOVFeatures
#    (Your LolPOVFeatures builds patch_major from patch, so patch MUST be present)
# ============================================================
RAW_REQUIRED_COLS = ["patch"]


# ============================================================
# 4) HELPERS
# ============================================================
def _unique_preserve_order(cols):
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _align_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Ensure df has ALL cols (add missing as NaN), then return df[cols] in stable order.
    This prevents schema drift + pipeline KeyErrors.
    """
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]


def _safe_proba(pipe, X: pd.DataFrame) -> float:
    """Return P(class=1) if predict_proba exists; else return predict."""
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)
        return float(proba[0, 1])
    pred = pipe.predict(X)
    return float(pred[0])


def _to_int01(x, default=0) -> int:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return int(default)
    if isinstance(x, (bool, np.bool_)):
        return int(x)
    if isinstance(x, (int, np.integer)):
        return int(1 if x != 0 else 0)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"1", "true", "t", "yes", "y", "win"}:
            return 1
        if s in {"0", "false", "f", "no", "n", "lose", "loss"}:
            return 0
        try:
            return int(float(s) != 0.0)
        except Exception:
            return int(default)
    try:
        return int(float(x) != 0.0)
    except Exception:
        return int(default)


# ============================================================
# 5) LOAD MODELS (cached if streamlit active and cache enabled)
# ============================================================
@_cache_resource
def load_models() -> Tuple[Any, Any, Any]:
    paths = {k: (MODEL_DIR / v) for k, v in MODEL_FILES.items()}

    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model files:\n" + "\n".join(missing) +
            f"\n\nExpected MODEL_DIR: {MODEL_DIR}"
        )

    intrinsic_pipe = joblib.load(paths["intrinsic_lgbm"])
    match_logreg_pipe = joblib.load(paths["match_logreg"])
    match_lgbm_cal_pipe = joblib.load(paths["match_lgbm_cal"])
    return intrinsic_pipe, match_logreg_pipe, match_lgbm_cal_pipe


def derive_feature_schema(pipe) -> list[str]:
    """
    Derive the expected input schema from the pipeline's prep step,
    PLUS any raw columns needed inside LolPOVFeatures.transform().
    """
    prep = pipe.named_steps.get("prep", None)
    if prep is None:
        raise AttributeError("Pipeline has no 'prep' step; cannot derive schema.")

    cols = []
    for attr in ("role_cols", "ban_cols", "cat_cols", "num_cols"):
        part = getattr(prep, attr, None)
        if part is None:
            continue
        cols.extend(list(part))

    cols = _unique_preserve_order(cols)

    # Ensure raw-required inputs exist (e.g. patch for patch_major engineering)
    for c in RAW_REQUIRED_COLS:
        if c not in cols:
            cols.append(c)

    if not cols:
        raise AttributeError("Could not derive schema from prep.*_cols (empty).")

    return cols


# ============================================================
# 6) BONI_COLS / USER ROW
# ============================================================
BONI_COLS = [
    "side", "firstPick",
    "ban1", "ban2", "ban3", "ban4", "ban5",
    "pick1", "pick2", "pick3", "pick4", "pick5",
    "league", "patch", "playoffs",

    # raw early10 (intrinsic)
    "goldat10", "xpat10", "csat10", "killsat10", "assistsat10",

    # opponent early10 (for match diffs)
    "opp_goldat10", "opp_xpat10", "opp_csat10", "opp_killsat10", "opp_assistsat10",

    # extra
    "firstdragon",

    # optional label/display
    "result",
]


def add_user_row(user_inputs: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a 1-row DataFrame with BONI_COLS schema.
    Missing keys become NaN.
    """
    row = {c: user_inputs.get(c, np.nan) for c in BONI_COLS}
    df = pd.DataFrame([row])

    # normalize common binary fields
    df.loc[0, "playoffs"] = _to_int01(df.loc[0, "playoffs"], default=0)
    df.loc[0, "firstPick"] = _to_int01(df.loc[0, "firstPick"], default=0)
    df.loc[0, "firstdragon"] = _to_int01(df.loc[0, "firstdragon"], default=0)

    # Keep patch as string-ish (LolPOVFeatures will parse it)
    if "patch" in df.columns and not pd.isna(df.loc[0, "patch"]):
        df.loc[0, "patch"] = str(df.loc[0, "patch"])

    return df


# ============================================================
# 7) MATCH FEATURE BUILD (diff + derived)
# ============================================================
def build_match_features(df_row: pd.DataFrame) -> pd.DataFrame:
    """
    Compute matchup diff features from raw team vs opp stats.
    Returns df with original columns + diff columns.
    """
    df = df_row.copy()

    # numeric conversions for diff inputs
    need_num = [
        "goldat10", "xpat10", "csat10", "killsat10", "assistsat10",
        "opp_goldat10", "opp_xpat10", "opp_csat10", "opp_killsat10", "opp_assistsat10",
    ]
    for c in need_num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # diffs
    df["golddiffat10"] = df["goldat10"] - df["opp_goldat10"]
    df["xpdiffat10"] = df["xpat10"] - df["opp_xpat10"]
    df["csdiffat10"] = df["csat10"] - df["opp_csat10"]
    df["killsdiffat10"] = df["killsat10"] - df["opp_killsat10"]
    df["assistsdiffat10"] = df["assistsat10"] - df["opp_assistsat10"]

    # gpkdiffat10 (gold per kill diff), safe division
    gpk_team = df["goldat10"] / df["killsat10"].replace(0, np.nan)
    gpk_opp = df["opp_goldat10"] / df["opp_killsat10"].replace(0, np.nan)
    df["gpkdiffat10"] = (gpk_team - gpk_opp).fillna(0.0)

    # firstkill heuristic
    k = df["killsat10"].fillna(0)
    ok = df["opp_killsat10"].fillna(0)
    df["firstkill"] = np.nan
    df.loc[(k > 0) & (ok == 0), "firstkill"] = 1
    df.loc[(k == 0) & (ok > 0), "firstkill"] = 0
    df["firstkill"] = df["firstkill"].fillna(0).astype(int)

    # ensure firstdragon is 0/1 int
    df["firstdragon"] = df["firstdragon"].apply(lambda v: _to_int01(v, default=0)).astype(int)

    return df


# ============================================================
# 8) MAIN INFERENCE
# ============================================================
def run_inference(user_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns:
      - p_win_intrinsic_lgbm
      - p_win_match_logreg
      - p_win_match_lgbm_cal
      - result ("Win"/"Lose"/"NYA")
    """
    intrinsic_pipe, match_logreg_pipe, match_lgbm_cal_pipe = load_models()

    # Build row (raw inputs)
    df_row = add_user_row(user_inputs)

    # Build match features
    df_match = build_match_features(df_row)

    # Drop engineered cols if user accidentally passed them (safe drop)
    for bad in ("patch_major",):
        df_row = df_row.drop(columns=[bad], errors="ignore")
        df_match = df_match.drop(columns=[bad], errors="ignore")

    # Derive schemas from pipelines (+ RAW_REQUIRED_COLS)
    intr_cols = derive_feature_schema(intrinsic_pipe)
    match_cols = derive_feature_schema(match_logreg_pipe)

    # Align inputs to schemas
    X_intr = _align_columns(df_row, intr_cols)
    X_match = _align_columns(df_match, match_cols)

    # Predict probabilities
    p_i = _safe_proba(intrinsic_pipe, X_intr)
    p_m_lr = _safe_proba(match_logreg_pipe, X_match)
    p_m_lgbm_cal = _safe_proba(match_lgbm_cal_pipe, X_match)

    # Optional actual label for display
    y = user_inputs.get("result", None)
    if y is None or (isinstance(y, float) and np.isnan(y)) or str(y).strip() == "":
        actual = "NYA"
    else:
        actual = "Win" if _to_int01(y, default=0) == 1 else "Lose"

    return {
        "p_win_intrinsic_lgbm": p_i,
        "p_win_match_logreg": p_m_lr,
        "p_win_match_lgbm_cal": p_m_lgbm_cal,
        "result": actual,
    }


# ============================================================
# 9) CLI TEST (optional)
# ============================================================
if __name__ == "__main__":
    sample = {
        "side": "Blue",
        "firstPick": 1,
        "ban1": "Aatrox", "ban2": "Sejuani", "ban3": "Kalista", "ban4": "Vi", "ban5": "Orianna",
        "pick1": "Renekton", "pick2": "LeeSin", "pick3": "Ahri", "pick4": "Jinx", "pick5": "Thresh",
        "league": "LCK",
        "patch": "15.1",
        "playoffs": 0,
        "goldat10": 18500, "xpat10": 20500, "csat10": 360, "killsat10": 2, "assistsat10": 4,
        "opp_goldat10": 17800, "opp_xpat10": 19800, "opp_csat10": 345, "opp_killsat10": 0, "opp_assistsat10": 1,
        "firstdragon": 1,
        "result": "",
    }
    print(run_inference(sample))