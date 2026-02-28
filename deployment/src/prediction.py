#prediciton.py 
#contain: model loading, BONI_COLS schema, bonismatchlist_df template, feature engineering, add_user_row, run_inference.

from __future__ import annotations

from pathlib import Path
import streamlit as st
import warnings
import numpy as np
import pandas as pd
import joblib
import sys

#Ensure src/ is on sys.path so joblib can import lol_features
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

#to surpress lbgm
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning
)

#for new names not in encoder
warnings.filterwarnings(
    "ignore",
    message="unknown class(es).*will be ignored",
    category=UserWarning
)

#path
#prediction.py is in: deployment/src/prediction.py
#models are in:      deployment/models/*.joblib
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


@st.cache_resource
def load_models():
    """Load models once (cached) to avoid reloading on every call."""
    pipe_cal_lgb_m = joblib.load(MODEL_DIR / "pipe_cal_lgb_m.joblib")
    pipe_best_lr_m = joblib.load(MODEL_DIR / "pipe_best_lr_m.joblib")
    pipe_best_lgb  = joblib.load(MODEL_DIR / "pipe_best_lgb.joblib")
    pipe_best_lr   = joblib.load(MODEL_DIR / "pipe_best_lr.joblib")
    return pipe_best_lgb, pipe_best_lr, pipe_cal_lgb_m, pipe_best_lr_m


#regrister users stats
BONI_COLS = [
    "gameid","playoffs","league","date","game","patch","participantid","side","teamname","teamid",
    "firstPick","ban1","ban2","ban3","ban4","ban5",
    "pick1","pick2","pick3","pick4","pick5",
    "firstdragon",
    "goldat10","xpat10","csat10","killsat10","assistsat10","deathsat10",
    "opp_goldat10","opp_xpat10","opp_csat10","opp_killsat10","opp_assistsat10","opp_deathsat10",
    "result"
]

#default empty registry
bonismatchlist_df = pd.DataFrame(columns=BONI_COLS)

def add_user_row(user_dict: dict, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Append a user row to df (or global bonismatchlist_df if df is None).
    Missing keys are filled with np.nan.
    """
    global bonismatchlist_df

    if df is None:
        df = bonismatchlist_df

    #ensure all columns exist
    for c in BONI_COLS:
        if c not in df.columns:
            df[c] = np.nan

    row = {c: user_dict.get(c, np.nan) for c in BONI_COLS}
    df.loc[len(df)] = row

    if df is bonismatchlist_df:
        bonismatchlist_df = df

    return df


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Make sure df has all cols; if missing, create as NaN."""
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


def _result_display(x) -> str:
    if pd.isna(x):
        return "NYA"
    try:
        return "Win" if int(x) == 1 else "Lose"
    except Exception:
        return "NYA"


def run_inference(*args):
    """
    Supported:
      - run_inference(gameid, teamname)  # uses global bonismatchlist_df
      - run_inference(df, gameid, teamname)

    Returns dict:
    {
      gameid, teamname,
      winning_i_lgbm, winning_i_logreg,
      winning_m_lgbm, winning_m_logreg,
      result
    }
    """
    global bonismatchlist_df

    #parse args
    if len(args) == 2:
        df_all = bonismatchlist_df
        gameid, teamname = args
    elif len(args) == 3:
        df_all, gameid, teamname = args
    else:
        raise TypeError("Use run_inference(gameid, teamname) or run_inference(df, gameid, teamname)")

    if df_all is None or len(df_all) == 0:
        raise ValueError("Registry dataframe is empty. Register a row first.")

    #lookup row
    df_row = df_all[(df_all["gameid"] == gameid) & (df_all["teamname"] == teamname)].copy()
    if df_row.empty:
        raise ValueError(f"No match found for gameid={gameid}, teamname={teamname}")

    #if multiple, take first
    df_row = df_row.iloc[[0]].reset_index(drop=True)

    #engineered features (matchup)
    #fillna(0) for numeric diffs so you can still score partial inputs
    df_row["golddiffat10"] = df_row["goldat10"].fillna(0) - df_row["opp_goldat10"].fillna(0)
    df_row["xpdiffat10"]   = df_row["xpat10"].fillna(0)  - df_row["opp_xpat10"].fillna(0)
    df_row["csdiffat10"]   = df_row["csat10"].fillna(0)  - df_row["opp_csat10"].fillna(0)

    df_row["killsdiffat10"]   = df_row["killsat10"].fillna(0)   - df_row["opp_killsat10"].fillna(0)
    df_row["assistsdiffat10"] = df_row["assistsat10"].fillna(0) - df_row["opp_assistsat10"].fillna(0)

    df_row["gold_per_kill_team"] = df_row["goldat10"].fillna(0) / (df_row["killsat10"].fillna(0) + 1)
    df_row["gold_per_kill_opp"]  = df_row["opp_goldat10"].fillna(0) / (df_row["opp_killsat10"].fillna(0) + 1)
    df_row["gpkdiffat10"]        = df_row["gold_per_kill_team"] - df_row["gold_per_kill_opp"]

    #firstkill fix
    if "firstkill" not in df_row.columns:
        df_row["firstkill"] = 0
    df_row.loc[
        (df_row["killsat10"].fillna(0) == 0) & (df_row["opp_killsat10"].fillna(0) == 0),
        "firstkill"
    ] = 0

    #ensure firstdragon exists
    if "firstdragon" not in df_row.columns:
        df_row["firstdragon"] = 0
    df_row["firstdragon"] = df_row["firstdragon"].fillna(0)

    #columns used by the models
    intr_cols = [
        "gameid","participantid","date","game","teamname",
        "side","patch","playoffs","firstPick",
        "ban1","ban2","ban3","ban4","ban5",
        "pick1","pick2","pick3","pick4","pick5",
        "goldat10","xpat10","csat10","killsat10","assistsat10","deathsat10"
    ]

    match_cols = [
        "gameid","participantid","date","game","teamname",
        "side","league","patch","playoffs","firstPick",
        "ban1","ban2","ban3","ban4","ban5",
        "pick1","pick2","pick3","pick4","pick5",
        "firstdragon",
        "golddiffat10","xpdiffat10","csdiffat10",
        "killsdiffat10","assistsdiffat10","gpkdiffat10",
        "firstkill"
    ]

    df_row = _ensure_cols(df_row, intr_cols + match_cols)

    X_intr = df_row[intr_cols].copy()
    X_match = df_row[match_cols].copy()

    #load models
    pipe_best_lgb, pipe_best_lr, pipe_cal_lgb_m, pipe_best_lr_m = load_models()

    #predict probabilities
    p_intr_lgb  = float(pipe_best_lgb.predict_proba(X_intr)[:, 1][0])
    p_intr_lr   = float(pipe_best_lr.predict_proba(X_intr)[:, 1][0])
    p_match_lgb = float(pipe_cal_lgb_m.predict_proba(X_match)[:, 1][0])
    p_match_lr  = float(pipe_best_lr_m.predict_proba(X_match)[:, 1][0])

    #result display
    result_display = _result_display(df_row.loc[0, "result"] if "result" in df_row.columns else np.nan)

    return {
        "gameid": gameid,
        "teamname": teamname,
        "winning_i_lgbm": round(p_intr_lgb, 4),
        "winning_i_logreg": round(p_intr_lr, 4),
        "winning_m_lgbm": round(p_match_lgb, 4),
        "winning_m_logreg": round(p_match_lr, 4),
        "result": result_display
    }