# deployment/src/streamlit_app.py
from __future__ import annotations

import os
import json
import warnings
from typing import List

import pandas as pd
import streamlit as st

# Import prediction robustly (works when run from repo root)
try:
    from deployment.src import prediction
except Exception:
    import prediction  #fallback if running from deployment/src directly

# App config
st.set_page_config(page_title="LoL Win Probability", layout="wide")

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)

DISABLE_CACHE = os.getenv("LOL_DISABLE_CACHE", "0") == "1"

st.title("League of Legends E-Sports Win Probability")
st.caption("Draft vs Matchup model comparison (10-minute state)")


# -----------------------------------------------------------------------------
# Champion list (Data Dragon)
#   - versions: https://ddragon.leagueoflegends.com/api/versions.json
#   - champions: https://ddragon.leagueoflegends.com/cdn/<ver>/data/en_US/champion.json
# Riot documents champion.json as the champion list file. :contentReference[oaicite:2]{index=2}
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_champions() -> List[str]:
    """
    Returns a sorted list of champion display names (e.g., "Lee Sin", "K'Sante").
    Uses Data Dragon so you don't need to maintain a list manually.
    """
    # Use stdlib to avoid dependency issues
    from urllib.request import urlopen

    versions_url = "https://ddragon.leagueoflegends.com/api/versions.json"
    with urlopen(versions_url, timeout=10) as r:
        versions = json.loads(r.read().decode("utf-8"))
    latest = versions[0]

    champs_url = f"https://ddragon.leagueoflegends.com/cdn/{latest}/data/en_US/champion.json"
    with urlopen(champs_url, timeout=10) as r:
        data = json.loads(r.read().decode("utf-8"))

    champs = [v["name"] for v in data["data"].values()]
    champs = sorted(set(champs))
    return champs


def get_champion_options() -> List[str]:
    try:
        champs = load_champions()
        return ["—", *champs, "Other (type manually)"]
    except Exception:
        # Offline / blocked internet fallback
        return ["—", "Other (type manually)"]


CHAMPION_OPTIONS = get_champion_options()


# Session state init
REGISTRY_COLS = [
    "gameid", "teamname",
    "side", "firstPick",
    "league", "patch", "playoffs",
    "ban1", "ban2", "ban3", "ban4", "ban5",
    "pick1", "pick2", "pick3", "pick4", "pick5",
    "goldat10", "xpat10", "csat10", "killsat10", "assistsat10",
    "opp_goldat10", "opp_xpat10", "opp_csat10", "opp_killsat10", "opp_assistsat10",
    "firstdragon",
    "result",
]

if "registry_df" not in st.session_state:
    st.session_state.registry_df = pd.DataFrame(columns=REGISTRY_COLS)


# Sidebar controls
with st.sidebar:
    st.header("Controls")
    st.write("Run from repo root:")
    st.code("streamlit run deployment/src/streamlit_app.py")

    if st.button("Clear registry", use_container_width=True):
        st.session_state.registry_df = pd.DataFrame(columns=REGISTRY_COLS)
        st.success("Registry cleared.")

    st.divider()
    st.subheader("Debug")
    st.write("Cache:", "OFF" if DISABLE_CACHE else "ON")
    st.write("Champion options:", len(CHAMPION_OPTIONS))
    if st.checkbox("Show raw registry dataframe"):
        st.dataframe(st.session_state.registry_df, use_container_width=True)


# Helpers
def _append_row(df: pd.DataFrame, row: dict) -> pd.DataFrame:
    new_row = {c: row.get(c, pd.NA) for c in REGISTRY_COLS}
    return pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)


def _format_pct(x: float) -> str:
    try:
        return f"{100.0 * float(x):.1f}%"
    except Exception:
        return "—"


def _prob_bar(label: str, p: float):
    c1, c2 = st.columns([2, 8])
    with c1:
        st.write(label)
    with c2:
        st.progress(min(max(float(p), 0.0), 1.0))
        st.caption(_format_pct(p))


def champ_picker(label: str, key: str) -> str:
    """
    Dropdown + optional manual fallback.
    Returns a cleaned champion string or "".
    """
    choice = st.selectbox(label, CHAMPION_OPTIONS, index=0, key=f"{key}_sel")

    if choice == "—":
        return ""
    if choice == "Other (type manually)":
        manual = st.text_input(f"{label} (manual)", key=f"{key}_manual").strip()
        return manual
    return choice

#-----------------------------------------------------------------------
# Layout: Tabs
# --- Tab navigation state ---
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Registration"

tab_labels = ["Registration", "Prediction", "Team Registry"]

tab_register, tab_predict, tab_registry = st.tabs(
    ["➕ Registration", "🎯 Prediction", "📒 Team Registry"]
)

# TAB 1: Registration
with tab_register:
    st.subheader("Register Your Team")
    with st.form("register_form", clear_on_submit=False):
        top1, top2, top3 = st.columns(3)

        with top1:
            gameid = st.text_input("Game ID", placeholder="e.g., 2025_LCK_Week3_Game2")
            teamname = st.text_input("Team Name", placeholder="e.g., T1")
            league = st.text_input("League (optional)", placeholder="e.g., LCK")
            patch = st.text_input("Patch (optional)", placeholder="e.g., 15.1")

        with top2:
            side = st.selectbox("Side", ["Blue", "Red"])

            match_type = st.selectbox("Match Type", ["Regular Season", "Playoffs"], index=0)
            playoffs = 1 if match_type == "Playoffs" else 0

            firstPick_ui = st.selectbox("First Pick?", ["No", "Yes"], index=0)
            firstPick = 1 if firstPick_ui == "Yes" else 0

            firstdragon_ui = st.selectbox("First Dragon?", ["No", "Yes"], index=0)
            firstdragon = 1 if firstdragon_ui == "Yes" else 0

        with top3:
            result = st.selectbox("Actual Result (optional)", ["", "Win", "Lose"], index=0)

        st.divider()
        st.markdown("### Draft (optional but recommended)")
        d1, d2 = st.columns(2)
        with d1:
            ban1 = champ_picker("Ban 1", "ban1")
            ban2 = champ_picker("Ban 2", "ban2")
            ban3 = champ_picker("Ban 3", "ban3")
            ban4 = champ_picker("Ban 4", "ban4")
            ban5 = champ_picker("Ban 5", "ban5")

        with d2:
            pick1 = champ_picker("Pick 1", "pick1")
            pick2 = champ_picker("Pick 2", "pick2")
            pick3 = champ_picker("Pick 3", "pick3")
            pick4 = champ_picker("Pick 4", "pick4")
            pick5 = champ_picker("Pick 5", "pick5")

        st.divider()
        st.markdown("### Team Stats (10-minute mark)")
        st.caption("Enter your team’s stats at exactly 10:00.")
        cA, cB, cC, cD, cE = st.columns(5)
        with cA:
            goldat10 = st.number_input("Gold", value=0, step=100)
        with cB:
            xpat10 = st.number_input("XP", value=0, step=100)
        with cC:
            csat10 = st.number_input("CS", value=0, step=1)
        with cD:
            killsat10 = st.number_input("Kills", value=0, step=1)
        with cE:
            assistsat10 = st.number_input("Assists", value=0, step=1)

        st.markdown("### Opponent Stats (10-minute mark)")
        st.caption("Enter the opponent’s stats at exactly 10:00.")
        oA, oB, oC, oD, oE = st.columns(5)
        with oA:
            opp_goldat10 = st.number_input("Opp Gold", value=0, step=100)
        with oB:
            opp_xpat10 = st.number_input("Opp XP", value=0, step=100)
        with oC:
            opp_csat10 = st.number_input("Opp CS", value=0, step=1)
        with oD:
            opp_killsat10 = st.number_input("Opp Kills", value=0, step=1)
        with oE:
            opp_assistsat10 = st.number_input("Opp Assists", value=0, step=1)

        submitted = st.form_submit_button("Register This Team", use_container_width=True)

        if submitted:
            if not gameid.strip() or not teamname.strip():
                st.error("Game ID and Team Name are required.")
            else:
                # Safe defaults if blank (won't crash pipeline)
                league_value = league.strip() if league.strip() else "UNKNOWN"
                patch_value = patch.strip() if patch.strip() else "0.0"

                user_row = {
                    "gameid": gameid.strip(),
                    "teamname": teamname.strip(),
                    "league": league_value,
                    "patch": patch_value,
                    "side": side,
                    "playoffs": int(playoffs),
                    "firstPick": int(firstPick),
                    "firstdragon": int(firstdragon),

                    "ban1": ban1 or pd.NA,
                    "ban2": ban2 or pd.NA,
                    "ban3": ban3 or pd.NA,
                    "ban4": ban4 or pd.NA,
                    "ban5": ban5 or pd.NA,
                    "pick1": pick1 or pd.NA,
                    "pick2": pick2 or pd.NA,
                    "pick3": pick3 or pd.NA,
                    "pick4": pick4 or pd.NA,
                    "pick5": pick5 or pd.NA,

                    "goldat10": float(goldat10),
                    "xpat10": float(xpat10),
                    "csat10": float(csatat10) if "csatat10" in locals() else float(csat10),  # harmless
                    "killsat10": float(killsat10),
                    "assistsat10": float(assistsat10),

                    "opp_goldat10": float(opp_goldat10),
                    "opp_xpat10": float(opp_xpat10),
                    "opp_csat10": float(opp_csat10),
                    "opp_killsat10": float(opp_killsat10),
                    "opp_assistsat10": float(opp_assistsat10),

                    "result": 1 if result == "Win" else (0 if result == "Lose" else ""),
                }

                st.session_state.registry_df = _append_row(st.session_state.registry_df, user_row)
                st.success("Row registered ✅  Now click the **🎯 Prediction** tab to run win probability.")


# TAB 2: Prediction
with tab_predict:
    if st.session_state.active_tab != "Prediction":
        st.empty()
    else:
        # rest of the tab content
        st.subheader("Run Win Probability")

    df = st.session_state.registry_df.copy()
    df_valid = df.dropna(subset=["gameid", "teamname", "patch"])
    df_valid = df_valid[
        (df_valid["gameid"].astype(str).str.strip() != "")
        & (df_valid["teamname"].astype(str).str.strip() != "")
    ]

    if len(df_valid) == 0:
        st.info("No valid registered rows yet. Add one in **Registration**.")
    else:
        left, right = st.columns([3, 2])

        with left:
            game_options = sorted(df_valid["gameid"].unique().tolist())
            selected_game = st.selectbox("Select Game ID", game_options)

            team_options = sorted(df_valid[df_valid["gameid"] == selected_game]["teamname"].unique().tolist())
            selected_team = st.selectbox("Select Team", team_options)

        with right:
            st.markdown("#### Actions")
            run_btn = st.button("Run Win Probability", type="primary", use_container_width=True)

        if run_btn:
            row_df = df_valid[(df_valid["gameid"] == selected_game) & (df_valid["teamname"] == selected_team)].head(1)
            row_dict = row_df.iloc[0].to_dict()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = prediction.run_inference(row_dict)

            p_i = float(out["p_win_intrinsic_lgbm"])
            p_lr = float(out["p_win_match_logreg"])
            p_cal = float(out["p_win_match_lgbm_cal"])

            st.markdown("## Draft Strength")
            st.caption(
                "This team’s probability to win based on their **draft strength and execution**, "
                "including competitive advantage. Measured using LGBM model."
            )
            st.metric("Winning Probability", _format_pct(p_i))

            st.divider()

            st.markdown("## Matchup Prediction")
            st.caption(
                "This team's probability to win **in comparison** to the opponent's. Probability to win is updated to:"
            )
            # Make LogReg feel like the primary model visually
            colA, colB = st.columns([3, 1])
            with colA:
                st.markdown("### Match (Primary)")
                st.markdown(f"<div style='font-size:52px; font-weight:800; line-height:1'>{_format_pct(p_lr)}</div>", unsafe_allow_html=True)
                st.caption("logistic regression model")

            with colB:
                st.metric("Secondary", _format_pct(p_cal))
                st.caption("LGBM")

            st.divider()

            st.markdown("### Actual Result")
            res = out.get("result", "NYA")
            if res == "NYA":
                st.write("Not Yet Available")
            else:
                st.write(res)

            st.caption(
                "For completedtournament matches, this prediction view can be used to evaluate "
                "teams' performance during the previous matchup."
            )

            with st.expander("Show raw output dict"):
                st.json(out)

# TAB 3: Registry
with tab_registry:
    if st.session_state.active_tab != "Team Registry":
        st.empty()
    else:
        st.subheader("Registry")
    st.dataframe(st.session_state.registry_df, use_container_width=True)