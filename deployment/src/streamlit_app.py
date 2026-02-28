#import streamlit, prediction, st.session_state registry, UI, calling prediction run interface. 

import streamlit as st
import prediction


st.set_page_config(
    page_title="LoL Win Probability",
    layout="wide"
)

st.title("LoL Esports Win Probability")
st.caption("Intrinsic vs Matchup Model Comparison (10-min state)")

#sesh regis
if "registry" not in st.session_state:
    st.session_state.registry = prediction.bonismatchlist_df.copy()

#registeam row
st.header("Register Team Row")

with st.form("register_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gameid = st.text_input("Game ID")
        teamname = st.text_input("Team Name")
        league = st.text_input("League")
        patch = st.text_input("Patch")

    with col2:
        side = st.selectbox("Side", ["Blue", "Red"])
        playoffs = st.selectbox("Playoffs", [0, 1])
        firstPick = st.selectbox("First Pick", [0, 1])
        firstdragon = st.selectbox("First Dragon", [0, 1])

    with col3:
        goldat10 = st.number_input("Gold @10", value=0)
        xpat10 = st.number_input("XP @10", value=0)
        csat10 = st.number_input("CS @10", value=0)
        killsat10 = st.number_input("Kills @10", value=0)
        assistsat10 = st.number_input("Assists @10", value=0)
        deathsat10 = st.number_input("Deaths @10", value=0)

    st.subheader("Opponent @10")

    col4, col5 = st.columns(2)

    with col4:
        opp_goldat10 = st.number_input("Opp Gold @10", value=0)
        opp_xpat10 = st.number_input("Opp XP @10", value=0)
        opp_csat10 = st.number_input("Opp CS @10", value=0)

    with col5:
        opp_killsat10 = st.number_input("Opp Kills @10", value=0)
        opp_assistsat10 = st.number_input("Opp Assists @10", value=0)
        opp_deathsat10 = st.number_input("Opp Deaths @10", value=0)

    submitted = st.form_submit_button("Register")

    if submitted:
        user_dict = {
            "gameid": gameid,
            "teamname": teamname,
            "league": league,
            "patch": patch,
            "side": side,
            "playoffs": playoffs,
            "firstPick": firstPick,
            "firstdragon": firstdragon,
            "goldat10": goldat10,
            "xpat10": xpat10,
            "csat10": csat10,
            "killsat10": killsat10,
            "assistsat10": assistsat10,
            "deathsat10": deathsat10,
            "opp_goldat10": opp_goldat10,
            "opp_xpat10": opp_xpat10,
            "opp_csat10": opp_csat10,
            "opp_killsat10": opp_killsat10,
            "opp_assistsat10": opp_assistsat10,
            "opp_deathsat10": opp_deathsat10,
        }

        st.session_state.registry = prediction.add_user_row(
            user_dict,
            st.session_state.registry
        )

        st.success("Row Registered")

#inference
st.header("Run Inference")

if len(st.session_state.registry) > 0:

    #valid dropdown entries.
    df_valid = st.session_state.registry.dropna(subset=["gameid", "teamname"])
    df_valid = df_valid[
        (df_valid["gameid"] != "") & 
        (df_valid["teamname"] != "")
    ]

    if len(df_valid) == 0:
        st.info("No valid registered matches yet.")
    else:
        game_options = df_valid["gameid"].unique()
        selected_game = st.selectbox("Select Game ID", game_options)

    team_options = (
    df_valid[
        df_valid["gameid"] == selected_game
    ]["teamname"]
    .unique()
    )

    selected_team = st.selectbox("Select Team", team_options)

    if st.button("Run Prediction"):

        result = prediction.run_inference(
            st.session_state.registry,
            selected_game,
            selected_team
        )

        st.subheader("Predictions")

        colA, colB = st.columns(2)

        with colA:
            st.metric("Intrinsic LGBM", result["winning_i_lgbm"])
            st.metric("Intrinsic LogReg", result["winning_i_logreg"])

        with colB:
            st.metric("Matchup LGBM", result["winning_m_lgbm"])
            st.metric("Matchup LogReg", result["winning_m_logreg"])

        st.write("Result:", result["result"])

else:
    st.info("No registered matches yet.")

#show registry
st.header("Registry")
st.dataframe(st.session_state.registry)