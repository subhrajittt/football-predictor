import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from build_dataset import build_final_dataset
from predict import train_full_model, predict_match, FEATURE_COLS


st.set_page_config(page_title="PL Match Predictor", layout="centered")

@st.cache_data
def load_data():
    data, ratings = build_final_dataset()
    return data, ratings

@st.cache_resource
def get_model(data):
    return train_full_model(data)


data, ratings = load_data()
model, scaler = get_model(data)

teams = sorted(set(data['HomeTeam']) | set(data['AwayTeam']))

st.title("Premier League Match Predictor")

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Home team", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
with col2:
    away_team = st.selectbox("Away team", teams, index=teams.index("Chelsea") if "Chelsea" in teams else 1)

if st.button("Predict"):
    if home_team == away_team:
        st.warning("Pick two different teams.")
    else:
        result = predict_match(model, scaler, data, home_team, away_team)
        if result:
            labels = {'H': f'{home_team} win', 'A': f'{away_team} win', 'D': 'Draw'}
            probs_df = pd.DataFrame({
                'Outcome': [labels[k] for k in result.keys()],
                'Probability': [v for v in result.values()]
            }).sort_values('Probability', ascending=False)

            st.bar_chart(probs_df.set_index('Outcome'))
            for _, row in probs_df.iterrows():
                st.write(f"**{row['Outcome']}**: {row['Probability']:.1%}")

st.divider()
st.subheader("Current Elo Ratings")

ratings_df = pd.DataFrame(
    sorted(ratings.items(), key=lambda x: -x[1]),
    columns=['Team', 'Elo']
)
ratings_df['Elo'] = ratings_df['Elo'].round(1)
ratings_df.index = ratings_df.index + 1

st.dataframe(ratings_df, use_container_width=True)