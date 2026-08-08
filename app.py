import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from build_dataset import build_final_dataset
from predict import train_full_model, predict_match


st.set_page_config(page_title="Football Match Predictor", layout="centered")

# cosmetic only - display nicer names in the UI without touching underlying data
DISPLAY_NAMES = {
    'Ath Madrid': 'Atletico Madrid',
    'Ath Bilbao': 'Athletic Bilbao',
    "M'gladbach": 'Borussia Monchengladbach',
    'FC Koln': 'FC Koln',
    'Sociedad': 'Real Sociedad',
    'Vallecano': 'Rayo Vallecano',
    'Espanol': 'Espanyol',
    'Paris SG': 'Paris Saint-Germain',
    "Nott'm Forest": 'Nottingham Forest',
    'Sheffield United': 'Sheffield United',
    'West Brom': 'West Bromwich Albion',
}

def display_name(team):
    return DISPLAY_NAMES.get(team, team)


@st.cache_data
def load_data():
    return build_final_dataset()

@st.cache_resource
def get_model(_data, league):
    return train_full_model(_data, league=league)


data, ratings = load_data()

st.title("Football Match Predictor")

leagues = sorted(data['League'].unique())
league = st.selectbox("League", leagues)

league_data = data[data['League'] == league]
teams = sorted(set(league_data['HomeTeam']) | set(league_data['AwayTeam']))
team_display = {display_name(t): t for t in teams}
display_list = sorted(team_display.keys())

model, scaler = get_model(data, league)

col1, col2 = st.columns(2)
with col1:
    home_display = st.selectbox("Home team", display_list, index=0)
with col2:
    away_display = st.selectbox("Away team", display_list, index=1)

home_team = team_display[home_display]
away_team = team_display[away_display]

if st.button("Predict"):
    if home_team == away_team:
        st.warning("Pick two different teams.")
    else:
        result = predict_match(model, scaler, data, home_team, away_team, league)
        if result:
            labels = {
                'H': f'{home_display} win',
                'A': f'{away_display} win',
                'D': 'Draw'
            }
            probs_df = pd.DataFrame({
                'Outcome': [labels[k] for k in result.keys()],
                'Probability': [v for v in result.values()]
            }).sort_values('Probability', ascending=False)

            st.bar_chart(probs_df.set_index('Outcome'))
            for _, row in probs_df.iterrows():
                st.write(f"**{row['Outcome']}**: {row['Probability']:.1%}")

st.divider()
st.subheader(f"Current Elo Ratings - {league}")

league_ratings = {k[1]: v for k, v in ratings.items() if k[0] == league}
ratings_df = pd.DataFrame(
    sorted(league_ratings.items(), key=lambda x: -x[1]),
    columns=['Team', 'Elo']
)
ratings_df['Team'] = ratings_df['Team'].apply(display_name)
ratings_df['Elo'] = ratings_df['Elo'].round(1)
ratings_df.index = ratings_df.index + 1

st.dataframe(ratings_df, use_container_width=True)