import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from build_dataset import build_final_dataset
from predict import train_full_model, predict_match


st.set_page_config(page_title="Match Predictor", layout="centered", page_icon="⚽")

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
    'West Brom': 'West Bromwich Albion',
    'Ein Frankfurt': 'Eintracht Frankfurt',
}

LEAGUE_ACCENTS = {
    'Premier League': '#D4A72C',
    'La Liga': '#EE8707',
    'Serie A': '#008FD7',
    'Bundesliga': '#D3010C',
    'Ligue 1': '#DAE025',
}

BADGE_COLORS = {'W': '#3FA65C', 'D': '#8A8A80', 'L': '#C4453C'}


def display_name(team):
    return DISPLAY_NAMES.get(team, team)


def get_last_n_results(data, team, league, n=5):
    league_data = data[data['League'] == league].sort_values('Date')

    home_matches = league_data[league_data['HomeTeam'] == team][['Date', 'FTR']].copy()
    home_matches['result'] = home_matches['FTR'].map({'H': 'W', 'D': 'D', 'A': 'L'})

    away_matches = league_data[league_data['AwayTeam'] == team][['Date', 'FTR']].copy()
    away_matches['result'] = away_matches['FTR'].map({'H': 'L', 'D': 'D', 'A': 'W'})

    combined = pd.concat([home_matches[['Date', 'result']], away_matches[['Date', 'result']]])
    combined = combined.sort_values('Date').tail(n)

    return list(combined['result'])


def get_h2h_results(data, home_team, away_team, league, n=5):
    league_data = data[data['League'] == league].sort_values('Date')

    pair_matches = league_data[
        ((league_data['HomeTeam'] == home_team) & (league_data['AwayTeam'] == away_team)) |
        ((league_data['HomeTeam'] == away_team) & (league_data['AwayTeam'] == home_team))
    ].copy()

    def result_for_home_team(row):
        if row['FTR'] == 'D':
            return 'D'
        winner = row['HomeTeam'] if row['FTR'] == 'H' else row['AwayTeam']
        return 'W' if winner == home_team else 'L'

    pair_matches['result'] = pair_matches.apply(result_for_home_team, axis=1)
    pair_matches = pair_matches.sort_values('Date').tail(n)

    return list(pair_matches['result'])


def build_badge_row(label, results):
    if not results:
        badges_html = '<span style="font-size:0.8rem; color:#7A9285;">No matches yet</span>'
    else:
        badges_html = "".join(
            f'<div class="form-badge" style="background-color:{BADGE_COLORS[r]};">{r}</div>'
            for r in results
        )
        badges_html = f'<div class="form-badges">{badges_html}</div>'

    return (
        f'<div class="form-strip-wrap">'
        f'<span class="form-label">{label}</span>'
        f'{badges_html}'
        f'</div>'
    )


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #0B3D2E 0%, #0E4A38 100%);
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 3.5rem;
    max-width: 780px;
}

h1, h2, h3 {
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 0.03em;
    color: #F4F4EF !important;
}

p, span, label, .stMarkdown {
    color: #E8E8E0;
}

.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.2rem;
    color: #F4F4EF;
    text-align: center;
    letter-spacing: 0.06em;
    line-height: 1;
    margin-bottom: 0.2rem;
}

.hero-sub {
    text-align: center;
    color: #A8C9B8;
    font-size: 0.95rem;
    margin-bottom: 2rem;
    letter-spacing: 0.02em;
}

.pitch-line {
    border: none;
    border-top: 2px dashed rgba(244, 244, 239, 0.15);
    margin: 1.8rem 0;
}

.section-label {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1rem;
    letter-spacing: 0.05em;
    color: #D4A72C;
    margin-top: 1.4rem;
    margin-bottom: 0.3rem;
}

div[data-testid="stSelectbox"] label {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem;
    letter-spacing: 0.04em;
    color: #D4A72C !important;
}

div[data-testid="stSelectbox"] > div > div {
    background-color: #0E4A38;
    border: 1px solid rgba(244, 244, 239, 0.2);
    border-radius: 6px;
    color: #F4F4EF;
}

.stButton > button {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.3rem;
    letter-spacing: 0.05em;
    background-color: #D4A72C;
    color: #0B3D2E;
    border: none;
    border-radius: 6px;
    padding: 0.6rem 2rem;
    width: 100%;
    margin-top: 1rem;
    transition: transform 0.1s ease;
}

.stButton > button:hover {
    background-color: #E8BE3F;
    color: #0B3D2E;
    transform: scale(1.01);
}

.stButton > button:disabled {
    background-color: rgba(212, 167, 44, 0.3);
    color: rgba(11, 61, 46, 0.5);
}

.same-team-warning {
    background-color: rgba(196, 69, 60, 0.15);
    border: 1px solid rgba(196, 69, 60, 0.4);
    border-radius: 6px;
    padding: 0.6rem 1rem;
    color: #E8A29C;
    font-size: 0.85rem;
    margin-top: 0.6rem;
    text-align: center;
}

.scoreboard {
    background-color: rgba(0,0,0,0.2);
    border: 1px solid rgba(244, 244, 239, 0.15);
    border-radius: 10px;
    padding: 1.5rem 1rem;
    margin-top: 1.5rem;
    text-align: center;
}

.scoreboard-teams {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.6rem;
    color: #F4F4EF;
    letter-spacing: 0.03em;
    padding: 0 0.5rem;
}

.scoreboard-vs {
    font-family: 'Bebas Neue', sans-serif;
    color: #D4A72C;
    font-size: 1.1rem;
    padding: 0 1rem;
}

.stat-row {
    margin-top: 0.5rem;
    text-align: left;
}

.stat-label {
    font-size: 0.85rem;
    color: #C7C7BE;
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.15rem;
}

.stat-bar-bg {
    background-color: rgba(244, 244, 239, 0.1);
    border-radius: 4px;
    height: 10px;
    width: 100%;
    overflow: hidden;
}

.stat-bar-fill {
    height: 100%;
    border-radius: 4px;
}

.form-block {
    background-color: rgba(0,0,0,0.15);
    border: 1px solid rgba(244, 244, 239, 0.1);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-top: 0.4rem;
}

.form-strip-wrap {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.25rem 0;
}

.form-label {
    font-size: 0.85rem;
    color: #E8E8E0;
    letter-spacing: 0.01em;
}

.form-badges {
    display: flex;
    gap: 4px;
}

.form-badge {
    width: 22px;
    height: 22px;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    font-weight: 700;
    color: #F4F4EF;
}

.h2h-note {
    font-size: 0.72rem;
    color: #7A9285;
    margin-top: 0.3rem;
}

.elo-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.35rem 0;
}

.elo-rank {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1rem;
    color: #7A9285;
    width: 22px;
    text-align: right;
}

.elo-name {
    font-size: 0.85rem;
    color: #F4F4EF;
    width: 150px;
    flex-shrink: 0;
}

.elo-bar-bg {
    flex-grow: 1;
    background-color: rgba(244, 244, 239, 0.08);
    border-radius: 4px;
    height: 16px;
    overflow: hidden;
}

.elo-bar-fill {
    height: 100%;
    border-radius: 4px;
}

.elo-value {
    font-size: 0.8rem;
    color: #C7C7BE;
    width: 50px;
    text-align: right;
    flex-shrink: 0;
}

section[data-testid="stSidebar"] {
    background-color: #0A2E22;
    border-right: 1px solid rgba(244, 244, 239, 0.1);
}

.sidebar-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.3rem;
    letter-spacing: 0.05em;
    color: #D4A72C;
    margin-bottom: 0.8rem;
}

section[data-testid="stSidebar"] p {
    font-size: 0.82rem;
    color: #C7C7BE;
    line-height: 1.5;
}

section[data-testid="stSidebar"] .streamlit-expanderHeader {
    font-size: 0.85rem;
    color: #E8E8E0;
}

.sidebar-divider {
    border-top: 1px solid rgba(244, 244, 239, 0.15);
    margin: 1.2rem 0 0.8rem 0;
}

.sidebar-note {
    font-size: 0.75rem;
    color: #7A9285;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return build_final_dataset()

@st.cache_resource
def get_model(_data, league):
    return train_full_model(_data, league=league)


data, ratings = load_data()

# ---- Sidebar: about / how to read this ----
with st.sidebar:
    st.markdown('<div class="sidebar-title">HOW THIS WORKS</div>', unsafe_allow_html=True)

    with st.expander("Elo Rating"):
        st.write(
            "A running strength score for each team, starting at 1500. "
            "It moves up after a win, down after a loss, and shifts more "
            "after a surprising result than an expected one. Beating a much "
            "stronger team gains more than beating a weaker one."
        )

    with st.expander("Recent Form"):
        st.write(
            "Each team's average points and goals from their last 5 matches, "
            "regardless of opponent. Shown as W/D/L badges, most recent on the right."
        )

    with st.expander("Head to Head"):
        st.write(
            "Results from the last meetings between the two selected teams specifically, "
            "shown from the home team's perspective."
        )

    with st.expander("Why draws are rare in predictions"):
        st.write(
            "The model rarely predicts a draw outright. This is a known limitation "
            "in football prediction generally, draws don't carry as strong a "
            "statistical signal as a clear favourite or underdog. The probability "
            "shown for a draw is usually real, just rarely the single most likely outcome."
        )

    with st.expander("On the probabilities"):
        st.write(
            "Predictions are calibrated, meaning the model's stated confidence is "
            "checked against how often it's actually right at that confidence level. "
            "Earlier versions of this model were overconfident for clear favourites; "
            "calibration corrected that."
        )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-note">Trained on 6 seasons (2020-21 to 2025-26) per league. Model accuracy sits around 50-53%, against a naive baseline of 39-47% depending on league.</div>', unsafe_allow_html=True)

# ---- Header ----
st.markdown('<div class="hero-title">MATCH PREDICTOR</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Elo ratings, form, and head-to-head history across five leagues</div>', unsafe_allow_html=True)

# ---- League + team selection ----
leagues = sorted(data['League'].unique())
league = st.selectbox("LEAGUE", leagues)
accent = LEAGUE_ACCENTS.get(league, '#D4A72C')

league_data = data[data['League'] == league]
teams = sorted(set(league_data['HomeTeam']) | set(league_data['AwayTeam']))
team_display = {display_name(t): t for t in teams}
display_list = sorted(team_display.keys())

model, scaler = get_model(data, league)

col1, col2 = st.columns(2)
with col1:
    home_display = st.selectbox("HOME", display_list, index=0)
with col2:
    away_display = st.selectbox("AWAY", display_list, index=1)

home_team = team_display[home_display]
away_team = team_display[away_display]

same_team = home_team == away_team

if same_team:
    st.markdown('<div class="same-team-warning">Pick two different teams to compare.</div>', unsafe_allow_html=True)

# ---- Recent form ----
st.markdown('<div class="section-label">RECENT FORM</div>', unsafe_allow_html=True)

home_form = get_last_n_results(data, home_team, league)
away_form = get_last_n_results(data, away_team, league)

form_html = build_badge_row(home_display, home_form) + build_badge_row(away_display, away_form)
st.markdown(f'<div class="form-block">{form_html}</div>', unsafe_allow_html=True)

# ---- Head to head ----
if not same_team:
    st.markdown('<div class="section-label">HEAD TO HEAD</div>', unsafe_allow_html=True)

    h2h_results = get_h2h_results(data, home_team, away_team, league)
    h2h_html = build_badge_row(f"{home_display} record", h2h_results)
    st.markdown(f'<div class="form-block">{h2h_html}<div class="h2h-note">From {home_display}\'s perspective, oldest to most recent</div></div>', unsafe_allow_html=True)

# ---- Predict ----
predict_clicked = st.button("PREDICT", disabled=same_team)

if predict_clicked and not same_team:
    result = predict_match(model, scaler, data, home_team, away_team, league)
    if result:
        outcome_order = ['H', 'D', 'A']
        labels = {'H': home_display, 'A': away_display, 'D': 'Draw'}

        bars_html = ""
        for key in sorted(outcome_order, key=lambda k: -result[k]):
            prob = result[key]
            bar_color = accent if key != 'D' else '#8A8A80'
            bars_html += (
                f'<div class="stat-row">'
                f'<div class="stat-label"><span>{labels[key]}</span><span>{prob:.1%}</span></div>'
                f'<div class="stat-bar-bg">'
                f'<div class="stat-bar-fill" style="width:{prob*100}%; background-color:{bar_color};"></div>'
                f'</div></div>'
            )

        scoreboard_html = (
            f'<div class="scoreboard">'
            f'<div class="scoreboard-teams">'
            f'<span>{home_display}</span>'
            f'<span class="scoreboard-vs">VS</span>'
            f'<span>{away_display}</span>'
            f'</div>'
            f'{bars_html}'
            f'</div>'
        )

        st.markdown(scoreboard_html, unsafe_allow_html=True)

# ---- Elo ratings, themed bar list instead of a plain table ----
st.markdown('<hr class="pitch-line">', unsafe_allow_html=True)
st.markdown(f'<h3>ELO RATINGS — {league.upper()}</h3>', unsafe_allow_html=True)

league_ratings = {k[1]: v for k, v in ratings.items() if k[0] == league}
sorted_ratings = sorted(league_ratings.items(), key=lambda x: -x[1])
max_elo = sorted_ratings[0][1]
min_elo = sorted_ratings[-1][1]
elo_range = max_elo - min_elo if max_elo != min_elo else 1

elo_rows_html = ""
for i, (team, elo) in enumerate(sorted_ratings, start=1):
    pct = ((elo - min_elo) / elo_range) * 100
    pct = max(pct, 4)
    bar_color = '#D4A72C' if i <= 3 else accent
    elo_rows_html += (
        f'<div class="elo-row">'
        f'<span class="elo-rank">{i}</span>'
        f'<span class="elo-name">{display_name(team)}</span>'
        f'<div class="elo-bar-bg"><div class="elo-bar-fill" style="width:{pct}%; background-color:{bar_color};"></div></div>'
        f'<span class="elo-value">{elo:.0f}</span>'
        f'</div>'
    )

st.markdown(f'<div class="form-block">{elo_rows_html}</div>', unsafe_allow_html=True)