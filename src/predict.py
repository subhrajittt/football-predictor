import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from build_dataset import build_final_dataset


FEATURE_COLS = [
    'Home_Form_Points', 'Home_Form_GoalsFor', 'Home_Form_GoalsAgainst',
    'Away_Form_Points', 'Away_Form_GoalsFor', 'Away_Form_GoalsAgainst',
    'Home_Elo', 'Away_Elo'
]


def train_full_model(data):
    # train on everything we have, for real predictions (not evaluation)
    X = data[FEATURE_COLS]
    y = data['FTR']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    return model, scaler


def get_latest_team_stats(data, team):
    # last home appearance for Home_* stats, last away appearance for Away_* stats
    home_rows = data[data['HomeTeam'] == team].sort_values('Date')
    away_rows = data[data['AwayTeam'] == team].sort_values('Date')

    if len(home_rows) == 0 or len(away_rows) == 0:
        return None

    latest_home = home_rows.iloc[-1]
    latest_away = away_rows.iloc[-1]

    # elo is team strength regardless of venue, take whichever match is more recent
    if latest_home['Date'] > latest_away['Date']:
        elo = latest_home['Home_Elo']
    else:
        elo = latest_away['Away_Elo']

    return {
        'form_points': latest_home['Home_Form_Points'],
        'form_goals_for': latest_home['Home_Form_GoalsFor'],
        'form_goals_against': latest_home['Home_Form_GoalsAgainst'],
        'away_form_points': latest_away['Away_Form_Points'],
        'away_form_goals_for': latest_away['Away_Form_GoalsFor'],
        'away_form_goals_against': latest_away['Away_Form_GoalsAgainst'],
        'elo': elo
    }


def predict_match(model, scaler, data, home_team, away_team):
    home_stats = get_latest_team_stats(data, home_team)
    away_stats = get_latest_team_stats(data, away_team)

    if home_stats is None or away_stats is None:
        print(f"No data found for one of: {home_team}, {away_team}")
        return None

    row = pd.DataFrame([{
        'Home_Form_Points': home_stats['form_points'],
        'Home_Form_GoalsFor': home_stats['form_goals_for'],
        'Home_Form_GoalsAgainst': home_stats['form_goals_against'],
        'Away_Form_Points': away_stats['away_form_points'],
        'Away_Form_GoalsFor': away_stats['away_form_goals_for'],
        'Away_Form_GoalsAgainst': away_stats['away_form_goals_against'],
        'Home_Elo': home_stats['elo'],
        'Away_Elo': away_stats['elo']
    }])

    row_scaled = scaler.transform(row)
    probs = model.predict_proba(row_scaled)[0]

    result = dict(zip(model.classes_, probs))
    return result


if __name__ == "__main__":
    data, ratings = build_final_dataset()
    model, scaler = train_full_model(data)

    matchups = [
        ('Arsenal', 'Chelsea'),
        ('Man City', 'Liverpool'),
        ('Burnley', 'Man United'),
    ]

    for home, away in matchups:
        result = predict_match(model, scaler, data, home, away)
        if result:
            print(f"\n{home} vs {away}")
            for outcome, prob in sorted(result.items(), key=lambda x: -x[1]):
                label = {'H': f'{home} win', 'A': f'{away} win', 'D': 'Draw'}[outcome]
                print(f"  {label}: {prob:.1%}")