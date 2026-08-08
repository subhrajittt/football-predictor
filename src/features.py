import pandas as pd
from data_prep import load_all_seasons

def build_team_match_log(data):
    home = data[['Date', 'HomeTeam', 'FTHG', 'FTAG', 'FTR', 'League']].copy()
    home.columns = ['Date', 'Team', 'GoalsFor', 'GoalsAgainst', 'Result', 'League']
    home['Points'] = home['Result'].map({'H': 3, 'D': 1, 'A': 0})
    home['Venue'] = 'Home'

    away = data[['Date', 'AwayTeam', 'FTAG', 'FTHG', 'FTR', 'League']].copy()
    away.columns = ['Date', 'Team', 'GoalsFor', 'GoalsAgainst', 'Result', 'League']
    away['Points'] = away['Result'].map({'H': 0, 'D': 1, 'A': 3})
    away['Venue'] = 'Away'

    team_log = pd.concat([home, away], ignore_index=True)
    return team_log.sort_values(['Team', 'Date']).reset_index(drop=True)


def add_rolling_form(team_log, window=5):
    team_log = team_log.sort_values(['Team', 'Date'])

    for col, out in [('Points', 'Form_Points'), ('GoalsFor', 'Form_GoalsFor'), ('GoalsAgainst', 'Form_GoalsAgainst')]:
        team_log[out] = team_log.groupby('Team')[col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
    return team_log


def merge_form_into_matches(data, team_log):
    home_form = team_log[team_log['Venue'] == 'Home'][
        ['Date', 'Team', 'Form_Points', 'Form_GoalsFor', 'Form_GoalsAgainst']
    ].rename(columns={'Team': 'HomeTeam', 'Form_Points': 'Home_Form_Points',
                       'Form_GoalsFor': 'Home_Form_GoalsFor', 'Form_GoalsAgainst': 'Home_Form_GoalsAgainst'})

    away_form = team_log[team_log['Venue'] == 'Away'][
        ['Date', 'Team', 'Form_Points', 'Form_GoalsFor', 'Form_GoalsAgainst']
    ].rename(columns={'Team': 'AwayTeam', 'Form_Points': 'Away_Form_Points',
                       'Form_GoalsFor': 'Away_Form_GoalsFor', 'Form_GoalsAgainst': 'Away_Form_GoalsAgainst'})

    merged = data.merge(home_form, on=['Date', 'HomeTeam'], how='left')
    merged = merged.merge(away_form, on=['Date', 'AwayTeam'], how='left')
    return merged


if __name__ == "__main__":
    data = load_all_seasons()
    team_log = build_team_match_log(data)
    team_log = add_rolling_form(team_log)
    final = merge_form_into_matches(data, team_log)
    print(final[['Date', 'League', 'HomeTeam', 'AwayTeam', 'Home_Form_Points', 'Away_Form_Points']].head())