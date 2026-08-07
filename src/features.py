import pandas as pd
from data_prep import load_all_seasons

def build_team_match_log(data):
    """
    Reshape from one-row-per-match to one-row-per-team-per-match.
    This makes it much easier to compute rolling stats per team.
    """
    home = data[['Date', 'HomeTeam', 'FTHG', 'FTAG', 'FTR']].copy()
    home.columns = ['Date', 'Team', 'GoalsFor', 'GoalsAgainst', 'Result']
    home['Points'] = home['Result'].map({'H': 3, 'D': 1, 'A': 0})
    home['Venue'] = 'Home'

    away = data[['Date', 'AwayTeam', 'FTAG', 'FTHG', 'FTR']].copy()
    away.columns = ['Date', 'Team', 'GoalsFor', 'GoalsAgainst', 'Result']
    away['Points'] = away['Result'].map({'H': 0, 'D': 1, 'A': 3})
    away['Venue'] = 'Away'

    team_log = pd.concat([home, away], ignore_index=True)
    team_log = team_log.sort_values(['Team', 'Date']).reset_index(drop=True)
    return team_log


def add_rolling_form(team_log, window=5):
    """
    For each team, compute rolling average points/goals over their last N matches.
    shift(1) is the critical part - it excludes the CURRENT match, using only prior ones.
    """
    team_log = team_log.sort_values(['Team', 'Date'])

    team_log['Form_Points'] = (
        team_log.groupby('Team')['Points']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    team_log['Form_GoalsFor'] = (
        team_log.groupby('Team')['GoalsFor']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    team_log['Form_GoalsAgainst'] = (
        team_log.groupby('Team')['GoalsAgainst']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    return team_log


def merge_form_into_matches(data, team_log):
    """
    Bring the rolling form stats back into the original match-per-row format,
    once for the home team's form, once for the away team's form.
    """
    home_form = team_log[team_log['Venue'] == 'Home'][
        ['Date', 'Team', 'Form_Points', 'Form_GoalsFor', 'Form_GoalsAgainst']
    ].rename(columns={
        'Team': 'HomeTeam',
        'Form_Points': 'Home_Form_Points',
        'Form_GoalsFor': 'Home_Form_GoalsFor',
        'Form_GoalsAgainst': 'Home_Form_GoalsAgainst'
    })

    away_form = team_log[team_log['Venue'] == 'Away'][
        ['Date', 'Team', 'Form_Points', 'Form_GoalsFor', 'Form_GoalsAgainst']
    ].rename(columns={
        'Team': 'AwayTeam',
        'Form_Points': 'Away_Form_Points',
        'Form_GoalsFor': 'Away_Form_GoalsFor',
        'Form_GoalsAgainst': 'Away_Form_GoalsAgainst'
    })

    merged = data.merge(home_form, on=['Date', 'HomeTeam'], how='left')
    merged = merged.merge(away_form, on=['Date', 'AwayTeam'], how='left')
    return merged


if __name__ == "__main__":
    data = load_all_seasons()
    team_log = build_team_match_log(data)
    team_log = add_rolling_form(team_log, window=5)

    print("Sample team match log (Arsenal, first 8 rows):")
    print(team_log[team_log['Team'] == 'Arsenal'].head(8))

    final = merge_form_into_matches(data, team_log)

    print("\nSample merged matches with form features:")
    print(final[['Date', 'HomeTeam', 'AwayTeam', 'FTR',
                  'Home_Form_Points', 'Away_Form_Points']].head(10))

    print(f"\nMissing values in form columns (expected for each team's 1st ever match):")
    print(final[['Home_Form_Points', 'Away_Form_Points']].isnull().sum())

    final.to_csv("data/processed/matches_with_form.csv", index=False)
    print("\nSaved to data/processed/matches_with_form.csv")