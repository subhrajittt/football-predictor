import pandas as pd
from data_prep import load_all_seasons

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def calculate_elo(data, k=20, initial_rating=1500, home_advantage=100):
    data = data.sort_values('Date').reset_index(drop=True)
    ratings = {}
    home_elo_list, away_elo_list = [], []

    for _, row in data.iterrows():
        home_team, away_team = row['HomeTeam'], row['AwayTeam']
        home_rating = ratings.get(home_team, initial_rating)
        away_rating = ratings.get(away_team, initial_rating)

        # store pre-match ratings, no leakage
        home_elo_list.append(home_rating)
        away_elo_list.append(away_rating)

        exp_home = expected_score(home_rating + home_advantage, away_rating)
        exp_away = 1 - exp_home

        if row['FTR'] == 'H':
            actual_home, actual_away = 1, 0
        elif row['FTR'] == 'A':
            actual_home, actual_away = 0, 1
        else:
            actual_home, actual_away = 0.5, 0.5

        ratings[home_team] = home_rating + k * (actual_home - exp_home)
        ratings[away_team] = away_rating + k * (actual_away - exp_away)

    data['Home_Elo'] = home_elo_list
    data['Away_Elo'] = away_elo_list
    return data, ratings


if __name__ == "__main__":
    data = load_all_seasons()
    data, ratings = calculate_elo(data)
    top = sorted(ratings.items(), key=lambda x: x[1], reverse=True)[:10]
    for team, r in top:
        print(f"{team}: {r:.1f}")