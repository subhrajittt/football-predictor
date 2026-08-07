import pandas as pd
from data_prep import load_all_seasons

def expected_score(rating_a, rating_b):
    """
    Standard Elo formula: probability that team A beats team B,
    based on the rating gap between them.
    """
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def calculate_elo(data, k=20, initial_rating=1500, home_advantage=100):
    """
    Walk through matches in chronological order, updating each team's
    Elo rating after every match. Returns the match data with 
    pre-match Elo ratings attached (i.e. what each team's rating WAS
    going into that match - no leakage).
    """
    data = data.sort_values('Date').reset_index(drop=True)
    
    ratings = {}  # team_name -> current elo rating
    
    home_elo_list = []
    away_elo_list = []
    
    for idx, row in data.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        home_rating = ratings.get(home_team, initial_rating)
        away_rating = ratings.get(away_team, initial_rating)
        
        # Record PRE-MATCH ratings - this is what the model will use as a feature
        home_elo_list.append(home_rating)
        away_elo_list.append(away_rating)
        
        # Home advantage: boost home team's rating for the expected-score calc only
        adj_home_rating = home_rating + home_advantage
        
        exp_home = expected_score(adj_home_rating, away_rating)
        exp_away = 1 - exp_home
        
        # Actual result: 1 = win, 0.5 = draw, 0 = loss
        if row['FTR'] == 'H':
            actual_home, actual_away = 1, 0
        elif row['FTR'] == 'A':
            actual_home, actual_away = 0, 1
        else:
            actual_home, actual_away = 0.5, 0.5
        
        # Update ratings based on surprise (actual vs expected)
        new_home_rating = home_rating + k * (actual_home - exp_home)
        new_away_rating = away_rating + k * (actual_away - exp_away)
        
        ratings[home_team] = new_home_rating
        ratings[away_team] = new_away_rating
    
    data['Home_Elo'] = home_elo_list
    data['Away_Elo'] = away_elo_list
    
    return data, ratings


if __name__ == "__main__":
    data = load_all_seasons()
    data_with_elo, final_ratings = calculate_elo(data)
    
    print("Sample matches with Elo ratings:")
    print(data_with_elo[['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'Home_Elo', 'Away_Elo']].head(10))
    
    print("\nFinal Elo ratings (top 10 strongest teams, current):")
    sorted_ratings = sorted(final_ratings.items(), key=lambda x: x[1], reverse=True)
    for team, rating in sorted_ratings[:10]:
        print(f"  {team}: {rating:.1f}")
    
    print("\nFinal Elo ratings (bottom 5 weakest teams, current):")
    for team, rating in sorted_ratings[-5:]:
        print(f"  {team}: {rating:.1f}")