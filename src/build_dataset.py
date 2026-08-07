import pandas as pd
from data_prep import load_all_seasons
from features import build_team_match_log, add_rolling_form, merge_form_into_matches
from elo import calculate_elo


def build_final_dataset():
    # 1. Load raw matches
    data = load_all_seasons()

    # 2. Add rolling form features
    team_log = build_team_match_log(data)
    team_log = add_rolling_form(team_log, window=5)
    data = merge_form_into_matches(data, team_log)

    # 3. Add Elo ratings
    data, final_ratings = calculate_elo(data)

    # 4. Fill NaNs (early-season matches with no prior form) with league averages
    form_cols = [
        'Home_Form_Points', 'Home_Form_GoalsFor', 'Home_Form_GoalsAgainst',
        'Away_Form_Points', 'Away_Form_GoalsFor', 'Away_Form_GoalsAgainst'
    ]
    for col in form_cols:
        mean_val = data[col].mean()
        data[col] = data[col].fillna(mean_val)
        print(f"Filled NaNs in {col} with mean = {mean_val:.3f}")

    # 5. Sanity check - no NaNs should remain in our feature columns
    print("\nRemaining NaNs in feature columns:")
    print(data[form_cols + ['Home_Elo', 'Away_Elo']].isnull().sum())

    return data, final_ratings


if __name__ == "__main__":
    final_data, ratings = build_final_dataset()

    print(f"\nFinal dataset shape: {final_data.shape}")
    print(f"\nColumns: {list(final_data.columns)}")
    print(f"\nSample rows:")
    print(final_data[['Date', 'HomeTeam', 'AwayTeam', 'FTR',
                       'Home_Form_Points', 'Away_Form_Points',
                       'Home_Elo', 'Away_Elo']].sample(10))

    final_data.to_csv("data/processed/final_dataset.csv", index=False)
    print("\nSaved to data/processed/final_dataset.csv")