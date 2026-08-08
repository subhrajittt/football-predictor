import pandas as pd
from data_prep import load_all_seasons
from features import build_team_match_log, add_rolling_form, merge_form_into_matches
from elo import calculate_elo
from head_to_head import add_head_to_head


def build_final_dataset():
    data = load_all_seasons()

    team_log = build_team_match_log(data)
    team_log = add_rolling_form(team_log, window=5)
    data = merge_form_into_matches(data, team_log)

    data, ratings = calculate_elo(data)
    data = add_head_to_head(data)

    form_cols = [
        'Home_Form_Points', 'Home_Form_GoalsFor', 'Home_Form_GoalsAgainst',
        'Away_Form_Points', 'Away_Form_GoalsFor', 'Away_Form_GoalsAgainst'
    ]
    for col in form_cols:
        data[col] = data[col].fillna(data[col].mean())

    data['H2H_HomeWinRate'] = data['H2H_HomeWinRate'].fillna(0.5)

    return data, ratings


if __name__ == "__main__":
    final_data, ratings = build_final_dataset()
    print(f"Final dataset: {final_data.shape}")
    print(final_data['League'].value_counts())
    final_data.to_csv("data/processed/final_dataset.csv", index=False)