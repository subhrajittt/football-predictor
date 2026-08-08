import pandas as pd
from data_prep import load_all_seasons

# API name -> CSV name, only need entries where they actually differ
NAME_MAP = {
    'Manchester City': 'Man City',
    'Manchester United': 'Man United',
    'Nottingham Forest': "Nott'm Forest",
    'Newcastle United': 'Newcastle',
    'Wolverhampton Wanderers': 'Wolves',
    'Brighton & Hove Albion': 'Brighton',
    'Tottenham Hotspur': 'Tottenham',
    'West Ham United': 'West Ham',
    'Leicester City': 'Leicester',
    'Leeds United': 'Leeds',
    'Sheffield Utd': 'Sheffield United',
}

def normalize_names(df, col):
    df = df.copy()
    df[col] = df[col].replace(NAME_MAP)
    return df


def compare(api_csv_path, csv_season_tag):
    api_df = pd.read_csv(api_csv_path)
    api_df['Date'] = pd.to_datetime(api_df['Date'])
    api_df = normalize_names(api_df, 'HomeTeam')
    api_df = normalize_names(api_df, 'AwayTeam')

    csv_data = load_all_seasons()
    csv_data = csv_data[csv_data['Season'] == csv_season_tag]

    merged = api_df.merge(
        csv_data, on=['Date', 'HomeTeam', 'AwayTeam'],
        suffixes=('_api', '_csv'), how='outer', indicator=True
    )

    matched = merged[merged['_merge'] == 'both']
    only_api = merged[merged['_merge'] == 'left_only']
    only_csv = merged[merged['_merge'] == 'right_only']

    print(f"Matched matches: {len(matched)}")
    print(f"Only in API: {len(only_api)}")
    print(f"Only in CSV: {len(only_csv)}")

    mismatches = matched[matched['FTR_api'] != matched['FTR_csv']]
    print(f"\nResult mismatches on matched matches: {len(mismatches)}")
    if len(mismatches) > 0:
        print(mismatches[['Date', 'HomeTeam', 'AwayTeam', 'FTR_api', 'FTR_csv']])

    return merged


if __name__ == "__main__":
    merged = compare("data/raw_api/PL_2023-24_api.csv", "PL_2023-24")

    only_api = merged[merged['_merge'] == 'left_only']
    only_csv = merged[merged['_merge'] == 'right_only']

    print("\nSample only-in-API rows:")
    print(only_api[['Date', 'HomeTeam', 'AwayTeam']].head(10))

    print("\nSample only-in-CSV rows:")
    print(only_csv[['Date', 'HomeTeam', 'AwayTeam']].head(10))