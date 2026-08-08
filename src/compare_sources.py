import pandas as pd
from data_prep import load_all_seasons
from api_client import get_fixtures, fixtures_to_dataframe

NAME_MAP = {
    # Premier League
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

    # La Liga
    'Athletic Club': 'Ath Bilbao',
    'Atletico Madrid': 'Ath Madrid',
    'Celta Vigo': 'Celta',
    'Granada CF': 'Granada',
    'Rayo Vallecano': 'Vallecano',
    'Real Betis': 'Betis',

    # Serie A
    'AC Milan': 'Milan',
    'AS Roma': 'Roma',
    'Hellas Verona': 'Verona',

    # Bundesliga
    '1. FC Köln': 'FC Koln',
    '1899 Hoffenheim': 'Hoffenheim',
    'Bayer Leverkusen': 'Leverkusen',
    'Borussia Dortmund': 'Dortmund',
    'Borussia Monchengladbach': "M'gladbach",
    'Eintracht Frankfurt': 'Ein Frankfurt',
    'FC Augsburg': 'Augsburg',
    'FC Heidenheim': 'Heidenheim',
    'FSV Mainz 05': 'Mainz',
    'SC Freiburg': 'Freiburg',
    'SV Darmstadt 98': 'Darmstadt',

    # Ligue 1
    'Clermont Foot': 'Clermont',
    'Paris Saint Germain': 'Paris SG',
}

LEAGUE_IDS = {
    'Premier League': 39,
    'La Liga': 140,
    'Serie A': 135,
    'Bundesliga': 78,
    'Ligue 1': 61,
}


def normalize_names(df, col):
    df = df.copy()
    df[col] = df[col].replace(NAME_MAP)
    return df


def compare(league_name, league_id, season_year, csv_season_tag, show_mismatches=False):
    api_data = get_fixtures(league_id=league_id, season=season_year, status="FT")

    if api_data['results'] == 0:
        print(f"{league_name}: no data returned. Errors: {api_data['errors']}")
        return None

    api_df = fixtures_to_dataframe(api_data)
    api_df['Date'] = pd.to_datetime(api_df['Date'])
    api_df = normalize_names(api_df, 'HomeTeam')
    api_df = normalize_names(api_df, 'AwayTeam')

    csv_data = load_all_seasons()
    csv_data = csv_data[csv_data['Season'] == csv_season_tag]

    # drop any API rows involving teams that don't exist in our CSV data for this league
    # catches cases like a second-division team appearing in the API response
    known_teams = set(csv_data['HomeTeam']) | set(csv_data['AwayTeam'])
    before = len(api_df)
    api_df = api_df[api_df['HomeTeam'].isin(known_teams) & api_df['AwayTeam'].isin(known_teams)]
    dropped = before - len(api_df)
    if dropped > 0:
        print(f"  Dropped {dropped} API rows with teams not in our known {league_name} list")

    merged = api_df.merge(
        csv_data, on=['Date', 'HomeTeam', 'AwayTeam'],
        suffixes=('_api', '_csv'), how='outer', indicator=True
    )

    matched = merged[merged['_merge'] == 'both']
    only_api = merged[merged['_merge'] == 'left_only']

    print(f"{league_name} ({csv_season_tag}):")
    print(f"  Matched: {len(matched)} / {len(api_df)}")

    if show_mismatches and len(only_api) > 0:
        api_only_teams = sorted(set(only_api['HomeTeam']) | set(only_api['AwayTeam']))
        print(f"  Still unmatched team names: {api_only_teams[:15]}")

    return merged


if __name__ == "__main__":
    checks = [
        ('Premier League', 39, 2023, 'PL_2023-24'),
        ('La Liga', 140, 2023, 'LaLiga_2023-24'),
        ('Serie A', 135, 2023, 'SerieA_2023-24'),
        ('Bundesliga', 78, 2023, 'Bundesliga_2023-24'),
        ('Ligue 1', 61, 2023, 'Ligue1_2023-24'),
    ]

    for league_name, league_id, season_year, csv_tag in checks:
        compare(league_name, league_id, season_year, csv_tag, show_mismatches=True)