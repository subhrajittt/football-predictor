import os
import requests
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

API_KEY = os.getenv("API_FOOTBALL_KEY")
BASE_URL = "https://v3.football.api-sports.io"

HEADERS = {
    "x-apisports-key": API_KEY
}

def test_connection():
    response = requests.get(f"{BASE_URL}/status", headers=HEADERS)
    print(response.status_code)
    print(response.json())


if __name__ == "__main__":
    test_connection()

def get_fixtures(league_id, season, status=None):
    """
    status=None -> all fixtures (past + future)
    status='FT' -> only finished matches
    status='NS' -> only not-yet-started (upcoming) matches
    """
    params = {"league": league_id, "season": season}
    if status:
        params["status"] = status

    response = requests.get(f"{BASE_URL}/fixtures", headers=HEADERS, params=params)
    return response.json()


if __name__ == "__main__":
    test_connection()

    print("\nFetching a few Premier League fixtures (2025 season)...")
    data = get_fixtures(league_id=39, season=2025, status="FT")
    print(f"Results found: {data['results']}")
    print(f"Errors: {data['errors']}")
    print(f"Parameters sent: {data['parameters']}")
    
    if data['response']:
        sample = data['response'][0]
        print("\nSample fixture:")
        print(sample['teams'])
        print(sample['goals'])
        print(sample['fixture']['date'])

def fixtures_to_dataframe(api_response):
    rows = []
    for fixture in api_response['response']:
        home_goals = fixture['goals']['home']
        away_goals = fixture['goals']['away']

        if home_goals is None or away_goals is None:
            continue  # match not finished, skip

        if home_goals > away_goals:
            result = 'H'
        elif away_goals > home_goals:
            result = 'A'
        else:
            result = 'D'

        rows.append({
            'Date': fixture['fixture']['date'][:10],
            'HomeTeam': fixture['teams']['home']['name'],
            'AwayTeam': fixture['teams']['away']['name'],
            'FTHG': home_goals,
            'FTAG': away_goals,
            'FTR': result
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    test_connection()

    print("\nFetching Premier League 2023 season via API...")
    data = get_fixtures(league_id=39, season=2023, status="FT")
    print(f"Results found: {data['results']}")
    print(f"Errors: {data['errors']}")

    df = fixtures_to_dataframe(data)
    print(f"\nConverted to dataframe: {df.shape}")
    print(df.head())

    df.to_csv("data/raw_api/PL_2023-24_api.csv", index=False)
    print("\nSaved to data/raw_api/PL_2023-24_api.csv")