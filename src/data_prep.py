import pandas as pd
import glob
import os

LEAGUE_PREFIXES = {
    'PL': 'Premier League',
    'LaLiga': 'La Liga',
    'SerieA': 'Serie A',
    'Bundesliga': 'Bundesliga',
    'Ligue1': 'Ligue 1'
}

def load_all_seasons(raw_dir="data/raw/"):
    files = glob.glob(os.path.join(raw_dir, "*.csv"))
    core_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']

    dfs = []
    for f in files:
        filename = os.path.basename(f).replace('.csv', '')
        prefix = filename.split('_')[0]
        league = LEAGUE_PREFIXES.get(prefix, 'Unknown')

        df = pd.read_csv(f)
        df = df[core_cols].copy()
        df['Season'] = filename
        df['League'] = league
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined['Date'] = pd.to_datetime(combined['Date'], dayfirst=True)
    return combined.sort_values('Date').reset_index(drop=True)


if __name__ == "__main__":
    data = load_all_seasons()
    print(f"Total matches: {data.shape[0]}")
    print(data['League'].value_counts())

    # add temporarily to bottom of data_prep.py's __main__ block, or run standalone
if __name__ == "__main__":
    data = load_all_seasons()
    for league in data['League'].unique():
        teams = sorted(set(data[data['League']==league]['HomeTeam']))
        print(f"\n{league} ({len(teams)} teams):")
        print(teams)