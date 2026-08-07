import pandas as pd
import glob
import os

def load_all_seasons(raw_dir="data/raw/"):
    files = glob.glob(os.path.join(raw_dir, "*.csv"))
    core_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df = df[core_cols].copy()
        df['Season'] = os.path.basename(f).replace('.csv', '')
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined['Date'] = pd.to_datetime(combined['Date'], dayfirst=True)
    combined = combined.sort_values('Date').reset_index(drop=True)

    return combined


if __name__ == "__main__":
    data = load_all_seasons()
    print(f"Total matches: {data.shape[0]}")
    print(data['FTR'].value_counts())