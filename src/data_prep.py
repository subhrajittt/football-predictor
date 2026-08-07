import pandas as pd
import glob
import os

def load_all_seasons(raw_dir="data/raw/"):
    """Load all season CSVs and combine into one dataframe with just the core columns."""
    
    files = glob.glob(os.path.join(raw_dir, "*.csv"))
    print(f"Found {len(files)} files:")
    for f in files:
        print(f"  - {f}")
    
    core_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df = df[core_cols].copy()
        df['Season'] = os.path.basename(f).replace('.csv', '')  # tag which season this row came from
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Convert Date column to actual datetime (currently just text)
    combined['Date'] = pd.to_datetime(combined['Date'], dayfirst=True)
    
    # Sort chronologically - important for later when we do time-based train/test split
    combined = combined.sort_values('Date').reset_index(drop=True)
    
    return combined


if __name__ == "__main__":
    data = load_all_seasons()
    
    print(f"\nTotal matches loaded: {data.shape[0]}")
    print(f"\nColumns: {list(data.columns)}")
    print(f"\nFirst few rows:")
    print(data.head())
    print(f"\nLast few rows:")
    print(data.tail())
    print(f"\nResult breakdown (H=Home win, D=Draw, A=Away win):")
    print(data['FTR'].value_counts())
    print(f"\nAny missing values?")
    print(data.isnull().sum())