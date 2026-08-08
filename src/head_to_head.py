import pandas as pd
from data_prep import load_all_seasons

def add_head_to_head(data):
    data = data.sort_values('Date').reset_index(drop=True)
    history = {}

    winrates, meetings = [], []

    for row in data.itertuples():
        pair = frozenset([row.HomeTeam, row.AwayTeam])
        past = history.get(pair, [])

        if past:
            home_wins = sum(1 for w in past if w == row.HomeTeam)
            winrate = home_wins / len(past)
        else:
            winrate = None

        winrates.append(winrate)
        meetings.append(len(past))

        if row.FTR == 'H':
            winner = row.HomeTeam
        elif row.FTR == 'A':
            winner = row.AwayTeam
        else:
            winner = None

        history.setdefault(pair, []).append(winner)

    data['H2H_HomeWinRate'] = winrates
    data['H2H_Meetings'] = meetings
    return data


if __name__ == "__main__":
    data = load_all_seasons()
    data = add_head_to_head(data)
    print(data[['Date', 'League', 'HomeTeam', 'AwayTeam', 'H2H_HomeWinRate', 'H2H_Meetings']].tail(10))