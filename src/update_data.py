from build_dataset import build_final_dataset
from predict import train_full_model


def update():
    print("Rebuilding dataset from data/raw/ ...")
    data, ratings = build_final_dataset()
    data.to_csv("data/processed/final_dataset.csv", index=False)
    print(f"Dataset rebuilt: {data.shape[0]} matches, saved to data/processed/final_dataset.csv")

    print("\nRetraining models per league ...")
    models = {}
    for league in data['League'].unique():
        model, scaler = train_full_model(data, league=league)
        models[league] = (model, scaler)
        print(f"  {league}: retrained")

    print("\nTop 5 Elo per league after update:")
    for league in data['League'].unique():
        league_ratings = {k[1]: v for k, v in ratings.items() if k[0] == league}
        top = sorted(league_ratings.items(), key=lambda x: -x[1])[:5]
        print(f"\n  {league}:")
        for team, r in top:
            print(f"    {team}: {r:.1f}")

    return data, ratings, models


if __name__ == "__main__":
    update()