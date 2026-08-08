import sys
from build_dataset import build_final_dataset
from predict import train_full_model


def update():
    print("Rebuilding dataset from data/raw/ ...")
    data, ratings = build_final_dataset()
    data.to_csv("data/processed/final_dataset.csv", index=False)
    print(f"Dataset rebuilt: {data.shape[0]} matches, saved to data/processed/final_dataset.csv")

    print("\nRetraining model on full dataset ...")
    model, scaler = train_full_model(data)
    print("Model retrained.")

    print("\nTop 10 Elo ratings after update:")
    top = sorted(ratings.items(), key=lambda x: -x[1])[:10]
    for team, r in top:
        print(f"  {team}: {r:.1f}")

    return data, ratings, model, scaler


if __name__ == "__main__":
    update()