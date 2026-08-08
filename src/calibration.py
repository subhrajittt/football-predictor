import pandas as pd
import numpy as np
from build_dataset import build_final_dataset
from train import prepare_train_test, train_model, train_calibrated_model


def get_calibration_data(data, league, calibrated=False):
    X_train, X_test, y_train, y_test = prepare_train_test(data, league)

    if calibrated:
        model, scaler = train_calibrated_model(X_train, y_train)
    else:
        model, scaler = train_model(X_train, y_train)

    X_test_scaled = scaler.transform(X_test)
    probs = model.predict_proba(X_test_scaled)

    classes = model.classes_
    pred_idx = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    predicted_class = classes[pred_idx]

    actual = y_test.values
    correct = (predicted_class == actual)

    return pd.DataFrame({'confidence': confidence, 'correct': correct})


def calibration_report(df, bins=[0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]):
    df = df.copy()
    df['bucket'] = pd.cut(df['confidence'], bins=bins)

    report = df.groupby('bucket', observed=True).agg(
        count=('correct', 'size'),
        avg_confidence=('confidence', 'mean'),
        actual_accuracy=('correct', 'mean')
    )
    return report


if __name__ == "__main__":
    data, ratings = build_final_dataset()

    for league in data['League'].unique():
        print(f"\n=== {league} - STANDARD ===")
        cal_data = get_calibration_data(data, league, calibrated=False)
        print(calibration_report(cal_data))

        print(f"\n=== {league} - CALIBRATED ===")
        cal_data_c = get_calibration_data(data, league, calibrated=True)
        print(calibration_report(cal_data_c))