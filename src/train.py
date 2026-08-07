import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from build_dataset import build_final_dataset


def prepare_train_test(data, test_seasons=['PL_2024-25', 'PL_2025-26']):
    feature_cols = [
        'Home_Form_Points', 'Home_Form_GoalsFor', 'Home_Form_GoalsAgainst',
        'Away_Form_Points', 'Away_Form_GoalsFor', 'Away_Form_GoalsAgainst',
        'Home_Elo', 'Away_Elo'
    ]
    target_col = 'FTR'

    train = data[~data['Season'].isin(test_seasons)]
    test = data[data['Season'].isin(test_seasons)]

    X_train, y_train = train[feature_cols], train[target_col]
    X_test, y_test = test[feature_cols], test[target_col]

    return X_train, X_test, y_train, y_test, feature_cols


def naive_baseline_accuracy(y_test):
    most_common = y_test.mode()[0]
    naive_preds = [most_common] * len(y_test)
    return accuracy_score(y_test, naive_preds), most_common


def train_model(X_train, y_train, balanced=False):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    weight = 'balanced' if balanced else None
    model = LogisticRegression(max_iter=1000, class_weight=weight)
    model.fit(X_train_scaled, y_train)

    return model, scaler


def evaluate(model, scaler, X_test, y_test, label=""):
    X_test_scaled = scaler.transform(X_test)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)

    print(f"\n=== {label} ===")
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_test, preds, zero_division=0))
    print(model.classes_)
    print(confusion_matrix(y_test, preds, labels=model.classes_))
    return acc


if __name__ == "__main__":
    data, ratings = build_final_dataset()
    X_train, X_test, y_train, y_test, feature_cols = prepare_train_test(data)

    print(f"Train set: {X_train.shape[0]} matches")
    print(f"Test set: {X_test.shape[0]} matches")

    naive_acc, most_common = naive_baseline_accuracy(y_test)
    print(f"Naive baseline (always predict '{most_common}'): {naive_acc:.3f}")

    model, scaler = train_model(X_train, y_train)
    acc = evaluate(model, scaler, X_test, y_test, label="Logistic Regression")

    # tried class_weight='balanced' to force draw predictions - accuracy dropped
    # to 0.467, draw recall only hit 0.11. not worth the tradeoff, keeping standard.
    model_bal, scaler_bal = train_model(X_train, y_train, balanced=True)
    acc_bal = evaluate(model_bal, scaler_bal, X_test, y_test, label="Balanced (comparison only)")

    print(f"\nBeats naive baseline: {acc > naive_acc}")