from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from data import load_data, split_data

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "wine_model.pkl"


def train():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Test accuracy: {acc:.4f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train()
