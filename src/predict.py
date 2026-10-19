from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "wine_model.pkl"
CLASS_NAMES = ["class_0", "class_1", "class_2"]

_model = None


def get_model():
    """Load the model once and reuse it."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict(features):
    """features: list of 13 floats. Returns predicted class and probabilities."""
    model = get_model()
    pred = int(model.predict([features])[0])
    probs = model.predict_proba([features])[0]
    return {
        "class_id": pred,
        "class_name": CLASS_NAMES[pred],
        "probabilities": {CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)},
    }
