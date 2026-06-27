from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_MODEL_PATH = PROJECT_ROOT / "models" / "final" / "final_model.joblib"

DEFAULT_INPUT = {
    "G2": 13,
    "absences": 6,
    "G1": 12,
    "age": 15,
    "reason": "other",
    "freetime": 3,
    "health": 3,
    "Fedu": 1,
    "school": "GP",
    "failures": 0,
}

CATEGORY_OPTIONS = {
    "reason": ["course", "home", "other", "reputation"],
    "school": ["GP", "MS"],
}

NUMERIC_FEATURES = ["G2", "absences", "G1", "age", "freetime", "health", "Fedu", "failures"]


def load_final_model(model_path=FINAL_MODEL_PATH):
    """Ucitava sacuvani finalni model bundle."""
    if not model_path.exists():
        raise FileNotFoundError(f"Finalni model ne postoji: {model_path}")

    return joblib.load(model_path)


def get_feature_columns(model_bundle):
    """Vraca atribute koje finalni model ocekuje."""
    return list(model_bundle["feature_columns"])


def prepare_input_dataframe(values, feature_columns):
    """Priprema jedan red podataka za predikciju."""
    missing_features = [feature for feature in feature_columns if feature not in values]
    if missing_features:
        missing_text = ", ".join(missing_features)
        raise ValueError(f"Nedostaju ulazni atributi: {missing_text}")

    input_data = {feature: values[feature] for feature in feature_columns}
    input_df = pd.DataFrame([input_data], columns=feature_columns)

    for feature in NUMERIC_FEATURES:
        if feature in input_df.columns:
            input_df[feature] = pd.to_numeric(input_df[feature])

    return input_df


def predict_final_grade(values, model_bundle=None):
    """Vraca predikciju zavrsne ocene G3 za jedan ulazni primer."""
    if model_bundle is None:
        model_bundle = load_final_model()

    feature_columns = get_feature_columns(model_bundle)
    input_df = prepare_input_dataframe(values, feature_columns)
    prediction = model_bundle["pipeline"].predict(input_df)[0]

    return float(prediction), input_df


def format_prediction(prediction):
    """Formatira predikciju i ogranicava prikaz na skalu ocena 0-20."""
    clipped_prediction = min(20.0, max(0.0, prediction))
    return f"{clipped_prediction:.2f}"
