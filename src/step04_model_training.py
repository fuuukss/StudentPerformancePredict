from pathlib import Path
from math import sqrt

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from csv_utils import save_csv_if_changed


# Putanje do osnovnih foldera i fajlova u projektu.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "split"
LOGS_DIR = PROJECT_ROOT / "data" / "logs" / "step04_model_training"

TRAIN_PATH = SPLIT_DIR / "train.csv"
VALIDATION_PATH = SPLIT_DIR / "validation.csv"
TEST_PATH = SPLIT_DIR / "test.csv"
MODEL_COMPARISON_REPORT_PATH = LOGS_DIR / "model_comparison_report.csv"

TARGET_COLUMN = "G3"
RANDOM_STATE = 42


def format_value(value):
    """Formatira vrednosti za citljiviji CSV izlaz."""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))

        return f"{value:.4f}"

    return str(value)


def get_feature_columns(df, excluded_columns):
    """Vraca ulazne atribute nakon izbacivanja ciljnih i scenario kolona."""
    return [column for column in df.columns if column not in excluded_columns]


def split_feature_types(df, feature_columns):
    """Deli atribute na numericke i kategorijske."""
    features = df[feature_columns]
    numeric_features = features.select_dtypes(include="number").columns.tolist()
    categorical_features = features.select_dtypes(exclude="number").columns.tolist()

    return numeric_features, categorical_features


def create_preprocessor(numeric_features, categorical_features):
    """Priprema preprocessing objekat za numericke i kategorijske atribute."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )


def create_scenarios(df):
    """Definise scenarije sa G1/G2 i bez G1/G2."""
    scenario_exclusions = {
        "with_G1_G2": [TARGET_COLUMN],
        "without_G1_G2": ["G1", "G2", TARGET_COLUMN],
    }
    scenarios = {}

    for scenario_name, excluded_columns in scenario_exclusions.items():
        feature_columns = get_feature_columns(df, excluded_columns)
        numeric_features, categorical_features = split_feature_types(df, feature_columns)

        scenarios[scenario_name] = {
            "features": feature_columns,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
        }

    return scenarios


def create_models():
    """Definise dogovorene modele za poredjenje."""
    return {
        "DummyRegressor": DummyRegressor(strategy="mean"),
        "Ridge Regression": Ridge(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
        ),
    }


def create_pipeline(preprocessor, model):
    """Kreira pipeline koji prvo radi preprocessing, pa trenira model."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def calculate_metrics(y_true, y_pred):
    """Racuna regresione metrike za predikcije modela."""
    mse = mean_squared_error(y_true, y_pred)

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": sqrt(mse),
        "R2": r2_score(y_true, y_pred),
    }


def evaluate_model(pipeline, X, y):
    """Racuna metrike za dati skup podataka."""
    predictions = pipeline.predict(X)
    return calculate_metrics(y, predictions)


def create_model_comparison_report(train_df, validation_df, test_df):
    """Trenira modele po scenarijima i kreira CSV izvestaj sa metrikama."""
    scenarios = create_scenarios(train_df)
    report_rows = []

    for scenario_name, scenario in scenarios.items():
        models = create_models()
        feature_columns = scenario["features"]

        X_train = train_df[feature_columns]
        y_train = train_df[TARGET_COLUMN]
        X_validation = validation_df[feature_columns]
        y_validation = validation_df[TARGET_COLUMN]
        X_test = test_df[feature_columns]
        y_test = test_df[TARGET_COLUMN]

        for model_name, model in models.items():
            preprocessor = create_preprocessor(
                scenario["numeric_features"],
                scenario["categorical_features"],
            )
            pipeline = create_pipeline(preprocessor, model)
            pipeline.fit(X_train, y_train)

            validation_metrics = evaluate_model(pipeline, X_validation, y_validation)
            test_metrics = evaluate_model(pipeline, X_test, y_test)

            report_rows.append(
                {
                    "scenario": scenario_name,
                    "model": model_name,
                    "validation_MAE": validation_metrics["MAE"],
                    "validation_RMSE": validation_metrics["RMSE"],
                    "validation_R2": validation_metrics["R2"],
                    "test_MAE": test_metrics["MAE"],
                    "test_RMSE": test_metrics["RMSE"],
                    "test_R2": test_metrics["R2"],
                }
            )

    report = pd.DataFrame(
        report_rows,
        columns=[
            "scenario",
            "model",
            "validation_MAE",
            "validation_RMSE",
            "validation_R2",
            "test_MAE",
            "test_RMSE",
            "test_R2",
        ],
    )

    metric_columns = [
        "validation_MAE",
        "validation_RMSE",
        "validation_R2",
        "test_MAE",
        "test_RMSE",
        "test_R2",
    ]
    report[metric_columns] = report[metric_columns].map(format_value)

    return report


def main():
    # 1. Ucitavanje prethodno sacuvanih train, validation i test skupova.
    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # 2. Kreiranje foldera za CSV logove ako ne postoji.
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Treniranje dogovorenih modela po scenarijima i kreiranje izvestaja.
    model_comparison_report = create_model_comparison_report(
        train_df,
        validation_df,
        test_df,
    )
    save_csv_if_changed(model_comparison_report, MODEL_COMPARISON_REPORT_PATH, PROJECT_ROOT)

    print("Treniranje i poredjenje modela je zavrseno.")


if __name__ == "__main__":
    main()
