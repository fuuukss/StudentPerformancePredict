from itertools import product
from math import sqrt
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from csv_utils import save_csv_if_changed


# Putanje do osnovnih foldera i fajlova u projektu.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "split"
LOGS_DIR = PROJECT_ROOT / "data" / "logs" / "step07_hyperparameter_tuning"

TRAIN_PATH = SPLIT_DIR / "train.csv"
VALIDATION_PATH = SPLIT_DIR / "validation.csv"
TEST_PATH = SPLIT_DIR / "test.csv"
TUNING_DETAILS_REPORT_PATH = LOGS_DIR / "hyperparameter_tuning_details.csv"
TUNING_SUMMARY_REPORT_PATH = LOGS_DIR / "hyperparameter_tuning_summary.csv"

TARGET_COLUMN = "G3"
RANDOM_STATE = 42


def format_value(value):
    """Formatira vrednosti za citljiviji CSV izlaz."""
    if isinstance(value, bool):
        return "Da" if value else "Ne"

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


def create_model(model_name, parameters):
    """Kreira model za datu kombinaciju hiperparametara."""
    if model_name == "Ridge Regression":
        return Ridge(alpha=parameters["alpha"])

    if model_name == "RandomForestRegressor":
        return RandomForestRegressor(
            n_estimators=parameters["n_estimators"],
            max_depth=parameters["max_depth"],
            min_samples_leaf=parameters["min_samples_leaf"],
            random_state=RANDOM_STATE,
        )

    raise ValueError(f"Nepoznat model: {model_name}")


def create_parameter_grid():
    """Definise mali, citljiv grid hiperparametara za tuning."""
    ridge_grid = [{"alpha": alpha} for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]]

    random_forest_grid = [
        {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
        }
        for n_estimators, max_depth, min_samples_leaf in product(
            [100, 200],
            [None, 5, 10],
            [1, 2, 5],
        )
    ]

    return {
        "Ridge Regression": ridge_grid,
        "RandomForestRegressor": random_forest_grid,
    }


def format_parameters(parameters):
    """Pretvara hiperparametre u kratak tekst za CSV report."""
    return "; ".join(f"{key}={value}" for key, value in parameters.items())


def tune_on_validation(train_df, validation_df):
    """Trenira sve kombinacije i racuna samo validation metrike."""
    scenarios = create_scenarios(train_df)
    parameter_grid = create_parameter_grid()
    report_rows = []

    for scenario_name, scenario in scenarios.items():
        feature_columns = scenario["features"]
        X_train = train_df[feature_columns]
        y_train = train_df[TARGET_COLUMN]
        X_validation = validation_df[feature_columns]
        y_validation = validation_df[TARGET_COLUMN]

        for model_name, parameter_combinations in parameter_grid.items():
            for parameters in parameter_combinations:
                preprocessor = create_preprocessor(
                    scenario["numeric_features"],
                    scenario["categorical_features"],
                )
                model = create_model(model_name, parameters)
                pipeline = create_pipeline(preprocessor, model)
                pipeline.fit(X_train, y_train)

                validation_metrics = evaluate_model(
                    pipeline,
                    X_validation,
                    y_validation,
                )

                report_rows.append(
                    {
                        "scenario": scenario_name,
                        "model": model_name,
                        "parameters": format_parameters(parameters),
                        "validation_MAE": validation_metrics["MAE"],
                        "validation_RMSE": validation_metrics["RMSE"],
                        "validation_R2": validation_metrics["R2"],
                        "is_best": False,
                    }
                )

    details_report = pd.DataFrame(report_rows)
    details_report = mark_best_parameter_combinations(details_report)

    return details_report


def mark_best_parameter_combinations(details_report):
    """Oznacava najbolju kombinaciju po validation RMSE za svaki scenario/model."""
    details_report = details_report.copy()

    best_indexes = (
        details_report.groupby(["scenario", "model"])["validation_RMSE"].idxmin()
    )
    details_report.loc[best_indexes, "is_best"] = True

    return details_report


def parse_parameters(parameters_text):
    """Vraca hiperparametre iz tekstualnog zapisa u CSV reportu."""
    parameters = {}

    for item in parameters_text.split("; "):
        key, value = item.split("=")

        if value == "None":
            parameters[key] = None
        elif "." in value:
            parameters[key] = float(value)
        else:
            parameters[key] = int(value)

    return parameters


def create_tuning_summary(details_report, train_df, validation_df, test_df):
    """Za najbolje kombinacije racuna test metrike i pravi summary report."""
    scenarios = create_scenarios(train_df)
    best_rows = details_report[details_report["is_best"]].copy()
    summary_rows = []

    for _, best_row in best_rows.iterrows():
        scenario_name = best_row["scenario"]
        model_name = best_row["model"]
        parameters = parse_parameters(best_row["parameters"])
        scenario = scenarios[scenario_name]
        feature_columns = scenario["features"]

        X_train = train_df[feature_columns]
        y_train = train_df[TARGET_COLUMN]
        X_test = test_df[feature_columns]
        y_test = test_df[TARGET_COLUMN]

        preprocessor = create_preprocessor(
            scenario["numeric_features"],
            scenario["categorical_features"],
        )
        model = create_model(model_name, parameters)
        pipeline = create_pipeline(preprocessor, model)
        pipeline.fit(X_train, y_train)

        test_metrics = evaluate_model(pipeline, X_test, y_test)

        summary_rows.append(
            {
                "scenario": scenario_name,
                "model": model_name,
                "best_parameters": best_row["parameters"],
                "selection_metric": "validation_RMSE",
                "validation_MAE": best_row["validation_MAE"],
                "validation_RMSE": best_row["validation_RMSE"],
                "validation_R2": best_row["validation_R2"],
                "test_MAE": test_metrics["MAE"],
                "test_RMSE": test_metrics["RMSE"],
                "test_R2": test_metrics["R2"],
            }
        )

    return pd.DataFrame(summary_rows)


def format_report_metrics(report, metric_columns):
    """Formatira numericke kolone u reportu za citljiv CSV izlaz."""
    report = report.copy()

    if "is_best" in report.columns:
        report = report.sort_values(
            ["is_best", "scenario", "model", "validation_RMSE"],
            ascending=[False, True, True, True],
        ).reset_index(drop=True)

    report[metric_columns] = report[metric_columns].map(format_value)

    if "is_best" in report.columns:
        report["is_best"] = report["is_best"].map(format_value)

    return report


def main():
    # 1. Ucitavanje prethodno sacuvanih train, validation i test skupova.
    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # 2. Kreiranje foldera za CSV logove ako ne postoji.
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Rucni grid search: sve kombinacije se porede samo na validation skupu.
    tuning_details = tune_on_validation(train_df, validation_df)
    formatted_details = format_report_metrics(
        tuning_details,
        ["validation_MAE", "validation_RMSE", "validation_R2"],
    )
    save_csv_if_changed(
        formatted_details,
        TUNING_DETAILS_REPORT_PATH,
        PROJECT_ROOT,
    )

    # 4. Test metrike se racunaju samo za najbolje kombinacije.
    tuning_summary = create_tuning_summary(
        tuning_details,
        train_df,
        validation_df,
        test_df,
    )
    formatted_summary = format_report_metrics(
        tuning_summary,
        [
            "validation_MAE",
            "validation_RMSE",
            "validation_R2",
            "test_MAE",
            "test_RMSE",
            "test_R2",
        ],
    )
    save_csv_if_changed(
        formatted_summary,
        TUNING_SUMMARY_REPORT_PATH,
        PROJECT_ROOT,
    )

    print("Hyperparameter tuning je zavrsen.")


if __name__ == "__main__":
    main()
