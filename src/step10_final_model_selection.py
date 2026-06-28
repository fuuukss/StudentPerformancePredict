from math import sqrt
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from csv_utils import relative_path, save_csv_if_changed


# Putanje do osnovnih foldera i fajlova u projektu.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "split"
STEP09_LOGS_DIR = PROJECT_ROOT / "data" / "logs" / "step09_top_features"
LOGS_DIR = PROJECT_ROOT / "data" / "logs" / "step10_final_model_selection"
GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs" / "step10_final_model_selection"
FINAL_MODEL_DIR = PROJECT_ROOT / "models" / "final"

TRAIN_PATH = SPLIT_DIR / "train.csv"
VALIDATION_PATH = SPLIT_DIR / "validation.csv"
TEST_PATH = SPLIT_DIR / "test.csv"
TOP_FEATURES_WITH_G1_G2_PATH = STEP09_LOGS_DIR / "top_features_with_G1_G2.csv"

FINAL_MODEL_REPORT_PATH = LOGS_DIR / "final_model_report.csv"
FINAL_MODEL_PREDICTIONS_PATH = LOGS_DIR / "final_model_predictions.csv"
FINAL_MODEL_PATH = FINAL_MODEL_DIR / "final_model.joblib"

ACTUAL_VS_PREDICTED_GRAPH_PATH = GRAPHS_DIR / "final_model_actual_vs_predicted.png"
RESIDUALS_GRAPH_PATH = GRAPHS_DIR / "final_model_residuals.png"
METRICS_GRAPH_PATH = GRAPHS_DIR / "final_model_metrics.png"

TARGET_COLUMN = "G3"
FINAL_SCENARIO = "top_features_with_G1_G2"
FINAL_MODEL_NAME = "Ridge Regression"
FINAL_MODEL_VERSION = "top_features"
FINAL_MODEL_PARAMETERS = "alpha=0.01"
FINAL_SELECTION_REASON = (
    "Izabran je Ridge Regression na top_features_with_G1_G2 scenariju jer daje "
    "jak validation rezultat, najbolji test RMSE medju poredjenim kandidatima, "
    "koristi samo 10 atributa i jednostavniji je za tumacenje od Random Forest "
    "modela. Parametar alpha=0.01 je izabran tuningom top_features scenarija. "
    "Prakticno ogranicenje je da koristi G1 i G2."
)


def format_value(value):
    """Formatira vrednosti za citljiviji CSV izlaz."""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))

        return f"{value:.4f}"

    return str(value)


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


def create_final_pipeline(train_validation_df, feature_columns):
    """Kreira finalni pipeline za izabrani Ridge model."""
    numeric_features, categorical_features = split_feature_types(
        train_validation_df,
        feature_columns,
    )
    preprocessor = create_preprocessor(numeric_features, categorical_features)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=0.01)),
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


def get_final_feature_columns():
    """Vraca listu atributa za izabrani finalni top_features scenario."""
    final_features = pd.read_csv(TOP_FEATURES_WITH_G1_G2_PATH)
    final_features["rank"] = final_features["rank"].astype(int)

    return final_features.sort_values("rank")["feature"].tolist()


def train_and_save_final_model(train_df, validation_df, test_df, feature_columns):
    """Trenira finalni model na train+validation skupu i cuva ga kao joblib."""
    train_validation_df = pd.concat([train_df, validation_df], ignore_index=True)

    X_train_validation = train_validation_df[feature_columns]
    y_train_validation = train_validation_df[TARGET_COLUMN]
    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]

    pipeline = create_final_pipeline(train_validation_df, feature_columns)
    pipeline.fit(X_train_validation, y_train_validation)
    test_predictions = pipeline.predict(X_test)
    final_test_metrics = calculate_metrics(y_test, test_predictions)

    predictions_report = pd.DataFrame(
        {
            "actual_G3": y_test,
            "predicted_G3": test_predictions,
            "residual": y_test - test_predictions,
        }
    ).reset_index(drop=True)

    final_model_bundle = {
        "pipeline": pipeline,
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "scenario": FINAL_SCENARIO,
        "model": FINAL_MODEL_NAME,
        "model_version": FINAL_MODEL_VERSION,
        "parameters": FINAL_MODEL_PARAMETERS,
        "trained_on": "train+validation",
    }
    joblib.dump(final_model_bundle, FINAL_MODEL_PATH)

    return final_test_metrics, predictions_report


def create_final_model_report(final_test_metrics):
    """Kreira report sa finalnim izborom modela."""
    return pd.DataFrame(
        [
            {
                "final_model_path": FINAL_MODEL_PATH.relative_to(
                    PROJECT_ROOT
                ).as_posix(),
                "scenario": FINAL_SCENARIO,
                "model": FINAL_MODEL_NAME,
                "parameters": FINAL_MODEL_PARAMETERS,
                "final_test_MAE": final_test_metrics["MAE"],
                "final_test_RMSE": final_test_metrics["RMSE"],
                "final_test_R2": final_test_metrics["R2"],
                "selection_reason": FINAL_SELECTION_REASON,
            }
        ]
    )


def format_report(report):
    """Formatira numericke kolone za stabilan CSV izlaz."""
    report = report.copy()

    for column in report.columns:
        if column.endswith(("MAE", "RMSE", "R2", "G3", "residual")):
            report[column] = report[column].astype(float).map(format_value)

    return report


def save_graph(path):
    """Cuva trenutni grafik i zatvara figuru."""
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grafik sacuvan: {relative_path(path, PROJECT_ROOT)}")


def create_actual_vs_predicted_graph(predictions_report):
    """Kreira scatter grafik stvarnih i predvidjenih G3 vrednosti."""
    fig, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(
        predictions_report["actual_G3"],
        predictions_report["predicted_G3"],
        alpha=0.75,
        color="#4c78a8",
        edgecolor="black",
        linewidth=0.4,
    )
    axis.plot(
        [0, 20],
        [0, 20],
        color="#e15759",
        linewidth=1.5,
        label="idealna predikcija",
    )
    axis.set_title("Finalni model: stvarna vs predvidjena G3")
    axis.set_xlabel("Stvarna G3")
    axis.set_ylabel("Predvidjena G3")
    axis.set_xlim(0, 20)
    axis.set_ylim(0, 20)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)

    save_graph(ACTUAL_VS_PREDICTED_GRAPH_PATH)


def create_residuals_graph(predictions_report):
    """Kreira histogram gresaka finalnog modela."""
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(
        predictions_report["residual"],
        bins=14,
        color="#59a14f",
        edgecolor="black",
        alpha=0.85,
    )
    axis.axvline(0, color="#e15759", linewidth=1.5)
    axis.set_title("Finalni model: raspodela reziduala")
    axis.set_xlabel("Greska: stvarna G3 - predvidjena G3")
    axis.set_ylabel("Broj ucenika")
    axis.grid(axis="y", alpha=0.25)

    save_graph(RESIDUALS_GRAPH_PATH)


def create_metrics_graph(final_test_metrics):
    """Kreira bar grafik finalnih test metrika."""
    metrics = pd.DataFrame(
        {
            "metric": ["MAE", "RMSE", "R2"],
            "value": [
                final_test_metrics["MAE"],
                final_test_metrics["RMSE"],
                final_test_metrics["R2"],
            ],
        }
    )

    fig, axis = plt.subplots(figsize=(7, 5))
    bars = axis.bar(
        metrics["metric"],
        metrics["value"],
        color=["#4c78a8", "#f28e2b", "#59a14f"],
        edgecolor="black",
        linewidth=0.6,
    )
    for bar in bars:
        value = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axis.set_title("Finalni model: test metrike")
    axis.set_ylabel("Vrednost metrike")
    axis.grid(axis="y", alpha=0.25)

    save_graph(METRICS_GRAPH_PATH)


def create_final_model_graphs(final_test_metrics, predictions_report):
    """Kreira grafike za finalni model."""
    create_actual_vs_predicted_graph(predictions_report)
    create_residuals_graph(predictions_report)
    create_metrics_graph(final_test_metrics)


def main():
    # 1. Ucitavanje splitova i kreiranje output foldera.
    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)
    test_df = pd.read_csv(TEST_PATH)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Treniranje finalnog modela na train+validation skupu i cuvanje modela.
    final_feature_columns = get_final_feature_columns()
    final_test_metrics, predictions_report = train_and_save_final_model(
        train_df,
        validation_df,
        test_df,
        final_feature_columns,
    )

    # 3. Cuvanje predikcija i grafika finalnog modela.
    save_csv_if_changed(
        format_report(predictions_report),
        FINAL_MODEL_PREDICTIONS_PATH,
        PROJECT_ROOT,
    )
    create_final_model_graphs(final_test_metrics, predictions_report)

    # 4. Cuvanje finalnog report-a.
    final_model_report = create_final_model_report(final_test_metrics)
    save_csv_if_changed(
        format_report(final_model_report),
        FINAL_MODEL_REPORT_PATH,
        PROJECT_ROOT,
    )

    print(f"Finalni model sacuvan: {FINAL_MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print("Izbor finalnog modela je zavrsen.")


if __name__ == "__main__":
    main()
