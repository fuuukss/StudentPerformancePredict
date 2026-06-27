from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from csv_utils import relative_path, save_csv_if_changed


# Putanje do osnovnih foldera i fajlova u projektu.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "split"
STEP08_LOGS_DIR = PROJECT_ROOT / "data" / "logs" / "step08_feature_importance"
LOGS_DIR = PROJECT_ROOT / "data" / "logs" / "step09_top_features"
GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs" / "step09_top_features"

TRAIN_PATH = SPLIT_DIR / "train.csv"
VALIDATION_PATH = SPLIT_DIR / "validation.csv"
TEST_PATH = SPLIT_DIR / "test.csv"
FEATURE_IMPORTANCE_REPORT_PATH = STEP08_LOGS_DIR / "feature_importance_report.csv"

TOP_FEATURES_SELECTION_PATH = LOGS_DIR / "top_features_selection.csv"
TOP_FEATURES_MODEL_REPORT_PATH = LOGS_DIR / "top_features_model_comparison_report.csv"

VALIDATION_GRAPH_PATH = GRAPHS_DIR / "validation_top_features_comparison.png"
TEST_GRAPH_PATH = GRAPHS_DIR / "test_top_features_comparison.png"

TARGET_COLUMN = "G3"
RANDOM_STATE = 42
TOP_FEATURE_COUNT = 10

TOP_FEATURE_SCENARIO_SOURCES = {
    "top_features_with_G1_G2": "with_G1_G2",
    "top_features_without_G1_G2": "without_G1_G2",
}
SCENARIO_ORDER = [
    "with_G1_G2",
    "top_features_with_G1_G2",
    "without_G1_G2",
    "top_features_without_G1_G2",
]
SCENARIO_LABELS = {
    "with_G1_G2": "svi atributi\nsa G1/G2",
    "top_features_with_G1_G2": "top atributi\nsa G1/G2",
    "without_G1_G2": "svi atributi\nbez G1/G2",
    "top_features_without_G1_G2": "top atributi\nbez G1/G2",
}
MODEL_ORDER = ["DummyRegressor", "Ridge Regression", "RandomForestRegressor"]
MODEL_COLORS = {
    "DummyRegressor": "#bab0ac",
    "Ridge Regression": "#4c78a8",
    "RandomForestRegressor": "#59a14f",
}


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


def select_top_features(feature_importance_report):
    """Bira top atribute posebno za scenarije sa G1/G2 i bez G1/G2."""
    selections = []

    for top_features_scenario, source_scenario in TOP_FEATURE_SCENARIO_SOURCES.items():
        source_report = feature_importance_report[
            feature_importance_report["scenario"] == source_scenario
        ].copy()
        source_report["importance"] = source_report["importance"].astype(float)
        selected = (
            source_report.sort_values("rank")
            .head(TOP_FEATURE_COUNT)
            .reset_index(drop=True)
            .copy()
        )
        selected["top_features_scenario"] = top_features_scenario
        selected["selection_source_scenario"] = source_scenario
        selections.append(selected)

    selected_features = pd.concat(selections, ignore_index=True)

    return selected_features[
        [
            "top_features_scenario",
            "selection_source_scenario",
            "feature",
            "importance",
            "rank",
        ]
    ]


def format_top_features_selection(top_features_selection):
    """Formatira importance kolonu za stabilan CSV izlaz."""
    top_features_selection = top_features_selection.copy()
    top_features_selection["importance"] = top_features_selection["importance"].map(
        lambda value: f"{value:.6f}"
    )

    return top_features_selection


def get_top_features_by_scenario(top_features_selection):
    """Vraca recnik top atributa po top_features scenariju."""
    return {
        scenario_name: group.sort_values("rank")["feature"].tolist()
        for scenario_name, group in top_features_selection.groupby(
            "top_features_scenario"
        )
    }


def create_scenarios(df, top_features_by_scenario):
    """Definise scenarije sa svim i samo najvaznijim atributima."""
    scenario_features = {
        "with_G1_G2": get_feature_columns(df, [TARGET_COLUMN]),
        "top_features_with_G1_G2": top_features_by_scenario[
            "top_features_with_G1_G2"
        ],
        "without_G1_G2": get_feature_columns(df, ["G1", "G2", TARGET_COLUMN]),
        "top_features_without_G1_G2": top_features_by_scenario[
            "top_features_without_G1_G2"
        ],
    }
    scenarios = {}

    for scenario_name, feature_columns in scenario_features.items():
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


def create_model_comparison_report(
    train_df,
    validation_df,
    test_df,
    top_features_by_scenario,
):
    """Trenira modele po scenarijima i kreira CSV izvestaj sa metrikama."""
    scenarios = create_scenarios(train_df, top_features_by_scenario)
    report_rows = []

    for scenario_name in SCENARIO_ORDER:
        scenario = scenarios[scenario_name]
        feature_columns = scenario["features"]

        X_train = train_df[feature_columns]
        y_train = train_df[TARGET_COLUMN]
        X_validation = validation_df[feature_columns]
        y_validation = validation_df[TARGET_COLUMN]
        X_test = test_df[feature_columns]
        y_test = test_df[TARGET_COLUMN]

        for model_name, model in create_models().items():
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
                    "number_of_features": len(feature_columns),
                    "selected_features": ", ".join(feature_columns),
                    "numeric_features": len(scenario["numeric_features"]),
                    "categorical_features": len(scenario["categorical_features"]),
                    "train_rows": len(train_df),
                    "validation_rows": len(validation_df),
                    "test_rows": len(test_df),
                    "validation_MAE": validation_metrics["MAE"],
                    "validation_RMSE": validation_metrics["RMSE"],
                    "validation_R2": validation_metrics["R2"],
                    "test_MAE": test_metrics["MAE"],
                    "test_RMSE": test_metrics["RMSE"],
                    "test_R2": test_metrics["R2"],
                }
            )

    report = pd.DataFrame(report_rows)
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


def save_graph(path):
    """Cuva trenutni grafik i zatvara figuru."""
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grafik sacuvan: {relative_path(path, PROJECT_ROOT)}")


def add_value_labels(axis, bars, metric_name):
    """Dodaje kratke vrednosti iznad stubica na grafikonu."""
    for bar in bars:
        value = bar.get_height()
        label = f"{value:.2f}" if metric_name != "R2" else f"{value:.3f}"
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )


def get_metric_values(report, metric_column):
    """Priprema tabelu vrednosti za izabranu metriku."""
    metric_table = report.pivot(
        index="scenario",
        columns="model",
        values=metric_column,
    )

    return metric_table.loc[SCENARIO_ORDER, MODEL_ORDER]


def draw_grouped_metric_bars(axis, report, metric_column, metric_name):
    """Crta grouped bar chart za jednu metriku."""
    metric_table = get_metric_values(report, metric_column)
    x_positions = range(len(SCENARIO_ORDER))
    bar_width = 0.20

    for model_index, model_name in enumerate(MODEL_ORDER):
        offset = (model_index - 1) * bar_width
        values = metric_table[model_name].tolist()
        bars = axis.bar(
            [position + offset for position in x_positions],
            values,
            width=bar_width,
            label=model_name,
            color=MODEL_COLORS[model_name],
            edgecolor="black",
            linewidth=0.6,
        )
        add_value_labels(axis, bars, metric_name)

    axis.set_title(metric_name)
    axis.set_xticks(list(x_positions))
    axis.set_xticklabels([SCENARIO_LABELS[scenario] for scenario in SCENARIO_ORDER])
    axis.grid(axis="y", alpha=0.25)

    if metric_name == "R2":
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_ylabel("Veca vrednost je bolja")
    else:
        axis.set_ylabel("Manja vrednost je bolja")


def create_metrics_comparison_graph(report, metric_prefix, path, title):
    """Kreira graficko poredjenje MAE, RMSE i R2 metrika."""
    report = report.copy()
    for metric_name in ["MAE", "RMSE", "R2"]:
        report[f"{metric_prefix}_{metric_name}"] = report[
            f"{metric_prefix}_{metric_name}"
        ].astype(float)

    metrics = [
        (f"{metric_prefix}_MAE", "MAE"),
        (f"{metric_prefix}_RMSE", "RMSE"),
        (f"{metric_prefix}_R2", "R2"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(21, 5.5))
    fig.suptitle(title, fontsize=14)

    for axis, (metric_column, metric_name) in zip(axes, metrics):
        draw_grouped_metric_bars(axis, report, metric_column, metric_name)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(bottom=0.30, top=0.82, wspace=0.22)

    save_graph(path)


def create_top_features_graphs(report):
    """Kreira grafike za poredjenje scenarija sa top atributima."""
    create_metrics_comparison_graph(
        report,
        "validation",
        VALIDATION_GRAPH_PATH,
        "Poredjenje top_features scenarija na validation skupu",
    )
    create_metrics_comparison_graph(
        report,
        "test",
        TEST_GRAPH_PATH,
        "Poredjenje top_features scenarija na test skupu",
    )


def print_top_feature_summary(top_features_selection):
    """Ispisuje izabrane top atribute."""
    for scenario_name, group in top_features_selection.groupby("top_features_scenario"):
        selected_features = ", ".join(group.sort_values("rank")["feature"].tolist())
        print(f"Izabrani atributi za {scenario_name}: {selected_features}")


def main():
    # 1. Ucitavanje prethodno sacuvanih train, validation i test skupova.
    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # 2. Ucitavanje Step 08 feature importance report-a i izbor top atributa.
    feature_importance_report = pd.read_csv(FEATURE_IMPORTANCE_REPORT_PATH)
    top_features_selection = select_top_features(feature_importance_report)
    top_features_by_scenario = get_top_features_by_scenario(top_features_selection)

    # 3. Kreiranje foldera za CSV logove i grafike ako ne postoje.
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    # 4. Cuvanje liste izabranih top atributa.
    save_csv_if_changed(
        format_top_features_selection(top_features_selection),
        TOP_FEATURES_SELECTION_PATH,
        PROJECT_ROOT,
    )

    # 5. Treniranje dogovorenih modela po scenarijima.
    model_comparison_report = create_model_comparison_report(
        train_df,
        validation_df,
        test_df,
        top_features_by_scenario,
    )
    save_csv_if_changed(
        model_comparison_report,
        TOP_FEATURES_MODEL_REPORT_PATH,
        PROJECT_ROOT,
    )

    # 6. Graficko poredjenje scenarija.
    create_top_features_graphs(model_comparison_report)
    print_top_feature_summary(top_features_selection)

    print("Top features scenario je zavrsen.")


if __name__ == "__main__":
    main()
