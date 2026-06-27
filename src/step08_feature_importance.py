from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from csv_utils import relative_path, save_csv_if_changed


# Putanje do osnovnih foldera i fajlova u projektu.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "split"
LOGS_DIR = PROJECT_ROOT / "data" / "logs" / "step08_feature_importance"
GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs" / "step08_feature_importance"

TRAIN_PATH = SPLIT_DIR / "train.csv"
VALIDATION_PATH = SPLIT_DIR / "validation.csv"
TEST_PATH = SPLIT_DIR / "test.csv"
WITH_G1_G2_REPORT_PATH = LOGS_DIR / "feature_importance_with_G1_G2.csv"
WITHOUT_G1_G2_REPORT_PATH = LOGS_DIR / "feature_importance_without_G1_G2.csv"

WITH_G1_G2_GRAPH_PATH = GRAPHS_DIR / "feature_importance_with_G1_G2.png"
WITHOUT_G1_G2_GRAPH_PATH = GRAPHS_DIR / "feature_importance_without_G1_G2.png"

TARGET_COLUMN = "G3"
RANDOM_STATE = 42
TOP_FEATURES_TO_PLOT = 15

SCENARIO_ORDER = ["with_G1_G2", "without_G1_G2"]
SCENARIO_LABELS = {
    "with_G1_G2": "sa G1/G2",
    "without_G1_G2": "bez G1/G2",
}
SCENARIO_PARAMETERS = {
    "with_G1_G2": {
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_leaf": 5,
    },
    "without_G1_G2": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_leaf": 2,
    },
}


def format_value(value):
    """Formatira vrednosti za citljiviji CSV izlaz."""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))

        return f"{value:.6f}"

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


def create_random_forest(parameters):
    """Kreira tuned RandomForestRegressor za dati scenario."""
    return RandomForestRegressor(
        n_estimators=parameters["n_estimators"],
        max_depth=parameters["max_depth"],
        min_samples_leaf=parameters["min_samples_leaf"],
        random_state=RANDOM_STATE,
    )


def create_pipeline(preprocessor, model):
    """Kreira pipeline koji prvo radi preprocessing, pa trenira model."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def get_transformed_feature_mapping(preprocessor, numeric_features, categorical_features):
    """Vraca mapiranje transformisanih feature-a na originalne atribute."""
    transformed_features = []

    for feature_name in numeric_features:
        transformed_features.append(
            {
                "transformed_feature": feature_name,
                "original_feature": feature_name,
            }
        )

    encoder = preprocessor.named_transformers_["categorical"]
    encoded_feature_names = encoder.get_feature_names_out(categorical_features)
    encoded_feature_index = 0

    for categorical_feature, categories in zip(
        categorical_features,
        encoder.categories_,
    ):
        for _ in categories:
            transformed_features.append(
                {
                    "transformed_feature": encoded_feature_names[encoded_feature_index],
                    "original_feature": categorical_feature,
                }
            )
            encoded_feature_index += 1

    return pd.DataFrame(transformed_features)


def train_feature_importance_pipeline(train_df, scenario_name, scenario):
    """Trenira tuned Random Forest i vraca pipeline."""
    feature_columns = scenario["features"]
    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]

    preprocessor = create_preprocessor(
        scenario["numeric_features"],
        scenario["categorical_features"],
    )
    model = create_random_forest(SCENARIO_PARAMETERS[scenario_name])
    pipeline = create_pipeline(preprocessor, model)
    pipeline.fit(X_train, y_train)

    return pipeline


def create_detailed_importance_rows(scenario_name, scenario, pipeline):
    """Kreira detaljan report za sve transformisane feature-e."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_mapping = get_transformed_feature_mapping(
        preprocessor,
        scenario["numeric_features"],
        scenario["categorical_features"],
    )

    detailed_report = feature_mapping.copy()
    detailed_report["importance"] = model.feature_importances_
    detailed_report["scenario"] = scenario_name
    detailed_report = detailed_report.sort_values(
        ["importance", "transformed_feature"],
        ascending=[False, True],
    ).reset_index(drop=True)
    detailed_report["rank"] = detailed_report.index + 1

    return detailed_report[
        [
            "scenario",
            "transformed_feature",
            "original_feature",
            "importance",
            "rank",
        ]
    ]


def aggregate_importance(detailed_report):
    """Sabira importance vrednosti dummy kolona nazad na originalne atribute."""
    aggregated = (
        detailed_report.groupby(["scenario", "original_feature"], as_index=False)[
            "importance"
        ]
        .sum()
        .rename(columns={"original_feature": "feature"})
    )
    aggregated = aggregated.sort_values(
        ["scenario", "importance", "feature"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    aggregated["rank"] = aggregated.groupby("scenario").cumcount() + 1

    return aggregated[["scenario", "feature", "importance", "rank"]]


def create_feature_importance_reports(train_df):
    """Trenira Random Forest po scenarijima i kreira importance report-e."""
    scenarios = create_scenarios(train_df)
    detailed_reports = []

    for scenario_name in SCENARIO_ORDER:
        scenario = scenarios[scenario_name]
        pipeline = train_feature_importance_pipeline(train_df, scenario_name, scenario)
        detailed_report = create_detailed_importance_rows(
            scenario_name,
            scenario,
            pipeline,
        )
        detailed_reports.append(detailed_report)

    detailed_report = pd.concat(detailed_reports, ignore_index=True)
    aggregated_report = aggregate_importance(detailed_report)

    return aggregated_report, detailed_report


def format_report(report):
    """Formatira importance kolonu za stabilan CSV izlaz."""
    report = report.copy()
    report["importance"] = report["importance"].map(format_value)

    return report


def get_scenario_report(report, scenario_name):
    """Vraca report za jedan scenario bez interne scenario kolone."""
    scenario_report = report[report["scenario"] == scenario_name].copy()
    scenario_report = scenario_report.sort_values("rank").reset_index(drop=True)

    return scenario_report[["rank", "feature", "importance"]]


def save_graph(path):
    """Cuva trenutni grafik i zatvara figuru."""
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grafik sacuvan: {relative_path(path, PROJECT_ROOT)}")


def add_value_labels(axis, bars):
    """Dodaje importance vrednosti na horizontalne stubice."""
    for bar in bars:
        value = bar.get_width()
        axis.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f" {value:.3f}",
            va="center",
            fontsize=8,
        )


def draw_feature_importance_subplot(axis, report, scenario_name):
    """Crta top feature importance za jedan scenario."""
    scenario_report = report[report["scenario"] == scenario_name]
    plot_data = scenario_report.head(TOP_FEATURES_TO_PLOT).sort_values("importance")

    bars = axis.barh(
        plot_data["feature"],
        plot_data["importance"],
        color="#4c78a8",
        edgecolor="black",
        linewidth=0.6,
    )
    add_value_labels(axis, bars)

    axis.set_title(f"Top atributi - {SCENARIO_LABELS[scenario_name]}")
    axis.set_xlabel("Feature importance")
    axis.grid(axis="x", alpha=0.25)


def create_single_scenario_graph(report, scenario_name, path):
    """Kreira feature importance grafik za jedan scenario."""
    fig, axis = plt.subplots(figsize=(10, 7))
    draw_feature_importance_subplot(axis, report, scenario_name)
    fig.suptitle("Random Forest feature importance", fontsize=14)
    fig.tight_layout()

    save_graph(path)


def create_feature_importance_graphs(report):
    """Kreira pojedinacne i zajednicki feature importance grafik."""
    create_single_scenario_graph(
        report,
        "with_G1_G2",
        WITH_G1_G2_GRAPH_PATH,
    )
    create_single_scenario_graph(
        report,
        "without_G1_G2",
        WITHOUT_G1_G2_GRAPH_PATH,
    )


def print_top_features(report):
    """Ispisuje najvaznije atribute po scenariju."""
    for scenario_name in SCENARIO_ORDER:
        scenario_report = report[report["scenario"] == scenario_name].head(5)
        top_features = ", ".join(scenario_report["feature"].tolist())
        print(f"Top atributi za {scenario_name}: {top_features}")


def main():
    # 1. Ucitavanje prethodno sacuvanih train, validation i test skupova.
    train_df = pd.read_csv(TRAIN_PATH)
    pd.read_csv(VALIDATION_PATH)
    pd.read_csv(TEST_PATH)

    # 2. Kreiranje foldera za CSV logove i grafike ako ne postoje.
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Treniranje tuned Random Forest modela i izvlacenje importance vrednosti.
    feature_importance_report, detailed_feature_importance_report = (
        create_feature_importance_reports(train_df)
    )

    # 4. Cuvanje odvojenih feature importance reporta po scenarijima.
    save_csv_if_changed(
        format_report(get_scenario_report(feature_importance_report, "with_G1_G2")),
        WITH_G1_G2_REPORT_PATH,
        PROJECT_ROOT,
    )
    save_csv_if_changed(
        format_report(get_scenario_report(feature_importance_report, "without_G1_G2")),
        WITHOUT_G1_G2_REPORT_PATH,
        PROJECT_ROOT,
    )

    # 5. Kreiranje grafika za oba scenarija.
    create_feature_importance_graphs(feature_importance_report)
    print_top_features(feature_importance_report)

    print("Feature importance analiza je zavrsena.")


if __name__ == "__main__":
    main()
