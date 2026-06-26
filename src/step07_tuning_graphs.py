from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from csv_utils import relative_path


# Putanje do osnovnih foldera i fajlova u projektu.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "logs"
    / "step04_model_training"
    / "model_comparison_report.csv"
)
TUNED_RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "logs"
    / "step06_hyperparameter_tuning"
    / "hyperparameter_tuning_summary.csv"
)
GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs" / "step07_tuning_graphs"

VALIDATION_TUNING_GRAPH_PATH = GRAPHS_DIR / "default_vs_tuned_validation.png"
TEST_TUNING_GRAPH_PATH = GRAPHS_DIR / "default_vs_tuned_test.png"

SCENARIO_ORDER = ["with_G1_G2", "without_G1_G2"]
SCENARIO_LABELS = {
    "with_G1_G2": "sa G1/G2",
    "without_G1_G2": "bez G1/G2",
}
MODEL_ORDER = ["Ridge Regression", "RandomForestRegressor"]
VERSION_ORDER = ["default", "tuned"]
VERSION_COLORS = {
    "default": "#bab0ac",
    "tuned": "#4c78a8",
}


def save_graph(path):
    """Cuva trenutni grafik i zatvara figuru."""
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grafik sacuvan: {relative_path(path, PROJECT_ROOT)}")


def prepare_comparison_report(default_results, tuned_results):
    """Spaja default i tuned rezultate u jedan DataFrame za grafike."""
    default_filtered = default_results[
        default_results["model"].isin(MODEL_ORDER)
    ].copy()
    tuned_filtered = tuned_results[tuned_results["model"].isin(MODEL_ORDER)].copy()

    default_filtered["version"] = "default"
    default_filtered["parameters"] = "default"
    tuned_filtered["version"] = "tuned"
    tuned_filtered["parameters"] = tuned_filtered["best_parameters"]

    columns = [
        "scenario",
        "model",
        "version",
        "parameters",
        "validation_MAE",
        "validation_RMSE",
        "validation_R2",
        "test_MAE",
        "test_RMSE",
        "test_R2",
    ]

    comparison = pd.concat(
        [default_filtered[columns], tuned_filtered[columns]],
        ignore_index=True,
    )

    for metric_column in [
        "validation_MAE",
        "validation_RMSE",
        "validation_R2",
        "test_MAE",
        "test_RMSE",
        "test_R2",
    ]:
        comparison[metric_column] = comparison[metric_column].astype(float)

    return comparison


def add_value_labels(axis, bars, metric_name):
    """Dodaje kratke vrednosti iznad stubica na grafikonu."""
    for bar in bars:
        value = bar.get_height()
        label = f"{value:.2f}" if metric_name in ["MAE", "RMSE"] else f"{value:.3f}"
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )


def get_metric_values(comparison, scenario_name, model_name, metric_column):
    """Vraca default i tuned vrednosti za jednu metriku."""
    filtered = comparison[
        (comparison["scenario"] == scenario_name)
        & (comparison["model"] == model_name)
    ]
    metric_values = filtered.set_index("version")[metric_column]

    return [metric_values.loc[version] for version in VERSION_ORDER]


def draw_metric_subplot(axis, comparison, metric_column, metric_name):
    """Crta poredjenje default i tuned rezultata za jednu metriku."""
    labels = []
    x_positions = []
    current_position = 0
    bar_width = 0.34

    for scenario_name in SCENARIO_ORDER:
        for model_name in MODEL_ORDER:
            labels.append(f"{SCENARIO_LABELS[scenario_name]}\n{model_name}")
            x_positions.append(current_position)

            values = get_metric_values(
                comparison,
                scenario_name,
                model_name,
                metric_column,
            )

            for version_index, version_name in enumerate(VERSION_ORDER):
                offset = (version_index - 0.5) * bar_width
                bars = axis.bar(
                    current_position + offset,
                    values[version_index],
                    width=bar_width,
                    color=VERSION_COLORS[version_name],
                    edgecolor="black",
                    linewidth=0.6,
                    label=version_name if current_position == 0 else None,
                )
                add_value_labels(axis, bars, metric_name)

            current_position += 1

        current_position += 0.4

    axis.set_title(metric_name)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels, rotation=12, ha="right")
    axis.grid(axis="y", alpha=0.25)

    if metric_name == "R2":
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_ylabel("Veca vrednost je bolja")
    else:
        axis.set_ylabel("Manja vrednost je bolja")


def create_tuning_comparison_graph(comparison, metric_prefix, path, title):
    """Kreira graficko poredjenje default i tuned MAE/RMSE/R2 metrika."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=14)

    draw_metric_subplot(
        axes[0],
        comparison,
        f"{metric_prefix}_MAE",
        "MAE",
    )
    draw_metric_subplot(
        axes[1],
        comparison,
        f"{metric_prefix}_RMSE",
        "RMSE",
    )
    draw_metric_subplot(
        axes[2],
        comparison,
        f"{metric_prefix}_R2",
        "R2",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(bottom=0.32, top=0.82, wspace=0.22)

    save_graph(path)


def main():
    # 1. Ucitavanje default rezultata iz Step 04 i tuned rezultata iz Step 06.
    default_results = pd.read_csv(DEFAULT_RESULTS_PATH)
    tuned_results = pd.read_csv(TUNED_RESULTS_PATH)

    # 2. Kreiranje foldera za grafike ako ne postoji.
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Priprema objedinjene tabele za poredjenje.
    comparison = prepare_comparison_report(default_results, tuned_results)

    # 4. Graficko poredjenje na validation skupu.
    create_tuning_comparison_graph(
        comparison,
        "validation",
        VALIDATION_TUNING_GRAPH_PATH,
        "Default vs tuned modeli na validation skupu",
    )

    # 5. Graficko poredjenje na test skupu.
    create_tuning_comparison_graph(
        comparison,
        "test",
        TEST_TUNING_GRAPH_PATH,
        "Default vs tuned modeli na test skupu",
    )

    print("Grafici za poredjenje default i tuned modela su zavrseni.")


if __name__ == "__main__":
    main()
