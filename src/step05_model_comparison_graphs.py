from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from csv_utils import relative_path


# Putanje do osnovnih foldera i fajlova u projektu.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_COMPARISON_REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "logs"
    / "step04_model_training"
    / "model_comparison_report.csv"
)
GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs" / "step05_model_comparison"

VALIDATION_METRICS_GRAPH_PATH = GRAPHS_DIR / "validation_metrics_comparison.png"
TEST_METRICS_GRAPH_PATH = GRAPHS_DIR / "test_metrics_comparison.png"

SCENARIO_ORDER = ["with_G1_G2", "without_G1_G2"]
SCENARIO_LABELS = ["sa G1/G2", "bez G1/G2"]
MODEL_ORDER = ["DummyRegressor", "Ridge Regression", "RandomForestRegressor"]
MODEL_COLORS = {
    "DummyRegressor": "#bab0ac",
    "Ridge Regression": "#4c78a8",
    "RandomForestRegressor": "#59a14f",
}


def save_graph(path):
    """Cuva trenutni grafik i zatvara figuru."""
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grafik sacuvan: {relative_path(path, PROJECT_ROOT)}")


def get_metric_values(report, metric_column):
    """Priprema tabelu vrednosti za izabranu metriku."""
    metric_table = report.pivot(
        index="scenario",
        columns="model",
        values=metric_column,
    )

    return metric_table.loc[SCENARIO_ORDER, MODEL_ORDER]


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


def draw_grouped_metric_bars(axis, report, metric_column, metric_name):
    """Crta grouped bar chart za jednu metriku."""
    metric_table = get_metric_values(report, metric_column)
    x_positions = range(len(SCENARIO_ORDER))
    bar_width = 0.24

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
    axis.set_xticklabels(SCENARIO_LABELS)
    axis.grid(axis="y", alpha=0.25)

    if metric_name == "R2":
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_ylabel("Veca vrednost je bolja")
    else:
        axis.set_ylabel("Manja vrednost je bolja")


def create_metrics_comparison_graph(report, metric_prefix, path, title):
    """Kreira graficko poredjenje MAE, RMSE i R2 metrika."""
    metrics = [
        (f"{metric_prefix}_MAE", "MAE"),
        (f"{metric_prefix}_RMSE", "RMSE"),
        (f"{metric_prefix}_R2", "R2"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
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
    fig.subplots_adjust(bottom=0.30, top=0.82)

    save_graph(path)


def main():
    # 1. Ucitavanje rezultata osnovnog poredjenja modela iz Step 04.
    report = pd.read_csv(MODEL_COMPARISON_REPORT_PATH)

    # 2. Kreiranje foldera za grafike ako ne postoji.
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Graficki prikaz validation metrika.
    create_metrics_comparison_graph(
        report,
        "validation",
        VALIDATION_METRICS_GRAPH_PATH,
        "Poredjenje modela na validation skupu",
    )

    # 4. Graficki prikaz test metrika.
    create_metrics_comparison_graph(
        report,
        "test",
        TEST_METRICS_GRAPH_PATH,
        "Poredjenje modela na test skupu",
    )

    print("Grafici za poredjenje modela su zavrseni.")


if __name__ == "__main__":
    main()
