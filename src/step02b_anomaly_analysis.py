from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

from csv_utils import relative_path, save_csv_if_changed


# Podešavanje UTF-8 ispisa je korisno na Windows konzolama.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Putanje do osnovnih foldera i fajlova u projektu.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student-por.csv"
LOGS_DIR = PROJECT_ROOT / "data" / "logs" / "step02b_anomaly_analysis"
GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs" / "step02b_anomaly_analysis"

IQR_REPORT_PATH = LOGS_DIR / "iqr_outlier_report.csv"
CATEGORICAL_REPORT_PATH = LOGS_DIR / "categorical_value_report.csv"
SUMMARY_PATH = LOGS_DIR / "anomaly_summary.csv"

NUMERIC_BOXPLOTS_PATH = GRAPHS_DIR / "numeric_boxplots.png"
ABSENCES_VS_G3_OUTLIERS_PATH = GRAPHS_DIR / "absences_vs_g3_outliers.png"
CATEGORICAL_UNIQUE_COUNTS_PATH = GRAPHS_DIR / "categorical_unique_counts.png"

RARE_CATEGORY_THRESHOLD = 5
BOXPLOT_COLUMNS = ["G1", "G2", "G3", "absences", "failures", "age"]
OUTLIER_SCATTER_COLUMNS = ["absences", "G3"]
ANALYSIS_CONCLUSION = (
    "Ekstremne numeričke vrednosti i retke kategorije nisu automatski "
    "greške i nisu uklonjene jer mogu predstavljati realne učenike i "
    "realne situacije u dataset-u."
)


def format_value(value):
    """Formatira vrednosti za čitljiviji CSV izlaz."""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))

        return f"{value:.2f}"

    return str(value)


def save_graph(path):
    """Čuva trenutni grafik i zatvara figuru."""
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Grafik sačuvan: {relative_path(path, PROJECT_ROOT)}")


def calculate_iqr_bounds(series):
    """Računa Q1, Q3, IQR i granice za jednu numeričku kolonu."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return q1, q3, iqr, lower_bound, upper_bound


def create_iqr_report(df):
    """Kreira IQR izveštaj za sve numeričke atribute."""
    numeric_df = df.select_dtypes(include="number")
    report_rows = []

    for column in numeric_df.columns:
        q1, q3, iqr, lower_bound, upper_bound = calculate_iqr_bounds(numeric_df[column])
        below_lower_count = int((numeric_df[column] < lower_bound).sum())
        above_upper_count = int((numeric_df[column] > upper_bound).sum())

        report_rows.append(
            {
                "attribute": column,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "below_lower_count": below_lower_count,
                "above_upper_count": above_upper_count,
                "total_potential_outliers": below_lower_count + above_upper_count,
            }
        )

    return pd.DataFrame(report_rows)


def get_iqr_outlier_mask(df, columns):
    """Vraća redove koji su IQR ekstremi u bar jednoj izabranoj koloni."""
    outlier_mask = pd.Series(False, index=df.index)

    for column in columns:
        _, _, _, lower_bound, upper_bound = calculate_iqr_bounds(df[column])
        outlier_mask = outlier_mask | (df[column] < lower_bound) | (df[column] > upper_bound)

    return outlier_mask


def format_list(values):
    """Formatira listu vrednosti za čitljiviji CSV izlaz."""
    if not values:
        return "Nema"

    return "; ".join(str(value) for value in values)


def create_categorical_report(df):
    """Kreira izveštaj o jedinstvenim i retkim kategorijskim vrednostima."""
    categorical_df = df.select_dtypes(exclude="number")
    report_rows = []

    for column in categorical_df.columns:
        value_counts = categorical_df[column].value_counts(dropna=False)
        rare_counts = value_counts[value_counts <= RARE_CATEGORY_THRESHOLD]

        report_rows.append(
            {
                "attribute": column,
                "unique_value_count": int(value_counts.size),
                "unique_values": format_list(value_counts.index.tolist()),
                "most_common_value": value_counts.index[0],
                "most_common_count": int(value_counts.iloc[0]),
                "rarest_value": value_counts.sort_values().index[0],
                "rarest_count": int(value_counts.min()),
                "rare_category_count": int(rare_counts.size),
                "rare_categories": format_list(rare_counts.index.tolist()),
            }
        )

    return pd.DataFrame(report_rows)


def create_summary(df, iqr_report, categorical_report, scatter_outlier_count):
    """Kreira kratak CSV sažetak najvažnijih nalaza step02b analize."""
    summary_rows = [
        {"Opis": "Ukupan broj redova u dataset-u", "Vrednost": len(df)},
        {
            "Opis": "Broj numeričkih atributa analiziranih IQR metodom",
            "Vrednost": len(iqr_report),
        },
        {
            "Opis": "Broj atributa kod kojih IQR pronalazi ekstremne vrednosti",
            "Vrednost": int((iqr_report["total_potential_outliers"] > 0).sum()),
        },
        {
            "Opis": "Ukupan broj potencijalnih IQR outlier vrednosti",
            "Vrednost": int(iqr_report["total_potential_outliers"].sum()),
        },
        {
            "Opis": "Broj redova označenih kao IQR ekstrem za absences ili G3",
            "Vrednost": scatter_outlier_count,
        },
        {"Opis": "Broj učenika sa G3 = 0", "Vrednost": int((df["G3"] == 0).sum())},
        {"Opis": "Maksimalan broj izostanaka", "Vrednost": df["absences"].max()},
        {
            "Opis": "Broj kategorijskih atributa",
            "Vrednost": len(categorical_report),
        },
        {
            "Opis": "Broj kategorijskih atributa sa retkim kategorijama",
            "Vrednost": int((categorical_report["rare_category_count"] > 0).sum()),
        },
        {
            "Opis": "Prag za retke kategorije",
            "Vrednost": RARE_CATEGORY_THRESHOLD,
        },
        {"Opis": "Zaključak", "Vrednost": ANALYSIS_CONCLUSION},
    ]

    summary = pd.DataFrame(summary_rows, columns=["Opis", "Vrednost"])
    summary["Vrednost"] = summary["Vrednost"].apply(format_value)

    return summary


def create_numeric_boxplots(df):
    """Kreira boxplot za izabrane numeričke atribute."""
    columns = [column for column in BOXPLOT_COLUMNS if column in df.columns]

    plt.figure(figsize=(10, 6))
    df[columns].boxplot()
    plt.title("Boxplot numeričkih atributa")
    plt.ylabel("Vrednost")
    plt.grid(alpha=0.25)
    save_graph(NUMERIC_BOXPLOTS_PATH)


def create_absences_vs_g3_outliers_scatter(df, outlier_mask):
    """Kreira scatter plot izostanaka i G3 ocene sa označenim IQR ekstremima."""
    normal_rows = df[~outlier_mask]
    outlier_rows = df[outlier_mask]

    plt.figure(figsize=(8, 5))
    plt.scatter(
        normal_rows["absences"],
        normal_rows["G3"],
        alpha=0.6,
        color="#4c78a8",
        label="Ostali redovi",
    )
    plt.scatter(
        outlier_rows["absences"],
        outlier_rows["G3"],
        alpha=0.85,
        color="#e45756",
        label="IQR ekstremi",
    )
    plt.title("Izostanci u odnosu na G3")
    plt.xlabel("absences")
    plt.ylabel("G3")
    plt.legend()
    plt.grid(alpha=0.25)
    save_graph(ABSENCES_VS_G3_OUTLIERS_PATH)


def create_categorical_unique_counts_graph(categorical_report):
    """Kreira bar graf broja jedinstvenih vrednosti po kategorijskom atributu."""
    sorted_report = categorical_report.sort_values("unique_value_count", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(
        sorted_report["attribute"],
        sorted_report["unique_value_count"],
        color="#59a14f",
        edgecolor="black",
    )
    plt.title("Broj jedinstvenih vrednosti po kategorijskom atributu")
    plt.xlabel("Kategorijski atribut")
    plt.ylabel("Broj jedinstvenih vrednosti")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.25)
    save_graph(CATEGORICAL_UNIQUE_COUNTS_PATH)


def main():
    # 1. Učitavanje originalnog dataset-a bez menjanja podataka.
    df = pd.read_csv(DATA_PATH)

    # 2. Kreiranje posebnih foldera za logove i grafike ovog step-a.
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Kreiranje IQR i kategorijskog izveštaja.
    iqr_report = create_iqr_report(df)
    categorical_report = create_categorical_report(df)
    scatter_outlier_mask = get_iqr_outlier_mask(df, OUTLIER_SCATTER_COLUMNS)
    summary = create_summary(
        df,
        iqr_report,
        categorical_report,
        int(scatter_outlier_mask.sum()),
    )

    # 4. Čuvanje CSV fajlova preko postojećeg helpera iz csv_utils.py.
    save_csv_if_changed(iqr_report, IQR_REPORT_PATH, PROJECT_ROOT)
    save_csv_if_changed(categorical_report, CATEGORICAL_REPORT_PATH, PROJECT_ROOT)
    save_csv_if_changed(summary, SUMMARY_PATH, PROJECT_ROOT)

    # 5. Kreiranje grafika za vizuelnu proveru ekstremnih i retkih vrednosti.
    create_numeric_boxplots(df)
    create_absences_vs_g3_outliers_scatter(df, scatter_outlier_mask)
    create_categorical_unique_counts_graph(categorical_report)

    print("Analiza ekstremnih i retkih vrednosti je završena.")


if __name__ == "__main__":
    main()
