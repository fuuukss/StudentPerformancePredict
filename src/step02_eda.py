from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Podešavanje UTF-8 ispisa je korisno na Windows konzolama.
# Putanje do osnovnih foldera i fajlova u projektu.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student-por.csv"
GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs"
LOGS_DIR = PROJECT_ROOT / "data" / "logs"

EDA_REPORT_PATH = LOGS_DIR / "eda_report.csv"

G3_DISTRIBUTION_GRAPH_PATH = GRAPHS_DIR / "g3_distribution.png"
NUMERIC_CORRELATION_MATRIX_GRAPH_PATH = GRAPHS_DIR / "numeric_correlation_matrix.png"
G1_VS_G3_GRAPH_PATH = GRAPHS_DIR / "g1_vs_g3.png"
G2_VS_G3_GRAPH_PATH = GRAPHS_DIR / "g2_vs_g3.png"
FAILURES_VS_G3_GRAPH_PATH = GRAPHS_DIR / "failures_vs_g3.png"
STUDYTIME_VS_G3_GRAPH_PATH = GRAPHS_DIR / "studytime_vs_g3.png"
ABSENCES_DISTRIBUTION_GRAPH_PATH = GRAPHS_DIR / "absences_distribution.png"

EDA_CONCLUSION = (
    "EDA pokazuje da su G1 i G2 najjače povezani sa završnom ocenom G3, "
    "zbog čega je opravdano porediti modele sa i bez ovih atributa. "
    "Vrednosti G3 = 0 i veći broj izostanaka se za sada ne brišu jer mogu "
    "predstavljati realne slučajeve u dataset-u."
)


def relative_path(path):
    """Vraća putanju u odnosu na koren projekta, radi kraćeg ispisa."""
    return path.relative_to(PROJECT_ROOT).as_posix()


def format_value(value):
    """Formatira vrednosti za čitljiviji CSV izlaz."""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))

        return f"{value:.2f}"

    return str(value)


def save_csv_if_changed(new_dataframe, path):
    """
    Čuva CSV samo ako fajl ne postoji ili ako se sadržaj promenio.

    Ova funkcija pomaže da se generisani CSV fajlovi ne menjaju pri svakom
    pokretanju skripte ako dataset nije promenjen.
    """
    new_dataframe_for_compare = new_dataframe.astype(str)

    if not path.exists():
        new_dataframe.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"CSV fajl kreiran: {relative_path(path)}")
        return

    existing_dataframe = pd.read_csv(
        path,
        dtype=str,
        encoding="utf-8-sig",
        keep_default_na=False,
    )

    if existing_dataframe.equals(new_dataframe_for_compare):
        print(f"CSV fajl nije promenjen: {relative_path(path)}")
        return

    new_dataframe.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"CSV fajl ažuriran: {relative_path(path)}")


def save_graph(path):
    """Čuva trenutni grafik i zatvara figuru."""
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Grafik sačuvan: {relative_path(path)}")


def create_g3_distribution_graph(df):
    """Kreira histogram završne ocene G3."""
    plt.figure(figsize=(8, 5))
    plt.hist(df["G3"], bins=range(0, 22), color="#4c78a8", edgecolor="black")
    plt.title("Raspodela završne ocene G3")
    plt.xlabel("G3 ocena")
    plt.ylabel("Broj učenika")
    plt.xticks(range(0, 21, 2))
    save_graph(G3_DISTRIBUTION_GRAPH_PATH)


def create_numeric_correlation_graph(correlation_matrix):
    """Kreira grafički prikaz korelacija numeričkih atributa."""
    columns = correlation_matrix.columns.tolist()

    plt.figure(figsize=(12, 10))
    plt.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Korelacija")
    plt.title("Korelaciona matrica numeričkih atributa")
    plt.xticks(range(len(columns)), columns, rotation=90)
    plt.yticks(range(len(columns)), columns)
    save_graph(NUMERIC_CORRELATION_MATRIX_GRAPH_PATH)


def create_scatter_graph(df, x_column, y_column, path, title):
    """Kreira scatter plot za dve numeričke kolone."""
    plt.figure(figsize=(7, 5))
    plt.scatter(df[x_column], df[y_column], alpha=0.6, color="#4c78a8")
    plt.title(title)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.grid(alpha=0.25)
    save_graph(path)


def create_mean_g3_bar_graph(df, group_column, path, title, x_label):
    """Kreira bar plot prosečne G3 ocene po vrednostima izabrane kolone."""
    grouped = df.groupby(group_column)["G3"].mean().sort_index()

    plt.figure(figsize=(7, 5))
    plt.bar(grouped.index.astype(str), grouped.values, color="#59a14f", edgecolor="black")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Prosečna G3 ocena")
    plt.ylim(0, 20)
    save_graph(path)


def create_absences_distribution_graph(df):
    """Kreira histogram broja izostanaka."""
    plt.figure(figsize=(8, 5))
    plt.hist(df["absences"], bins=20, color="#f28e2b", edgecolor="black")
    plt.title("Raspodela broja izostanaka")
    plt.xlabel("Broj izostanaka")
    plt.ylabel("Broj učenika")
    save_graph(ABSENCES_DISTRIBUTION_GRAPH_PATH)


def get_strongest_correlation_with_g3(correlation_matrix):
    """Vraća atribut sa najjačom korelacijom sa G3."""
    correlations = correlation_matrix["G3"].drop(labels=["G3"])
    strongest_feature = correlations.abs().idxmax()
    strongest_correlation = correlations.loc[strongest_feature]

    return f"{strongest_feature} ({strongest_correlation:.2f})"


def create_eda_report(df, correlation_matrix):
    """Kreira kratak CSV izveštaj sa najvažnijim EDA nalazima."""
    g3_zero_count = int((df["G3"] == 0).sum())
    g3_zero_percentage = g3_zero_count / len(df) * 100

    report_rows = [
        {"Opis": "Broj učenika", "Vrednost": len(df)},
        {"Opis": "Minimalna G3 ocena", "Vrednost": df["G3"].min()},
        {"Opis": "Maksimalna G3 ocena", "Vrednost": df["G3"].max()},
        {"Opis": "Prosečna G3 ocena", "Vrednost": df["G3"].mean()},
        {"Opis": "Broj učenika sa G3 = 0", "Vrednost": g3_zero_count},
        {"Opis": "Procenat učenika sa G3 = 0", "Vrednost": g3_zero_percentage},
        {"Opis": "Korelacija G1 i G3", "Vrednost": correlation_matrix.loc["G1", "G3"]},
        {"Opis": "Korelacija G2 i G3", "Vrednost": correlation_matrix.loc["G2", "G3"]},
        {"Opis": "Korelacija G1 i G2", "Vrednost": correlation_matrix.loc["G1", "G2"]},
        {
            "Opis": "Korelacija failures i G3",
            "Vrednost": correlation_matrix.loc["failures", "G3"],
        },
        {
            "Opis": "Korelacija studytime i G3",
            "Vrednost": correlation_matrix.loc["studytime", "G3"],
        },
        {
            "Opis": "Korelacija absences i G3",
            "Vrednost": correlation_matrix.loc["absences", "G3"],
        },
        {
            "Opis": "Najjača korelacija sa G3",
            "Vrednost": get_strongest_correlation_with_g3(correlation_matrix),
        },
        {"Opis": "Maksimalan broj izostanaka", "Vrednost": df["absences"].max()},
        {"Opis": "Zaključak", "Vrednost": EDA_CONCLUSION},
    ]

    report = pd.DataFrame(report_rows, columns=["Opis", "Vrednost"])
    report["Vrednost"] = report["Vrednost"].apply(format_value)

    return report


def main():
    # 1. Učitavanje dataset-a pomoću pandas biblioteke.
    df = pd.read_csv(DATA_PATH)

    # 2. Kreiranje foldera za grafike i CSV logove ako ne postoje.
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Raspodela ciljne promenljive G3.
    create_g3_distribution_graph(df)

    # 4. Korelacije numeričkih atributa.
    numeric_df = df.select_dtypes(include="number")
    correlation_matrix = numeric_df.corr()
    create_numeric_correlation_graph(correlation_matrix)

    # 5. Scatter plot G1 vs G3.
    create_scatter_graph(
        df,
        "G1",
        "G3",
        G1_VS_G3_GRAPH_PATH,
        "G1 u odnosu na G3",
    )

    # 6. Scatter plot G2 vs G3.
    create_scatter_graph(
        df,
        "G2",
        "G3",
        G2_VS_G3_GRAPH_PATH,
        "G2 u odnosu na G3",
    )

    # 7. Analiza atributa failures.
    create_mean_g3_bar_graph(
        df,
        "failures",
        FAILURES_VS_G3_GRAPH_PATH,
        "Prosečna G3 ocena po broju prethodnih neuspeha",
        "Broj prethodnih neuspeha",
    )

    # 8. Analiza atributa studytime.
    create_mean_g3_bar_graph(
        df,
        "studytime",
        STUDYTIME_VS_G3_GRAPH_PATH,
        "Prosečna G3 ocena po vremenu učenja",
        "Vreme učenja",
    )

    # 9. Analiza atributa absences.
    create_absences_distribution_graph(df)

    # 10. EDA report, uključujući broj i procenat učenika sa G3 = 0.
    eda_report = create_eda_report(df, correlation_matrix)
    save_csv_if_changed(eda_report, EDA_REPORT_PATH)

    print("EDA analiza je završena.")


if __name__ == "__main__":
    main()
