from pathlib import Path
import sys

import pandas as pd


# Podesavanje UTF-8 ispisa je korisno na Windows konzolama.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


# Putanje do osnovnih foldera i fajlova u projektu.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student-por.csv"
LOGS_DIR = PROJECT_ROOT / "data" / "logs"

SUMMARY_PATH = LOGS_DIR / "data_overview_summary.csv"
COLUMNS_OVERVIEW_PATH = LOGS_DIR / "columns_overview.csv"


def relative_path(path):
    """Vraca putanju u odnosu na koren projekta, radi kraceg ispisa."""
    return path.relative_to(PROJECT_ROOT).as_posix()


def format_value(value):
    """Formatira vrednosti za citljiviji CSV izlaz."""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))

        return f"{value:.2f}"

    return str(value)


def save_csv_if_changed(new_dataframe, path):
    """
    Cuva CSV samo ako fajl ne postoji ili ako se sadrzaj promenio.

    Ova funkcija pomaze da se generisani CSV fajlovi ne menjaju pri svakom
    pokretanju skripte ako dataset nije promenjen.
    """
    new_dataframe_for_compare = new_dataframe.astype(str)

    if not path.exists():
        new_dataframe.to_csv(path, index=False)
        print(f"CSV fajl kreiran: {relative_path(path)}")
        return

    existing_dataframe = pd.read_csv(path, dtype=str)

    if existing_dataframe.equals(new_dataframe_for_compare):
        print(f"CSV fajl nije promenjen: {relative_path(path)}")
        return

    new_dataframe.to_csv(path, index=False)
    print(f"CSV fajl ažuriran: {relative_path(path)}")


def create_data_overview_summary(df):
    """Kreira kratak pregled osnovnih informacija o dataset-u."""
    numeric_columns = df.select_dtypes(include="number").columns
    categorical_columns = df.select_dtypes(exclude="number").columns

    summary_rows = [
        {
            "Opis": "Broj redova",
            "Vrednost": df.shape[0],
        },
        {
            "Opis": "Broj kolona",
            "Vrednost": df.shape[1],
        },
        {
            "Opis": "Broj duplikata",
            "Vrednost": df.duplicated().sum(),
        },
        {
            "Opis": "Ukupan broj nedostajućih vrednosti",
            "Vrednost": df.isna().sum().sum(),
        },
        {
            "Opis": "Broj numeričkih kolona",
            "Vrednost": len(numeric_columns),
        },
        {
            "Opis": "Broj kategorijskih kolona",
            "Vrednost": len(categorical_columns),
        },
        {
            "Opis": "Minimalna završna ocena G3",
            "Vrednost": df["G3"].min(),
        },
        {
            "Opis": "Maksimalna završna ocena G3",
            "Vrednost": df["G3"].max(),
        },
        {
            "Opis": "Prosečna završna ocena G3",
            "Vrednost": df["G3"].mean(),
        },
        {
            "Opis": "Prosečna ocena G1",
            "Vrednost": df["G1"].mean(),
        },
        {
            "Opis": "Prosečna ocena G2",
            "Vrednost": df["G2"].mean(),
        },
        {
            "Opis": "Minimalna ocena G1",
            "Vrednost": df["G1"].min(),
        },
        {
            "Opis": "Minimalna ocena G2",
            "Vrednost": df["G2"].min(),
        },
        {
            "Opis": "Maksimalna ocena G1",
            "Vrednost": df["G1"].max(),
        },
        {
            "Opis": "Maksimalna ocena G2",
            "Vrednost": df["G2"].max(),
        },
        {
            "Opis": "Zaključak",
            "Vrednost": (
                "Dataset nema nedostajuće vrednosti i duplikate. G1 i G2 će "
                "biti posebno analizirani jer predstavljaju prethodne ocene i "
                "očekivano su povezani sa G3."
            ),
        },
    ]

    summary = pd.DataFrame(summary_rows, columns=["Opis", "Vrednost"])
    summary["Vrednost"] = summary["Vrednost"].apply(format_value)

    return summary


def create_columns_overview(df):
    """Kreira pregled tipova, nedostajucih i jedinstvenih vrednosti po kolonama."""
    columns_overview = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "missing_values": df.isna().sum().values,
            "unique_values": df.nunique().values,
        }
    )

    return columns_overview


def main():
    # 1. Ucitavanje dataset-a pomocu pandas biblioteke.
    df = pd.read_csv(DATA_PATH)

    # 2. Kreiranje foldera za CSV logove ako ne postoji.
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Kreiranje i cuvanje kratkog pregleda dataset-a.
    data_overview_summary = create_data_overview_summary(df)
    save_csv_if_changed(data_overview_summary, SUMMARY_PATH)

    # 4. Kreiranje i cuvanje pregleda svih kolona.
    columns_overview = create_columns_overview(df)
    save_csv_if_changed(columns_overview, COLUMNS_OVERVIEW_PATH)

    print("Pregled podataka je završen.")


if __name__ == "__main__":
    main()
