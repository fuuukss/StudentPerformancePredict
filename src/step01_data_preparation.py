from pathlib import Path
import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Podešavanje UTF-8 ispisa je korisno na Windows konzolama.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


# Putanje do osnovnih foldera i fajlova u projektu.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student-por.csv"
LOGS_DIR = PROJECT_ROOT / "data" / "logs"

DATASET_OVERVIEW_PATH = LOGS_DIR / "dataset_overview.csv"
FEATURES_OVERVIEW_PATH = LOGS_DIR / "features_overview.csv"
PREPROCESSING_REPORT_PATH = LOGS_DIR / "preprocessing_report.csv"

TARGET_COLUMN = "G3"
TOP_FEATURES_STATUS = (
    "Scenario top_features biće definisan kasnije nakon feature importance "
    "analize."
)
PLANNED_PREPROCESSING = (
    "StandardScaler za numeričke atribute i "
    'OneHotEncoder(handle_unknown="ignore") za kategorijske atribute kroz '
    "ColumnTransformer za oba scenarija."
)


def relative_path(path):
    """Vraća putanju u odnosu na koren projekta, radi kraćeg ispisa."""
    return path.relative_to(PROJECT_ROOT).as_posix()


def format_value(value):
    """Formatira vrednosti za čitljiviji CSV izlaz."""
    if isinstance(value, bool):
        return "Da" if value else "Ne"

    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))

        return f"{value:.2f}"

    if isinstance(value, list):
        if not value:
            return "Nema"

        return "; ".join(str(item) for item in value)

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

    existing_dataframe = pd.read_csv(path, dtype=str, encoding="utf-8-sig")

    if existing_dataframe.equals(new_dataframe_for_compare):
        print(f"CSV fajl nije promenjen: {relative_path(path)}")
        return

    new_dataframe.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"CSV fajl ažuriran: {relative_path(path)}")


def has_values_outside_range(df, column, minimum, maximum):
    """Proverava da li kolona ima vrednosti van očekivanog opsega."""
    return bool(((df[column] < minimum) | (df[column] > maximum)).any())


def get_feature_columns(df, excluded_columns):
    """Vraća ulazne atribute nakon izbacivanja ciljnih i scenario kolona."""
    return [column for column in df.columns if column not in excluded_columns]


def split_feature_types(df, feature_columns):
    """Deli atribute na numeričke i kategorijske."""
    features = df[feature_columns]
    numeric_features = features.select_dtypes(include="number").columns.tolist()
    categorical_features = features.select_dtypes(exclude="number").columns.tolist()

    return numeric_features, categorical_features


def create_preprocessor(numeric_features, categorical_features):
    """Priprema preprocessing objekat bez fitovanja."""
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
    """Definiše scenario atribute i priprema preprocessing objekte bez fitovanja."""
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
            "number_of_features": len(feature_columns),
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "preprocessor": create_preprocessor(numeric_features, categorical_features),
        }

    return scenarios


def find_removal_candidates(df):
    """Pronalazi očigledne ID ili konstantne kolone kao kandidate za uklanjanje."""
    candidates = []

    for column in df.columns:
        if column == TARGET_COLUMN:
            continue

        lower_column = column.lower()

        if lower_column == "id" or lower_column.endswith("_id"):
            candidates.append(column)
            continue

        if df[column].nunique(dropna=False) == 1:
            candidates.append(column)

    return candidates


def create_dataset_overview(df):
    """Kreira čitljiv osnovni pregled dataset-a."""
    numeric_columns = df.select_dtypes(include="number").columns
    categorical_columns = df.select_dtypes(exclude="number").columns

    overview_rows = [
        {"Opis": "Broj redova", "Vrednost": df.shape[0]},
        {"Opis": "Broj kolona", "Vrednost": df.shape[1]},
        {"Opis": "Broj duplikata", "Vrednost": df.duplicated().sum()},
        {
            "Opis": "Ukupan broj nedostajućih vrednosti",
            "Vrednost": df.isna().sum().sum(),
        },
        {"Opis": "Broj numeričkih kolona", "Vrednost": len(numeric_columns)},
        {"Opis": "Broj kategorijskih kolona", "Vrednost": len(categorical_columns)},
        {"Opis": "Minimalna završna ocena G3", "Vrednost": df["G3"].min()},
        {"Opis": "Maksimalna završna ocena G3", "Vrednost": df["G3"].max()},
        {"Opis": "Prosečna završna ocena G3", "Vrednost": df["G3"].mean()},
        {"Opis": "Prosečna ocena G1", "Vrednost": df["G1"].mean()},
        {"Opis": "Prosečna ocena G2", "Vrednost": df["G2"].mean()},
        {"Opis": "Minimalna ocena G1", "Vrednost": df["G1"].min()},
        {"Opis": "Minimalna ocena G2", "Vrednost": df["G2"].min()},
        {"Opis": "Maksimalna ocena G1", "Vrednost": df["G1"].max()},
        {"Opis": "Maksimalna ocena G2", "Vrednost": df["G2"].max()},
        {
            "Opis": "Zaključak",
            "Vrednost": (
                "Dataset nema nedostajuće vrednosti i duplikate. G1 i G2 će "
                "biti posebno analizirani jer predstavljaju prethodne ocene i "
                "očekivano su povezani sa G3."
            ),
        },
    ]

    overview = pd.DataFrame(overview_rows, columns=["Opis", "Vrednost"])
    overview["Vrednost"] = overview["Vrednost"].apply(format_value)

    return overview


def get_feature_note(column):
    """Vraća kratku napomenu o ulozi kolone."""
    if column == "G1":
        return "prethodna ocena, koristi se samo u scenariju sa G1 i G2"
    if column == "G2":
        return "prethodna ocena, koristi se samo u scenariju sa G1 i G2"
    if column == TARGET_COLUMN:
        return "ciljna promenljiva"

    return "input feature"


def create_features_overview(df, scenarios):
    """Kreira pregled svih kolona i njihove upotrebe po scenarijima."""
    with_g1_g2_features = set(scenarios["with_G1_G2"]["features"])
    without_g1_g2_features = set(scenarios["without_G1_G2"]["features"])
    overview_rows = []

    for column in df.columns:
        is_numeric = pd.api.types.is_numeric_dtype(df[column])
        feature_type = "target" if column == TARGET_COLUMN else "numeric" if is_numeric else "categorical"

        if is_numeric:
            unique_values = "numeric"
        else:
            unique_values = "; ".join(
                sorted(df[column].dropna().astype(str).unique().tolist())
            )

        overview_rows.append(
            {
                "column": column,
                "dtype": str(df[column].dtype),
                "feature_type": feature_type,
                "missing_values": df[column].isna().sum(),
                "unique_count": df[column].nunique(dropna=False),
                "unique_values": unique_values,
                "used_with_G1_G2": column in with_g1_g2_features,
                "used_without_G1_G2": column in without_g1_g2_features,
                "note": get_feature_note(column),
            }
        )

    overview = pd.DataFrame(
        overview_rows,
        columns=[
            "column",
            "dtype",
            "feature_type",
            "missing_values",
            "unique_count",
            "unique_values",
            "used_with_G1_G2",
            "used_without_G1_G2",
            "note",
        ],
    )
    overview["used_with_G1_G2"] = overview["used_with_G1_G2"].apply(format_value)
    overview["used_without_G1_G2"] = overview["used_without_G1_G2"].apply(format_value)

    return overview


def create_preprocessing_report(df, scenarios):
    """Kreira čitljiv izveštaj o početnom preprocessing-u."""
    removal_candidates = find_removal_candidates(df)

    report_rows = [
        {"Opis": "Missing vrednosti", "Vrednost": df.isna().sum().sum()},
        {"Opis": "Duplikati", "Vrednost": df.duplicated().sum()},
        {
            "Opis": "G1 van opsega 0-20",
            "Vrednost": has_values_outside_range(df, "G1", 0, 20),
        },
        {
            "Opis": "G2 van opsega 0-20",
            "Vrednost": has_values_outside_range(df, "G2", 0, 20),
        },
        {
            "Opis": "G3 van opsega 0-20",
            "Vrednost": has_values_outside_range(df, "G3", 0, 20),
        },
        {
            "Opis": "Age anomalije",
            "Vrednost": has_values_outside_range(df, "age", 10, 25),
        },
        {"Opis": "Maksimalan broj izostanaka", "Vrednost": df["absences"].max()},
        {"Opis": "Ciljna promenljiva", "Vrednost": TARGET_COLUMN},
        {
            "Opis": "Scenario sa G1 i G2",
            "Vrednost": (
                f"{scenarios['with_G1_G2']['number_of_features']} atributa; "
                "koriste se svi ulazni atributi osim G3."
            ),
        },
        {
            "Opis": "Scenario bez G1 i G2",
            "Vrednost": (
                f"{scenarios['without_G1_G2']['number_of_features']} atributa; "
                "koriste se svi ulazni atributi osim G1, G2 i G3."
            ),
        },
        {"Opis": "Status top_features scenarija", "Vrednost": TOP_FEATURES_STATUS},
        {"Opis": "Kandidati za uklanjanje", "Vrednost": removal_candidates},
        {"Opis": "Planirani preprocessing", "Vrednost": PLANNED_PREPROCESSING},
        {
            "Opis": "Zaključak",
            "Vrednost": (
                "Dataset nema nedostajuće vrednosti i duplikate. G1, G2 i G3 "
                "su u očekivanom opsegu 0-20. Ne postoje očigledne ID ili "
                "konstantne kolone za uklanjanje. Preprocessing će koristiti "
                "StandardScaler za numeričke i OneHotEncoder za kategorijske "
                "atribute. Scenario top_features biće definisan kasnije nakon "
                "feature importance analize."
            ),
        },
    ]

    report = pd.DataFrame(report_rows, columns=["Opis", "Vrednost"])
    report["Vrednost"] = report["Vrednost"].apply(format_value)

    return report


def main():
    # 1. Učitavanje dataset-a pomoću pandas biblioteke.
    df = pd.read_csv(DATA_PATH)

    # 2. Kreiranje foldera za CSV logove ako ne postoji.
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Definisanje scenarija i priprema preprocessing objekata bez fitovanja.
    scenarios = create_scenarios(df)

    # 4. Kreiranje i čuvanje objedninjenih CSV izveštaja.
    dataset_overview = create_dataset_overview(df)
    save_csv_if_changed(dataset_overview, DATASET_OVERVIEW_PATH)

    features_overview = create_features_overview(df, scenarios)
    save_csv_if_changed(features_overview, FEATURES_OVERVIEW_PATH)

    preprocessing_report = create_preprocessing_report(df, scenarios)
    save_csv_if_changed(preprocessing_report, PREPROCESSING_REPORT_PATH)

    print("Priprema podataka je završena.")


if __name__ == "__main__":
    main()
