from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from csv_utils import relative_path, save_csv_if_changed


# Putanje do osnovnih foldera i fajlova u projektu.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student-por.csv"
SPLIT_DIR = PROJECT_ROOT / "data" / "split"
LOGS_DIR = PROJECT_ROOT / "data" / "logs" / "step03_data_split"

TRAIN_PATH = SPLIT_DIR / "train.csv"
VALIDATION_PATH = SPLIT_DIR / "validation.csv"
TEST_PATH = SPLIT_DIR / "test.csv"
SPLIT_REPORT_PATH = LOGS_DIR / "split_report.csv"

TARGET_COLUMN = "G3"
RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


def format_value(value):
    """Formatira vrednosti za citljiviji CSV izlaz."""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))

        return f"{value:.2f}"

    return str(value)


def split_dataset(df):
    """Deli dataset jednom na train, validation i test skupove."""
    train_df, temp_df = train_test_split(
        df,
        train_size=TRAIN_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    validation_relative_size = VALIDATION_SIZE / (VALIDATION_SIZE + TEST_SIZE)
    validation_df, test_df = train_test_split(
        temp_df,
        train_size=validation_relative_size,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    return (
        train_df.sort_index().reset_index(drop=True),
        validation_df.sort_index().reset_index(drop=True),
        test_df.sort_index().reset_index(drop=True),
    )


def create_split_report(original_df, train_df, validation_df, test_df):
    """Kreira kratak CSV izvestaj o podeli podataka."""
    total_rows = len(original_df)
    report_rows = [
        {"Opis": "Izvorni dataset", "Vrednost": relative_path(DATA_PATH, PROJECT_ROOT)},
        {"Opis": "Ciljna promenljiva", "Vrednost": TARGET_COLUMN},
        {"Opis": "Random state", "Vrednost": RANDOM_STATE},
        {"Opis": "Ukupan broj redova", "Vrednost": total_rows},
        {"Opis": "Train redovi", "Vrednost": len(train_df)},
        {"Opis": "Validation redovi", "Vrednost": len(validation_df)},
        {"Opis": "Test redovi", "Vrednost": len(test_df)},
        {
            "Opis": "Train procenat",
            "Vrednost": len(train_df) / total_rows * 100,
        },
        {
            "Opis": "Validation procenat",
            "Vrednost": len(validation_df) / total_rows * 100,
        },
        {"Opis": "Test procenat", "Vrednost": len(test_df) / total_rows * 100},
        {
            "Opis": "Sacuvani train skup",
            "Vrednost": relative_path(TRAIN_PATH, PROJECT_ROOT),
        },
        {
            "Opis": "Sacuvani validation skup",
            "Vrednost": relative_path(VALIDATION_PATH, PROJECT_ROOT),
        },
        {"Opis": "Sacuvani test skup", "Vrednost": relative_path(TEST_PATH, PROJECT_ROOT)},
        {
            "Opis": "Zakljucak",
            "Vrednost": (
                "Dataset je jednom podeljen na train, validation i test skup. "
                "Isti redovi se koriste za scenarije sa G1/G2 i bez G1/G2, "
                "kako bi poredjenje modela bilo fer."
            ),
        },
    ]

    report = pd.DataFrame(report_rows, columns=["Opis", "Vrednost"])
    report["Vrednost"] = report["Vrednost"].apply(format_value)

    return report


def main():
    # 1. Ucitavanje originalnog raw dataset-a.
    df = pd.read_csv(DATA_PATH)

    # 2. Kreiranje foldera za split fajlove i CSV logove ako ne postoje.
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Jedna zajednicka podela na train, validation i test skupove.
    train_df, validation_df, test_df = split_dataset(df)

    # 4. Cuvanje splitovanih skupova bez dodatnog procesiranja atributa.
    save_csv_if_changed(train_df, TRAIN_PATH, PROJECT_ROOT)
    save_csv_if_changed(validation_df, VALIDATION_PATH, PROJECT_ROOT)
    save_csv_if_changed(test_df, TEST_PATH, PROJECT_ROOT)

    # 5. Kreiranje i cuvanje kratkog izvestaja o podeli.
    split_report = create_split_report(df, train_df, validation_df, test_df)
    save_csv_if_changed(split_report, SPLIT_REPORT_PATH, PROJECT_ROOT)

    print("Podela podataka je zavrsena.")


if __name__ == "__main__":
    main()
