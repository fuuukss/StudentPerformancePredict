import pandas as pd


def relative_path(path, project_root):
    """Vraca putanju u odnosu na koren projekta, radi kraceg ispisa."""
    return path.relative_to(project_root).as_posix()


def save_csv_if_changed(new_dataframe, path, project_root):
    """
    Cuva CSV samo ako fajl ne postoji ili ako se sadrzaj promenio.

    Ova funkcija pomaze da se generisani CSV fajlovi ne menjaju pri svakom
    pokretanju skripte ako dataset nije promenjen.
    """
    new_dataframe_for_compare = new_dataframe.astype(str)

    if not path.exists():
        new_dataframe.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"CSV fajl kreiran: {relative_path(path, project_root)}")
        return

    existing_dataframe = pd.read_csv(
        path,
        dtype=str,
        encoding="utf-8-sig",
        keep_default_na=False,
    )

    if existing_dataframe.equals(new_dataframe_for_compare):
        print(f"CSV fajl nije promenjen: {relative_path(path, project_root)}")
        return

    new_dataframe.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"CSV fajl azuriran: {relative_path(path, project_root)}")
