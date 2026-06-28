# Student Performance Prediction

## Opis projekta

Projekat predviđa završnu ocenu učenika `G3` na osnovu skupa podataka `student-por.csv`. Problem je regresioni, jer se predviđa numerička vrednost ocene.

U projektu se porede modeli i scenariji sa prethodnim ocenama `G1/G2` i bez njih. Time se proverava koliko prethodne ocene utiču na kvalitet predikcije i koliko je model praktično upotrebljiv kada one nisu dostupne.

## Struktura projekta

```text
data/raw/       originalni dataset
data/split/     train, validation i test skupovi
data/logs/      CSV izveštaji po koracima
data/graphs/    PNG grafici po koracima
models/final/   eksportovan finalni model
src/            Python skripte projekta
docs/           dokumentacija i walkthrough notebook
```

## Korišćeni modeli

- `DummyRegressor`
- `Ridge Regression`
- `RandomForestRegressor`

## Metrike

- MAE
- RMSE
- R2

## Instalacija zavisnosti

Pre pokretanja projekta potrebno je napraviti virtuelno okruženje i instalirati potrebne biblioteke:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Pokretanje projekta

Redosled pokretanja step skripti u PowerShell-u:

```powershell
.\.venv\Scripts\python.exe .\src\step01_data_preparation.py
.\.venv\Scripts\python.exe .\src\step02_eda.py
.\.venv\Scripts\python.exe .\src\step03_data_split.py
.\.venv\Scripts\python.exe .\src\step04_model_training.py
.\.venv\Scripts\python.exe .\src\step05_model_comparison_graphs.py
.\.venv\Scripts\python.exe .\src\step06_hyperparameter_tuning.py
.\.venv\Scripts\python.exe .\src\step07_tuning_graphs.py
.\.venv\Scripts\python.exe .\src\step08_feature_importance.py
.\.venv\Scripts\python.exe .\src\step09_top_features.py
.\.venv\Scripts\python.exe .\src\step10_final_model_selection.py
```

## Korišćenje finalnog modela

Predikcija iz terminala:

```powershell
.\.venv\Scripts\python.exe .\src\predict.py
```

Primer sa argumentima:

```powershell
.\.venv\Scripts\python.exe .\src\predict.py --G2 13 --absences 6 --G1 12 --age 15 --reason other --freetime 3 --health 3 --Fedu 1 --school GP --failures 0
```

Streamlit aplikacija:

```powershell
streamlit run .\src\streamlit_app.py
```

## Rezultati

Najbolji rezultat daje scenario sa `G1/G2`, jer su prethodne ocene veoma značajni atributi za predikciju završne ocene. Scenario bez `G1/G2` ima slabije metrike, ali je praktično koristan za raniju procenu uspeha učenika.

Finalni model je eksportovan u:

```text
models/final/final_model.joblib
```

## Dokumentacija

Detaljna dokumentacija nalazi se u `docs/project_documentation.md`.

Walkthrough notebook nalazi se u `docs/student_performance_walkthrough.ipynb`.
