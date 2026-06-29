# Dokumentacija projekta: Student Performance Prediction

## 1. Uvod

Cilj projekta je predikcija završne ocene učenika, označene atributom `G3`, na osnovu podataka iz skupa `student-por.csv`. Problem je formulisan kao regresioni zadatak, jer se predviđa numerička vrednost ocene na skali od 0 do 20.

Projekat ne posmatra samo najbolju metriku, već i praktičnu upotrebljivost modela. Zbog toga su posebno upoređena dva scenarija:

- `with_G1_G2`: model koristi prethodne ocene `G1` i `G2`;
- `without_G1_G2`: model ne koristi prethodne ocene `G1` i `G2`.

Ovakvo poređenje je važno zato što `G1` i `G2` nose mnogo informacija o završnoj oceni, ali u nekim praktičnim situacijama te ocene možda još nisu dostupne.

## 2. Opis skupa podataka

Korišćen je skup podataka `data/raw/student-por.csv`. Dataset sadrži podatke o učenicima, njihovom porodičnom okruženju, školskim navikama, izostancima, prethodnim ocenama i završnoj oceni.

Sažetak iz `data/logs/step01_data_preparation/dataset_overview.csv`:

| Opis | Vrednost |
| --- | ---: |
| Broj redova | 649 |
| Broj kolona | 33 |
| Broj numeričkih kolona | 16 |
| Broj kategorijskih kolona | 17 |
| Broj duplikata | 0 |
| Ukupan broj nedostajućih vrednosti | 0 |
| Minimalna završna ocena `G3` | 0 |
| Maksimalna završna ocena `G3` | 19 |
| Prosečna završna ocena `G3` | 11.91 |

Atributi `G1`, `G2` i `G3` predstavljaju ocene učenika:

- `G1`: prva periodična ocena;
- `G2`: druga periodična ocena;
- `G3`: završna ocena i ciljna promenljiva.

Prema `features_overview.csv`, svi atributi osim `G3` mogu biti ulazni atributi u scenariju sa `G1/G2`. U scenariju bez `G1/G2`, iz ulaza se dodatno uklanjaju `G1` i `G2`.

## 3. Početno preprocesiranje podataka

U koraku početne pripreme provereni su nedostajuće vrednosti, duplikati, osnovni opsezi ocena i očigledne greške za uklanjanje. Prema `data/logs/step01_data_preparation/preprocessing_report.csv`, dataset nema nedostajuće vrednosti ni duplikate, a ocene `G1`, `G2` i `G3` nalaze se u očekivanom opsegu 0-20.

| Provera | Rezultat |
| --- | --- |
| Missing vrednosti | 0 |
| Duplikati | 0 |
| `G1` van opsega 0-20 | Ne |
| `G2` van opsega 0-20 | Ne |
| `G3` van opsega 0-20 | Ne |
| Kandidati za uklanjanje | Nema |

Nije bilo očiglednih redova ili atributa za automatsko uklanjanje. Maksimalan broj izostanaka je 32, što se posmatra kao potencijalno ekstremna, ali moguća realna vrednost.

Enkodiranje kategorijskih atributa i skaliranje numeričkih atributa nisu urađeni unapred nad celim datasetom. Umesto toga, `OneHotEncoder` i `StandardScaler` koriste se kasnije unutar `sklearn Pipeline` objekata tokom treniranja modela. `StandardScaler` skalira numeričke atribute tako da budu uporedivih razmera, dok `OneHotEncoder` pretvara kategorijske vrednosti u numeričke indikatore koje modeli mogu da koriste.

Ovakav pristup sprečava data leakage. Transformatori se fituju samo na trening delu podataka, a zatim se ista naučena transformacija primenjuje na validacioni, test i nove korisničke unose. Validation i test skup zato ne učestvuju u učenju preprocessing-a.

## 4. Eksplorativna analiza podataka

Eksplorativna analiza je sprovedena u koraku `step02_eda.py`, a sažetak se nalazi u `data/logs/step02_eda/eda_report.csv`.

| Pokazatelj | Vrednost |
| --- | ---: |
| Broj učenika | 649 |
| Prosečna `G3` ocena | 11.91 |
| Broj učenika sa `G3 = 0` | 15 |
| Procenat učenika sa `G3 = 0` | 2.31% |
| Korelacija `G1` i `G3` | 0.83 |
| Korelacija `G2` i `G3` | 0.92 |
| Korelacija `failures` i `G3` | -0.39 |
| Korelacija `studytime` i `G3` | 0.25 |
| Korelacija `absences` i `G3` | -0.09 |

Distribucija završne ocene prikazuje raspored vrednosti ciljne promenljive:

![Distribucija završne ocene G3](../data/graphs/step02_eda/g3_distribution.png)

Korelaciona matrica numeričkih atributa pokazuje da su `G1` i naročito `G2` najjače povezani sa `G3`:

![Korelaciona matrica numeričkih atributa](../data/graphs/step02_eda/numeric_correlation_matrix.png)

Odnos prethodnih ocena i završne ocene dodatno potvrđuje jaku linearnu vezu:

![G1 u odnosu na G3](../data/graphs/step02_eda/g1_vs_g3.png)

![G2 u odnosu na G3](../data/graphs/step02_eda/g2_vs_g3.png)

Ostali atributi imaju slabiju, ali korisnu vezu sa završnom ocenom. Broj prethodnih neuspeha (`failures`) ima negativnu korelaciju sa `G3`, dok `studytime` pokazuje pozitivnu, ali znatno slabiju vezu.

![Failures u odnosu na G3](../data/graphs/step02_eda/failures_vs_g3.png)

![Studytime u odnosu na G3](../data/graphs/step02_eda/studytime_vs_g3.png)

Broj izostanaka ima slabu negativnu korelaciju sa završnom ocenom, ali distribucija izostanaka je važna za razumevanje ekstremnijih slučajeva:

![Distribucija izostanaka](../data/graphs/step02_eda/absences_distribution.png)

Vrednosti poput `G3 = 0` i većeg broja izostanaka nisu automatski obrisane, jer mogu predstavljati realne slučajeve u obrazovnom kontekstu.

## 5. Dodatna analiza ekstremnih i retkih vrednosti

Pored osnovne provere opsega, urađena je dodatna analiza ekstremnih i retkih vrednosti u koraku `step02b_anomaly_analysis.py`. Cilj ovog koraka je bolje razumevanje vrednosti koje statistički odskaču ili se retko pojavljuju, a ne filtriranje podataka ili menjanje toka treniranja modela.

Za numeričke atribute koristi se IQR metoda. IQR metoda koristi kvartile i interkvartilni raspon:

```text
IQR = Q3 - Q1
donja granica = Q1 - 1.5 * IQR
gornja granica = Q3 + 1.5 * IQR
```

Vrednosti ispod donje granice ili iznad gornje granice označavaju se kao potencijalni outlier-i. To znači da statistički odskaču od većine vrednosti u koloni, ali ne znači automatski da su greške u podacima.

Za kategorijske atribute ne koristi se IQR, jer njihove vrednosti nisu numeričke i nemaju prirodan redosled za računanje kvartila. Umesto toga proverava se broj jedinstvenih vrednosti po atributu i da li postoje kategorije koje se pojavljuju retko. U ovom projektu kategorija se smatra retkom ako se pojavljuje najviše 5 puta.

Izveštaji ovog koraka nalaze se u:

- `data/logs/step02b_anomaly_analysis/anomaly_summary.csv`;
- `data/logs/step02b_anomaly_analysis/iqr_outlier_report.csv`;
- `data/logs/step02b_anomaly_analysis/categorical_value_report.csv`.

Sažetak dodatne analize:

| Pokazatelj | Vrednost |
| --- | ---: |
| IQR analiza numeričkih atributa | 16 atributa |
| Atributi kod kojih IQR pronalazi ekstremne vrednosti | 11 od 16 |
| Broj učenika sa `G3 = 0` | 15 |
| Maksimalan broj izostanaka | 32 |
| Prag za retke kategorije | 5 |

![Boxplot numeričkih atributa](../data/graphs/step02b_anomaly_analysis/numeric_boxplots.png)

![Izostanci u odnosu na G3 sa označenim IQR ekstremima](../data/graphs/step02b_anomaly_analysis/absences_vs_g3_outliers.png)

![Broj jedinstvenih vrednosti po kategorijskom atributu](../data/graphs/step02b_anomaly_analysis/categorical_unique_counts.png)

Ekstremne numeričke vrednosti i retke kategorije nisu automatski greške. One mogu predstavljati realne učenike i realne situacije u dataset-u, pa redovi nisu uklonjeni. Ova analiza služi za proveru i bolje razumevanje podataka, a ne za filtriranje redova ili menjanje finalnog modela.

## 6. Podela podataka

Dataset je jednom podeljen na trening, validacioni i test skup. Trening skup služi za učenje modela, validacioni skup za izbor modela i hiperparametara, a test skup za finalnu proveru performansi na odvojenim podacima. Izveštaj se nalazi u `data/logs/step03_data_split/split_report.csv`.

| Skup | Broj redova | Procenat |
| --- | ---: | ---: |
| Train | 454 | 69.95% |
| Validation | 97 | 14.95% |
| Test | 98 | 15.10% |

Svi modeli i svi scenariji koriste iste redove iz ove podele. Time je poređenje modela fer i ponovljivo, jer razlike u rezultatima dolaze od modela i atributa, a ne od drugačije podele podataka.

## 7. Odabir i treniranje modela

U početnom poređenju trenirana su tri modela:

- `DummyRegressor`: baseline model koji služi za proveru da li ostali modeli uče koristan signal;
- `Ridge Regression`: linearni model sa regularizacijom;
- `RandomForestRegressor`: ansambl model koji može uhvatiti nelinearne zavisnosti.

Korišćene metrike su:

- MAE: prosečna apsolutna greška, izražena u jedinicama ocene;
- RMSE: koren srednje kvadratne greške, koji jače kažnjava veće greške;
- R2: procenat objašnjene varijanse target promenljive.

Za MAE i RMSE manje vrednosti su bolje. Za R2 veće vrednosti su bolje, dok vrednosti oko nule ili ispod nule znače da model ne objašnjava podatke bolje od veoma jednostavnog baseline pristupa.

Sažetak iz `data/logs/step04_model_training/model_comparison_report.csv`:

| Scenario | Model | Validation RMSE | Validation R2 | Test RMSE | Test R2 |
| --- | --- | ---: | ---: | ---: | ---: |
| with_G1_G2 | DummyRegressor | 3.1360 | -0.0004 | 3.5229 | -0.0291 |
| with_G1_G2 | Ridge Regression | 1.0555 | 0.8867 | 1.3314 | 0.8530 |
| with_G1_G2 | RandomForestRegressor | 0.9862 | 0.9011 | 1.3905 | 0.8397 |
| without_G1_G2 | DummyRegressor | 3.1360 | -0.0004 | 3.5229 | -0.0291 |
| without_G1_G2 | Ridge Regression | 2.8668 | 0.1639 | 3.0243 | 0.2416 |
| without_G1_G2 | RandomForestRegressor | 2.7779 | 0.2150 | 2.9105 | 0.2976 |

![Poređenje validacionih metrika](../data/graphs/step05_model_comparison/validation_metrics_comparison.png)

![Poređenje test metrika](../data/graphs/step05_model_comparison/test_metrics_comparison.png)

Modeli sa `G1` i `G2` ostvaruju znatno bolje rezultate. To je očekivano, jer prethodne ocene imaju direktnu i jaku vezu sa završnom ocenom. Bez `G1/G2`, modeli i dalje nadmašuju baseline, ali sa mnogo skromnijim `R2` vrednostima.

## 8. Podešavanje hiperparametara

Podešavanje hiperparametara sprovedeno je pomoću validacionog skupa. Hiperparametri su podešavanja modela koja nisu direktno naučena iz podataka, već se biraju pre ili tokom treniranja, na primer `alpha` kod Ridge Regression modela ili `max_depth` i `min_samples_leaf` kod Random Forest modela.

U skripti `step06_hyperparameter_tuning.py` korišćen je ručni grid search: unapred je definisana mala mreža kandidata, treniraju se sve kombinacije i porede se prema validacionom rezultatu. Test skup nije korišćen za izbor hiperparametara, već samo za proveru nakon izbora.

Sažetak iz `data/logs/step06_hyperparameter_tuning/hyperparameter_tuning_summary.csv`:

| Scenario | Model | Najbolji parametri | Validation RMSE | Test RMSE | Test R2 |
| --- | --- | --- | ---: | ---: | ---: |
| with_G1_G2 | Ridge Regression | `alpha=10.0` | 1.0457 | 1.3370 | 0.8518 |
| with_G1_G2 | RandomForestRegressor | `n_estimators=100; max_depth=5; min_samples_leaf=5` | 0.8831 | 1.4138 | 0.8343 |
| without_G1_G2 | Ridge Regression | `alpha=100.0` | 2.7782 | 2.9202 | 0.2929 |
| without_G1_G2 | RandomForestRegressor | `n_estimators=100; max_depth=10; min_samples_leaf=2` | 2.7633 | 2.9108 | 0.2974 |

![Default i tuned modeli na validacionom skupu](../data/graphs/step07_tuning_graphs/default_vs_tuned_validation.png)

![Default i tuned modeli na test skupu](../data/graphs/step07_tuning_graphs/default_vs_tuned_test.png)

Tuning donosi poboljšanje, posebno za `RandomForestRegressor` u scenariju sa `G1/G2`, gde validacioni RMSE pada sa 0.9862 na 0.8831. Ipak, razlika između validacionog i test rezultata pokazuje da izbor finalnog modela ne treba zasnivati samo na jednoj metrici, već i na generalizaciji, jednostavnosti i praktičnoj interpretabilnosti.

## 9. Analiza značajnosti atributa

Feature importance pokazuje koliko se model oslanja na pojedinačne atribute pri predikciji. Za procenu važnosti atributa korišćen je `RandomForestRegressor`, jer ovaj model prirodno daje procenu doprinosa atributa na osnovu svojih stabala odlučivanja. Rezultati se nalaze u:

- `data/logs/step08_feature_importance/feature_importance_with_G1_G2.csv`;
- `data/logs/step08_feature_importance/feature_importance_without_G1_G2.csv`.

U scenariju sa `G1/G2`, najvažniji atribut je `G2`, sa značajem 0.896595. Nakon njega slede `absences`, `G1`, `age` i `reason`.

![Važnost atributa sa G1 i G2](../data/graphs/step08_feature_importance/feature_importance_with_G1_G2.png)

U scenariju bez `G1/G2`, model se oslanja na druge osobine učenika. Najvažniji atribut je `failures`, zatim `reason`, `absences`, `school`, `higher`, `age`, `Fedu`, `health`, `Dalc` i `studytime`.

![Važnost atributa bez G1 i G2](../data/graphs/step08_feature_importance/feature_importance_without_G1_G2.png)

Ovaj deo projekta ispunjava zahtev za odabir najznačajnijih atributa i dodatno objašnjava zašto se performanse značajno razlikuju između dva scenarija.

## 10. Top features scenario

Top features scenario proverava da li manji broj najvažnijih atributa može dati slične ili bolje rezultate od kompletnog skupa atributa. Ideja je da se dobije jednostavniji model za upotrebu i tumačenje, bez velikog gubitka kvaliteta. Korišćeni su fajlovi:

- `data/logs/step09_top_features/top_features_with_G1_G2.csv`;
- `data/logs/step09_top_features/top_features_without_G1_G2.csv`;
- `data/logs/step09_top_features/top_features_tuning_details.csv`;
- `data/logs/step09_top_features/top_features_model_comparison_report.csv`.

Top 10 atributa u scenariju sa `G1/G2` su: `G2`, `absences`, `G1`, `age`, `reason`, `freetime`, `health`, `Fedu`, `school` i `failures`.

Top 10 atributa u scenariju bez `G1/G2` su: `failures`, `reason`, `absences`, `school`, `higher`, `age`, `Fedu`, `health`, `Dalc` i `studytime`.

Sažetak poređenja:

| Scenario | Model | Validation RMSE | Test RMSE | Test R2 |
| --- | --- | ---: | ---: | ---: |
| with_G1_G2 | Ridge Regression | 1.0457 | 1.3370 | 0.8518 |
| top_features_with_G1_G2 | Ridge Regression | 0.9735 | 1.2796 | 0.8642 |
| with_G1_G2 | RandomForestRegressor | 0.8831 | 1.4138 | 0.8343 |
| top_features_with_G1_G2 | RandomForestRegressor | 0.8656 | 1.4313 | 0.8301 |
| without_G1_G2 | Ridge Regression | 2.7782 | 2.9202 | 0.2929 |
| top_features_without_G1_G2 | Ridge Regression | 2.7461 | 3.0485 | 0.2294 |
| without_G1_G2 | RandomForestRegressor | 2.7633 | 2.9108 | 0.2974 |
| top_features_without_G1_G2 | RandomForestRegressor | 2.7076 | 3.0354 | 0.2360 |

![Top features poređenje na validacionom skupu](../data/graphs/step09_top_features/validation_top_features_comparison.png)

![Top features poređenje na test skupu](../data/graphs/step09_top_features/test_top_features_comparison.png)

Top features pristup je posebno uspešan u scenariju sa `G1/G2`. `Ridge Regression` sa top atributima postiže bolji test RMSE od punog modela i koristi samo 10 atributa, što ga čini jednostavnijim za upotrebu i tumačenje.

## 11. Izbor finalnog modela i analiza predikcija

Finalni model je eksportovan u `models/final/final_model.joblib`. Prema `data/logs/step10_final_model_selection/final_model_report.csv`, izabran je:

| Stavka | Vrednost |
| --- | --- |
| Scenario | `top_features_with_G1_G2` |
| Model | `Ridge Regression` |
| Parametar | `alpha=0.01` |
| Final test MAE | 0.7655 |
| Final test RMSE | 1.2737 |
| Final test R2 | 0.8655 |

Model je izabran jer daje jak validacioni rezultat, najbolji test RMSE među poređenim kandidatima, koristi samo 10 atributa i jednostavniji je za tumačenje od Random Forest modela. Pri izboru nisu posmatrane samo metrike, već i stabilnost na test skupu, jednostavnost upotrebe i praktična interpretabilnost. Praktično ograničenje je da koristi `G1` i `G2`.

![Metrike finalnog modela](../data/graphs/step10_final_model_selection/final_model_metrics.png)

Graf stvarnih i predviđenih vrednosti pokazuje koliko su predikcije blizu idealne dijagonale. Što su tačke bliže dijagonali, model bolje predviđa završnu ocenu.

![Stvarne i predviđene vrednosti](../data/graphs/step10_final_model_selection/final_model_actual_vs_predicted.png)

Residual graf prikazuje razlike između stvarne i predviđene vrednosti. Residual je definisan kao:

`residual = stvarna G3 - predviđena G3`

![Residual analiza finalnog modela](../data/graphs/step10_final_model_selection/final_model_residuals.png)

MAE od 0.7655 znači da model u proseku greši za manje od jedne ocene. RMSE od 1.2737 pokazuje da postoje pojedinačni slučajevi sa većom greškom, ali ukupno greške ostaju relativno male. R2 od 0.8655 znači da model objašnjava veliki deo varijanse završne ocene na test skupu.

## 12. Deployment modela

Finalni model se nalazi u folderu `models/final/`. Sačuvan je kao `.joblib` fajl zajedno sa preprocessing pipeline-om i listom atributa koje očekuje. Za novu predikciju korisnik unosi originalne vrednosti atributa, a pipeline interno radi skaliranje i enkodiranje.

Projekat sadrži pomoćne fajlove za korišćenje modela:

- `src/model_usage.py`: učitavanje modela, priprema ulaza i formatiranje predikcije;
- `src/predict.py`: CLI primer za predikciju;
- `src/streamlit_app.py`: Streamlit aplikacija za unos podataka i prikaz predikcije.

Predikcija iz terminala može se pokrenuti komandom:

```powershell
.\.venv\Scripts\python.exe .\src\predict.py
```

Primer sa eksplicitnim argumentima:

```powershell
.\.venv\Scripts\python.exe .\src\predict.py --G2 13 --absences 6 --G1 12 --age 15 --reason other --freetime 3 --health 3 --Fedu 1 --school GP --failures 0
```

Streamlit aplikacija se pokreće komandom:

```powershell
streamlit run .\src\streamlit_app.py
```

Korisnik može uneti vrednosti atributa i dobiti predikciju završne ocene `G3`.

## 13. Diskusija rezultata

Poređenje scenarija sa i bez `G1/G2` pokazuje jasnu razliku. Model sa `G1/G2` je precizniji jer prethodne ocene nose mnogo informacija o završnoj oceni. To potvrđuju i korelacije iz EDA faze: `G2` ima korelaciju 0.92 sa `G3`, a `G1` korelaciju 0.83 sa `G3`.

Scenario bez `G1/G2` ima slabije performanse, ali nije bez praktične vrednosti. On može biti koristan za raniju procenu uspeha učenika, kada prethodne ocene još nisu dostupne. U tom slučaju model se oslanja na informacije kao što su prethodni neuspesi, izostanci, razlog izbora škole, želja za višim obrazovanjem, starost i navike u učenju.

Finalni model može pomoći kao podrška u proceni rizika i očekivanog uspeha, ali ne treba da bude jedini osnov za donošenje odluka. Obrazovni kontekst uključuje faktore koje dataset ne mora potpuno da obuhvati, kao što su promene motivacije, zdravstvene okolnosti, kvalitet nastave, podrška nastavnika i vanredni događaji.

Glavna ograničenja modela su:

- skup podataka je ograničen na dostupne atribute;
- finalni model koristi `G1` i `G2`, pa nije pogodan za procenu pre pojave tih ocena;
- model predviđa numeričku ocenu, ali ne objašnjava uvek uzrok slabog ili dobrog uspeha;
- rezultati zavise od kvaliteta i reprezentativnosti originalnog dataset-a.

Moguća buduća unapređenja uključuju rad sa većim i raznovrsnijim podacima, dodatne modele, detaljniju validaciju na spoljnim podacima i razvoj posebnog modela za ranu identifikaciju učenika pod rizikom.

## 14. Zaključak

U projektu je pripremljen i analiziran dataset `student-por.csv`. Urađeno je početno preprocesiranje, provereni su nedostajuće vrednosti, duplikati i očigledne greške za uklanjanje, a transformacije su pravilno pomerene u `Pipeline` kako bi se izbegao data leakage.

Sprovedena je eksplorativna analiza, trenirano je više modela, izvršeno je poređenje performansi, podešeni su hiperparametri i analizirana je važnost atributa. Posebno su upoređeni scenariji sa i bez `G1/G2`, kao i modeli sa svim atributima i modeli sa najznačajnijim atributima.

Finalni model je `Ridge Regression` u scenariju `top_features_with_G1_G2`, eksportovan u `models/final/final_model.joblib`. Model ostvaruje dobre rezultate na test skupu, sa MAE 0.7655, RMSE 1.2737 i R2 0.8655. Praktična upotrebljivost modela zavisi od dostupnosti atributa `G1` i `G2`: sa njima je model znatno precizniji, dok scenario bez njih ostaje koristan za raniju, manje preciznu procenu uspeha učenika.
