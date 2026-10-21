# project-telco-churn

# Predykcja churnu klientów telekomunikacyjnych

Model, który wskazuje klientów firmy telekomunikacyjnej mogących odejść do konkurencji, żeby firma zdążyła zaproponować im ofertę.

## Dane

Zbiór Telco Customer Churn (Kaggle): 7043 klientów, 21 kolumn. Jeden wiersz to jeden klient, target to kolumna `Churn` (1 = odszedł). Odeszło 26,5% klientów, więc klasy są niezbalansowane.

## Struktura projektu

- `data/raw/` – surowe dane
- `data/processed/` – dane po czyszczeniu (`telco_clean.csv`)
- `notebooks/01_eda.ipynb` – analiza danych
- `notebooks/02_modeling.ipynb` – podział danych, preprocessing, modele
- `models/` – zapisany model
- `app/` – API (FastAPI)

## Najważniejsze wnioski z EDA

- Klienci na umowie miesięcznej odchodzą w 42,7% przypadków (umowa dwuletnia: 2,8%).
- Ponad połowa nowych klientów (staż do 6 miesięcy) odchodzi (52,9%).
- Fiber optic: 41,9% odejść wobec 19,0% przy DSL.
- Płatność elektronicznym czekiem: 45,3% odejść.
- TechSupport i OnlineSecurity wiążą się z wyraźnie niższym churnem (ok. 15% wobec ok. 42%).

## Metodologia

Podział train/validation/test 60/20/20 ze stratyfikacją po `Churn`, test odłożony do końcowej oceny. Preprocessing w `Pipeline` (OneHotEncoder dla kategorii, StandardScaler dla liczb).

## Metryki
Główna metryka: **recall** dla klasy churn, bo przeoczony odchodzący klient jest dla firmy kosztowniejszy niż niepotrzebny telefon do klienta, który zostanie.

Ograniczenie: **precision**, żeby model nie oznaczał wszystkich klientów. Sam recall można podbić do 100% oznaczeniem całej bazy, przy precision równej odsetkowi churnu, czyli ok. 26,5%.

Pomocniczo: **PR-AUC** i **F1** do oceny jakości rankingu i kompromisu.

## Wyniki

### Wyniki na zbiorze walidacyjnym

| Model | Recall | Precision | F1 | PR-AUC |
|---|---|---|---|---|
| DummyClassifier | 0,000 | 0,000 | 0,000 | – |
| Regresja logistyczna | 0,535 | 0,658 | 0,590 | 0,642 |

## Jak uruchomić
(uzupełnimy po zbudowaniu API: instalacja z `requirements.txt`, start serwera, przykładowe zapytanie)


