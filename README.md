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

| Model | Próg | Recall | Precision | F1 | PR-AUC |
|---|---|---|---|---|---|
| DummyClassifier (zawsze „zostanie”) | 0,50 | 0,000 | 0,000 | 0,000 | – |
| Regresja logistyczna (baseline) | 0,50 | 0,535 | 0,658 | 0,590 | 0,642 |
| Regresja logistyczna + `class_weight="balanced"` | 0,50 | 0,789 | 0,515 | 0,623 | 0,642 |
| Regresja logistyczna, obniżony próg (wstępnie) | 0,30 | 0,751 | 0,537 | 0,627 | 0,642 |

- Roboczy próg decyzyjny: 0,35, wybrany jako maksimum F1 na zbiorze walidacyjnym (F1 = 0,629, recall = 0,706, precision = 0,567; 110 przeoczonych klientów, 202 fałszywe alarmy). F1 jest jednak niemal płaskie w zakresie 0,20–0,45, więc wybór jest nieznacznie lepszy od sąsiednich progów. Próg zostanie ponownie wyznaczony po strojeniu modelu.

### Wnioski z etapu niezbalansowania klas

- Accuracy wprowadza w błąd: model, który zawsze mówi „klient zostanie”, ma accuracy 73,5%, a recall 0%.
- `class_weight="balanced"` i obniżenie progu dają ten sam efekt: zwiększają recall kosztem precision. Nie zmieniają jakości rankingu klientów (PR-AUC i ROC-AUC bez zmian).
- Próg 0,35: recall 0,706, precision 0,567, F1 0,629, PR-AUC 0,642. Wybór progu ma charakter wstępny: ostateczną decyzję podejmiemy w oparciu o koszty biznesowe po dostrojeniu modelu.
- SMOTE pominięto, bo `class_weight` i zmiana progu dały ten sam efekt bez dodatkowej biblioteki.

### Nowe cechy (feature engineering)

Sprawdzono cechy: `n_protect` (liczba z czterech dodatków ochronnych), `fiber_no_support` (światłowód bez OnlineSecurity i TechSupport), `ServiceCount` (liczba usług) i `ChargePerService` (opłata przypadająca na usługę). Żadna z nich nie poprawiła wyniku regresji logistycznej na zbiorze walidacyjnym (PR-AUC 0,642 bez zmian), a po dodaniu do modeli drzewiastych także nie dały zysku, więc nie wchodzą do modelu.

### Porównanie modeli

5-krotna walidacja krzyżowa na zbiorze treningowym (te same foldy, metryka: PR-AUC):

| Model | Nowe cechy | PR-AUC (średnia) | Odch. std. | ROC-AUC (średnia) |
|---|---|---|---|---|
| Regresja logistyczna | nie | 0,666 | 0,025 | 0,848 |
| Regresja logistyczna | tak | 0,666 | 0,025 | 0,848 |
| Random Forest | nie | 0,667 | 0,009 | 0,846 |
| Random Forest | tak | 0,667 | 0,010 | 0,846 |
| HistGradientBoosting | nie | 0,641 | 0,012 | 0,832 |
| HistGradientBoosting | tak | 0,639 | 0,010 | 0,828 |

Regresja logistyczna i Random Forest dają praktycznie ten sam wynik (różnica 0,001 jest dużo mniejsza niż odchylenie standardowe), a HistGradientBoosting wypada nieco gorzej. Wyboru modelu do dalszego strojenia dokonano w kolejnym etapie.

## Jak uruchomić
(uzupełnimy po zbudowaniu API: instalacja z `requirements.txt`, start serwera, przykładowe zapytanie)


