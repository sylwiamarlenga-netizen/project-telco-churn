# Autor
Sylwia Marlenga-Wasik

# Predykcja churnu klientów telekomunikacyjnych

Projekt przewiduje, którzy klienci firmy telekomunikacyjnej mogą odejść do konkurencji (churn), aby firma mogła zareagować, zanim klient zrezygnuje.

## Dane

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Zbiór Telco Customer Churn (Kaggle): 7043 klientów, 21 kolumn. Jeden wiersz to jeden klient, target to kolumna `Churn` (1 = odszedł). Odeszło 26,5% klientów, więc klasy są niezbalansowane i sama accuracy nie jest dobrą metryką.

Czyszczenie: kolumna `TotalCharges` była tekstem, a 11 pustych wartości dotyczyło klientów z `tenure = 0` (nowi klienci, którzy jeszcze nic nie zapłacili), więc wypełniono je zerami. Usunięto `customerID` (identyfikator bez wartości predykcyjnej), a `Churn` zamieniono z Yes/No na 1/0.

## Struktura projektu

- `data/raw/` – surowe dane
- `data/processed/` – dane po czyszczeniu (`telco_clean.csv`)
- `notebooks/01_eda.ipynb` – analiza danych
- `notebooks/02_modeling.ipynb` – podział danych, modele, próg, interpretacja, zapis modelu
- `models/churn_model.joblib` – zapisany model (Pipeline i próg 0,38)
- `app/main.py` – API (FastAPI)
- `reports/figures/` – wykresy (EDA, macierz pomyłek, krzywa precision-recall)
- `requirements.txt` – wersje bibliotek

## Najważniejsze wnioski z EDA

EDA pokazuje związki (korelacje), a nie przyczyny.

- **Umowa miesięczna:** odchodzi 42,7% klientów (umowa roczna: 11,3%, dwuletnia: 2,8%). Umowa miesięczna odpowiada za 88,6% wszystkich odejść.
- **Nowi klienci:** w grupie ze stażem do 6 miesięcy odchodzi 52,9% klientów, a klienci ci stanowią 42% wszystkich odejść.
- **Fiber optic:** 41,9% odejść wobec 19,0% przy DSL i 7,4% bez internetu. Fiber jest też najdroższy (średnio 91,5 $ miesięcznie wobec 58,1 $ przy DSL).
- **Płatność elektronicznym czekiem:** 45,3% odejść wobec 15–19% przy pozostałych metodach.
- **TechSupport i OnlineSecurity:** wśród klientów z internetem ci, którzy je mają, odchodzą ok. 15%, a ci bez nich ok. 42%.
- **Liczba usług:** klienci z 7-9 usługami odchodzą rzadziej (odpowiednio 22,5%, 12,4%, 5,3%) niż klienci z 2 lub 3 usługami (31% i 44,9%). Klienci z dużą liczbą usług mają jednak zwykle długi staż i umowę dwuletnią, więc liczba usług może być markerem lojalności, a nie jej przyczyną.
- **Profil gospodarstwa:** najwyższy odsetek odejść mają samotni seniorzy bez dzieci (49,2%, n=561). W danych nie ma informacji, ile z tych odejść wynika ze zgonów. Osoby niebędące seniorami, samotne i bez dzieci odchodzą w 31,2% przypadków.
- **Umowa miesięczna przy dużej liczbie usług:** w grupie 7-9 usług klienci na umowie miesięcznej odchodzą w 40,4% przypadków (n=270, średnia opłata ok. 102 $). Największy łączny utracony przychód pochodzi jednak z klientów Fiber na umowie misięcznej (ok. 100 tys. $ miesięcznie) i z grupy 2-4 usług na umowie miesięcznej (ok. 65 tys. $).

Utracony przychód to suma opłat miesięcznych klientów, którzy odeszli. To uproszczenie: nie uwzględnia marży ani czasu, przez jaki klient by został.

Rekomendacje biznesowe wynikające z EDA (program dla nowych klientów, zachęty do umów rocznych i płatności automatycznych, audyt oferty Fiber) są hipotezami do zweryfikowania.

## Metodologia

- Podział train/validation/test w proporcji 60/20/20 (4225/1409/1409 klientów), stratyfikowany po `Churn` (`random_state=42`). Zbiór testowy jest odłożony i został użyty jeden raz, na końcu.
- Preprocessing w `Pipeline` i `ColumnTransformer`: `OneHotEncoder` dla 15 kolumn tekstowych, `StandardScaler` dla `tenure`, `MonthlyCharges` i `TotalCharges`, `SeniorCitizen` bez zmian. Dopasowanie (fit) wyłącznie na zbiorze treningowym, co zapobiega wyciekowi danych.

## Metryki

Główna metryka: **recall** dla klasy churn, bo przeoczony odchodzący klient jest dla firmy kosztowniejszy niż niepotrzebny kontakt z klientem, który zostanie. Ograniczenie: **precision**, ponieważ sam recall można podbić do 100% oznaczeniem całej bazy (przy precision równej odsetkowi churnu, ok. 26,5%). Pomocniczo: **PR-AUC** (jakość rankingu niezależnie od progu) i F1. Ostatecznie próg decyzyjny wybrano jednak na podstawie oczekiwanego zysku (patrz niżej).

## Wyniki na zbiorze walidacyjnym

| Model | Próg | Recall | Precision | F1 | PR-AUC |
|---|---|---|---|---|---|
| DummyClassifier (zawsze „zostanie”) | 0,50 | 0,000 | 0,000 | 0,000 | – |
| Regresja logistyczna (baseline) | 0,50 | 0,535 | 0,658 | 0,590 | 0,642 |
| Regresja logistyczna + `class_weight="balanced"` | 0,50 | 0,789 | 0,515 | 0,623 | 0,642 |
| Regresja logistyczna, próg wybrany wg zysku | 0,38 | 0,674 | 0,592 | 0,630 | 0,642 |

Wnioski:

- Accuracy wprowadza w błąd: model, który zawsze mówi „klient zostanie”, ma accuracy 73,5%, a recall 0%.
- `class_weight="balanced"` i obniżenie progu dają ten sam efekt: zwiększają recall kosztem precision. Nie zmieniają jakości rankingu klientów (PR-AUC i ROC-AUC bez zmian).

## Nowe cechy (feature engineering)

Sprawdzono cechy: `n_protect` (liczba z czterech dodatków ochronnych), `fiber_no_support` (Fiber bez OnlineSecurity i TechSupport), `ServiceCount` (liczba usług) i `ChargePerService` (opłata przypadająca na usługę). Żadna nie poprawiła regresji logistycznej na zbiorze walidacyjnym (PR-AUC 0,642 bez zmian), a po dodaniu do modeli drzewiastych też nie dały zysku, więc nie wchodzą do modelu. Cecha „umowa miesięczna” (`IsMonthToMonth`) została odrzucona, bo dubluje kolumnę z `OneHotEncoder`.

## Porównanie modeli

5-krotna walidacja krzyżowa na zbiorze treningowym (te same foldy, metryka: PR-AUC):

| Model | Nowe cechy | PR-AUC (średnia) | Odch. std. | ROC-AUC (średnia) |
|---|---|---|---|---|
| Regresja logistyczna | nie | 0,666 | 0,025 | 0,848 |
| Regresja logistyczna | tak | 0,666 | 0,025 | 0,848 |
| Random Forest | nie | 0,667 | 0,009 | 0,846 |
| Random Forest | tak | 0,667 | 0,010 | 0,846 |
| HistGradientBoosting | nie | 0,641 | 0,012 | 0,832 |
| HistGradientBoosting | tak | 0,639 | 0,010 | 0,828 |

Regresja logistyczna i Random Forest dają praktycznie ten sam wynik, a HistGradientBoosting wypada nieco gorzej. Wybrano **regresję logistyczną** jako model prostszy i łatwiejszy do wyjaśnienia, bez utraty jakości.

## Strojenie

Strojenie parametru regularyzacji `C` (od 0,003 do 30, GridSearchCV, PR-AUC) zmieniło wynik tylko o 0,006 (0,662–0,668 przy odchyleniu standardowym ok. 0,02), więc pozostawiono wartość domyślną `C=1`. Na zbiorze walidacyjnym PR-AUC wynosi 0,64. Ograniczeniem jest zawartość danych, a nie ustawienia modelu.

## Wybór progu decyzyjnego

Próg wybrano na zbiorze walidacyjnym na podstawie oczekiwanego zysku, przy jawnie przyjętych **założeniach** (nie pochodzą z danych):

- koszt akcji retencyjnej na jednego klienta: **100 $** (kontakt i oferta, np. rabat rozłożony na 12 miesięcy),
- wartość uratowanego klienta: **780 $** (12 miesięcy × ok. 65 $ przychodu, bez uwzględnienia marży),
- skuteczność oferty: **35%** zagrożonych klientów po kontakcie zostaje.

Oczekiwany zysk z trafionego alarmu wynosi więc 0,35 × 780 − 100 = 173 $, a fałszywy alarm kosztuje 100 $. Maksymalny zysk daje **próg 0,38**: recall 0,674, precision 0,592, 426 alarmów (252 trafne, 174 fałszywe), 122 przeoczonych klientów, oczekiwany zysk ok. 26,2 tys. $ na zbiorze walidacyjnym (1409 klientów). Krzywa zysku jest płaska w zakresie progów 0,30-0,50 (zysk 24-26 tys. $), więc dokładna wartość progu nie jest krytyczna.

### Analiza wrażliwości na koszt kontaktu

| Koszt kontaktu [$] | Najlepszy próg | Zysk przy najlepszym progu [$] | Zysk: kontakt ze wszystkimi [$] |
|---|---|---|---|
| 10 | 0,04 | 89 035 | 88 012 |
| 50 | 0,24 | 52 592 | 31 652 |
| 100 | 0,38 | 26 196 | −38 798 |
| 150 | 0,56 | 10 548 | −109 248 |
| 200 | 0,75 | 1 063 | −179 698 |

Próg zależy od kosztów biznesowych, a nie tylko od modelu. Przy bardzo taniej akcji (ok. 10 $) opłaca się kontaktować prawie ze wszystkimi i model niewiele wnosi, natomiast im droższa akcja retencyjna, tym większa wartość modelu, który wskazuje, do kogo warto się zwrócić. Im droższa akcja, tym jednak mniejszy zysk netto.

## Ograniczenia

- Założenia kosztowe (koszt kontaktu, wartość klienta, skuteczność oferty) są przyjęte, a nie zmierzone. Wartość klienta jest liczona z przychodu, a nie z marży.
- Zysk jest liczony na zbiorze walidacyjnym i jest przybliżeniem; ostateczna ocena jest na zbiorze testowym (sekcja niżej).
- Dane nie zawierają informacji o przyczynach odejść (np. zgony wśród seniorów, oferty konkurencji), a EDA pokazuje korelacje, nie związki przyczynowe.

## Wynik na zbiorze testowym

Finalny model (regresja logistyczna, `C=1`) nauczono na zbiorze treningowym i walidacyjnym łącznie (80% danych) i oceniono jednokrotnie na zbiorze testowym (1409 klientów) przy progu 0,38 wybranym na walidacji.

| Metryka | Wynik | 95% przedział (bootstrap) |
|---|---|---|
| Recall | 0,687 | 0,638–0,732 |
| Precision | 0,560 | 0,513–0,604 |
| F1 | 0,617 | – |
| PR-AUC | 0,633 | – |
| ROC-AUC | 0,842 | – |

Macierz pomyłek (wiersze: stan faktyczny, kolumny: predykcja): 833 trafnie „zostanie”, 202 fałszywe alarmy, 117 przeoczonych klientów, 257 trafnie wyłapanych odejść. Oczekiwany zysk netto przy założeniach kosztowych wynosi ok. 24,3 tys. $ na 1409 klientów (na walidacji: 26,2 tys. $).

Wynik jest zbliżony do walidacyjnego (recall 0,674, precision 0,592, PR-AUC 0,642), co wskazuje, że model i wybór progu nie przeuczyły się na zbiorze walidacyjnym. Wyniki wynikają z założeń kosztowych opisanych wyżej i nie zostały zweryfikowane w praktyce.

## Green IT: efektywność modeli

Porównanie trzech modeli w tych samych ustawieniach co w walidacji krzyżowej (Random Forest: `n_estimators=300`, `min_samples_leaf=5`). Czas treningu to mediana z 3 pomiarów na moim komputerze, PR-AUC liczono na zbiorze walidacyjnym:

| Model | PR-AUC | Czas treningu [s] | Predykcja / 1000 klientów [s] | Rozmiar [KB] | PR-AUC na sekundę treningu |
|---|---|---|---|---|---|
| Regresja logistyczna | 0,642 | 0,052 | 0,0082 | 7,8 | 12,31 |
| Random Forest | 0,631 | 0,945 | 0,0441 | 12 154,3 | 0,67 |
| HistGradientBoosting | 0,632 | 0,304 | 0,0125 | 370,3 | 2,08 |

Jakość jest praktycznie taka sama (PR-AUC 0,631–0,642, różnice mniejsze niż odchylenie standardowe z walidacji krzyżowej), a koszt bardzo różny. Wybrana regresja logistyczna trenuje się ok. 18 razy szybciej niż Random Forest i ok. 6 razy szybciej niż HistGradientBoosting, a jej plik jest ok. 1500 razy mniejszy niż plik Random Forest (8 KB wobec 12 MB). Droższe modele nie dają w zamian lepszej jakości, więc nie ma powodu zużywać więcej zasobów.

Dodatkowe świadome wybory:
- pominięcie SMOTE (zmiana progu i `class_weight` dają ten sam efekt bez dodatkowych obliczeń),
- zawężone strojenie do jednego parametru (`C`), bo szerszy przegląd nie poprawiał wyniku,
- mały plik modelu (8 KB), dzięki czemu API startuje od razu.

Uwaga: czasy zależą od sprzętu i są podane orientacyjnie do porównania modeli, a nie jako bezwzględne wartości.

## Interpretacja modelu

Największy wpływ zmniejszający ryzyko odejścia mają: długi staż (`tenure`), umowa dwuletnia, internet DSL oraz brak internetu. Największy wpływ zwiększający ryzyko mają: internet Fiber optic, umowa miesięczna oraz wyższy `TotalCharges`. Przy równych pozostałych cechach umowa miesięczna zwiększa szanse odejścia ok. 3,9 raza wobec dwuletniej, a Fiber ok. 3,7 raza wobec DSL.

Cechy, które w EDA wyglądały na silne, mają w modelu mniejszy samodzielny wpływ, bo ich efekt jest w dużej mierze wyjaśniony przez umowę, staż i rodzaj internetu: płatność elektronicznym czekiem (ok. 1,5 raza wyższe szanse wobec karty kredytowej), brak OnlineSecurity (ok. 1,4 raza) i brak TechSupport (ok. 1,3 raza), a także bycie seniorem (ok. 1,15 raza) i brak dzieci na utrzymaniu (ok. 1,2 raza). Cechy te nadal przydają się do wskazywania segmentów ryzyka (np. samotni seniorzy bez dzieci odchodzą w 49,2% przypadków, n=561), ale nie należy ich traktować jako głównych przyczyn odejść.

Uwagi: 
- współczynniki skorelowanych cech (np. `MonthlyCharges`, Fiber, usługi streamingowe, `TotalCharges`) nie powinny być czytane pojedynczo; ujemny współczynnik `MonthlyCharges` nie oznacza, że wyższa opłata zmniejsza ryzyko; 
- wyniki opisują związki, a nie przyczyny, więc działania biznesowe należy sprawdzić testem A/B.

## Uruchomienie

Projekt testowano na Pythonie 3.14.

1. Sklonuj repozytorium i wejdź do jego folderu.
2. Zainstaluj biblioteki:

```
   py -m pip install -r requirements.txt
```

3. Uruchom API (w głównym folderze projektu):

```
   py -m uvicorn app.main:app --reload
```

4. Otwórz w przeglądarce `http://127.0.0.1:8000/docs`. Znajdziesz tam interaktywną dokumentację, w której można wysłać zapytanie bez dodatkowych narzędzi (**Try it out**, potem **Execute**).

Zapisany model (`models/churn_model.joblib`) jest w repozytorium, więc do uruchomienia API nie trzeba go uczyć. Aby odtworzyć analizę i model, uruchom po kolei notebooki `notebooks/01_eda.ipynb` i `notebooks/02_modeling.ipynb` (`py -m jupyter notebook`). Zapisany model wymaga tej samej wersji `scikit-learn` co w `requirements.txt`.

### API

- `GET /health` sprawdzenie, czy aplikacja działa. Odpowiedź: `{"status": "ok"}`.
- `POST /predict` dane klienta (19 pól jak w zbiorze, bez `customerID` i `Churn`). Zwraca prawdopodobieństwo odejścia, decyzję według progu 0,38 i użyty próg. Błędne dane (nieznana wartość, brakujące pole, ujemny staż) kończą się odpowiedzią 422 z opisem błędu.

Przykładowe zapytanie:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 14,
  "PhoneService": "Yes",
  "MultipleLines": "Yes",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "Yes",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 104.85,
  "TotalCharges": 1468.9
}
```

Odpowiedź:

```json
{
  "churn_probability": 0.7505,
  "churn_prediction": true,
  "threshold": 0.38
}
```