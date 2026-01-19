# Raport Lab 3 - Klasyfikacja Win

## Opis Modeli

W ramach zadania przygotowano i przetestowano dwa modele sieci neuronowych typu Sequential w bibliotece TensorFlow/Keras. Oba modele poprzedzone są warstwą `Normalization`, która dokonuje standaryzacji cech wejściowych (średnia=0, wariancja=1), co jest kluczowe dla zbieżności algorytmu przy tak zróżnicowanych zakresach danych (np. Proline ~1000 vs Phenols ~1).

### Model V1 (Prosty)
*   **Architektura**: Input (13) -> Normalization -> Dense(16, ReLU) -> Output(3, Softmax)
*   **Charakterystyka**: Płytka sieć z jedną warstwą ukrytą.
*   **Liczba parametrów**: Niewielka, szybki czas treningu.

### Model V2 (Głębszy)
*   **Architektura**: Input (13) -> Normalization -> Dense(64, ReLU, He Uniform) -> Dense(32, Tanh) -> Output(3, Softmax)
*   **Charakterystyka**: Głębsza sieć z dwiema warstwami ukrytymi, różnymi funkcjami aktywacji (ReLU, Tanh) oraz inicjalizacją wag metodą He.
*   **Optymalizator**: Adam (learning_rate=0.001)

## Wyniki Treningu

Trening przeprowadzono na 15 epokach z podziałem na zbiór treningowy (80%) i testowy (20%).

### Krzywe Uczenia

**Model V1 (Simple):**
![Krzywe uczenia V1](model_v1_(simple).png)

**Model V2 (Deep):**
![Krzywe uczenia V2](model_v2_(deep).png)

### Dokładność (Validation Accuracy)
*   **Model V1**: ~91.67%
*   **Model V2**: ~94.44%

## Wnioski

Obydwa modele osiągają bardzo dobre wyniki, jednak **Model V2 (Głębszy)** systematycznie osiąga wyższą dokładność i szybciej zbiega do minimalnego błędu.

**Dlaczego Model V2 jest lepszy?**
1.  **Głębokość sieci**: Dodatkowa warstwa ukryta pozwala na modelowanie bardziej złożonych, nieliniowych zależności między cechami chemicznymi wina a jego kategorią.
2.  **Inicjalizacja He**: Zastosowanie inicjalizacji wag `he_uniform` jest lepiej dopasowane do funkcji aktywacji ReLU, co przyspiesza zbieżność w początkowej fazie uczenia w porównaniu do domyślnej inicjalizacji Glorot (Xavier) w V1.
3.  **Większa pojemność**: Większa liczba neuronów (64, 32 vs 16) pozwala sieci "zapamiętać" i uogólnić więcej wzorców ze zbioru treningowego.

Ze względu na lepsze parametry, Model V2 został wybrany jako model docelowy i zapisany do pliku `wine_model_best.keras`.

## Uruchomienie Predykcji

Program umożliwia predykcję klasy wina na podstawie podanych parametrów z wiersza poleceń. Przykład:

```bash
python classification.py predict --features 14.23 1.71 2.43 15.6 127 2.8 3.06 .28 2.29 5.64 1.04 3.92 1065
```

Zwróci przewidywaną klasę (np. 1, 2 lub 3) z wysokim prawdopodobieństwem dzięki wbudowanej w model warstwie normalizacji.
