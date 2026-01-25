Raport: Klasyfikacja Wina z użyciem Sieci Neuronowych
=====================================================

### 1\. Opis zaimplementowanych modeli

W celu rozwiązania problemu klasyfikacji gatunków wina (3 klasy) przygotowano i przetestowano dwie różne architektury sieci neuronowych typu Sequential:

*   **Model A (Wersja "Lekka"):** Prosta sieć składająca się z warstwy wejściowej, dwóch warstw ukrytych (64 i 32 neurony) oraz warstwy wyjściowej. Jest to klasyczna struktura typu "lejek" (zmniejszająca się liczba neuronów), zaprojektowana do szybkiego uczenia się na prostych danych tabelarycznych.
    
*   **Model B (Wersja "Ciężka" z Dropout):** Znacznie bardziej rozbudowana sieć (start od 128 neuronów). Zastosowano w niej mechanizm **Dropout** (wyłączanie losowych neuronów), który zazwyczaj służy do zapobiegania "kuciu na pamięć" (overfittingowi) w bardzo dużych sieciach.
    

### 2\. Krzywe uczenia oraz dokładność

![graph](./Model%20graphs.png)

Na podstawie wykresów wygenerowanych podczas treningu:

*   **Model A (Niebieska linia):** Proces uczenia przebiegał wzorowo. Dokładność (accuracy) bardzo szybko wzrosła do **100%**, a błąd (loss) spadł niemal do zera. Wykres jest stabilny i gładki.
    
*   **Model B (Czerwona linia):** Proces był chaotyczny. Wykres jest poszarpany, co oznacza, że sieć miała problemy ze stabilizacją wag. Ostateczna dokładność na zbiorze walidacyjnym wyniosła jedynie ok. **72%**, a funkcja straty utrzymywała się na wysokim poziomie.
    

### 3\. Wyniki i wnioski - dlaczego tak wyszło?

Zdecydowanym zwycięzcą eksperymentu jest **Model A**.

**Dlaczego Model B (teoretycznie bardziej zaawansowany) poradził sobie gorzej?**Głównym powodem jest **niedopasowanie narzędzia do problemu**.

1.  **Zbyt mały zbiór danych:** Mamy tylko 178 próbek wina. Model B to "armata na wróbla" - posiadał za dużo parametrów jak na tak małą ilość informacji. Zamiast się uczyć, błądził.
    
2.  **Destrukcyjny wpływ Dropoutu:** Dropout (odrzucanie 30% informacji w każdym kroku) jest świetny przy tysiącach zdjęć, ale przy 178 wierszach danych działał na szkodę sieci. Model nie mógł ustalić pewnych reguł, bo ciągle "wyłączaliśmy mu prąd" w losowych miejscach.
    
3.  **Zasada prostoty:** W przypadku prostych danych z wyraźnymi zależnościami chemicznymi, mniejsza i prostsza sieć (Model A) szybciej znajduje rozwiązanie i jest bardziej stabilna.