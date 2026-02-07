import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler

# Ustawienie ziarna losowości dla powtarzalności
tf.random.set_seed(42)
np.random.seed(42)

WINDOW_SIZE = 128  # Długość sekwencji wejściowej (np. 60 kroków wstecz)

def parse_args():
    parser = argparse.ArgumentParser(description="time series prediction")
    parser.add_argument('--history', type=str, required=True, help="path to csv time serioes")
    parser.add_argument('--n', type=int, default=10, help="steps to predict")
    parser.add_argument('--result', type=str, default='predictions.csv', help="result file (csv)")
    return parser.parse_args()

def prepare_data(filepath):
    """Ładuje dane, dodaje harmoniczne i normalizuje."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Plik {filepath} nie istnieje.")

    df = pd.read_csv(filepath)
    # Próba parsowania czasu - zakładamy format standardowy
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # --- HARMONICZNE ---
    # Zamiana czasu na sygnał cykliczny (sezonowość roczna)
    timestamp_s = df['date'].map(pd.Timestamp.timestamp)
    day = 24 * 60 * 60
    week = 7 * day
    year = (365.2425) * day
    
    df['sin_time'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['cos_time'] = np.cos(timestamp_s * (2 * np.pi / year))
    df['sin_week'] = np.sin(timestamp_s * (2 * np.pi / week))
    df['cos_week'] = np.cos(timestamp_s * (2 * np.pi / week))

    # Wybór cech: Close price + Harmoniczne czasu
    feature_cols = ['close', 'sin_time', 'cos_time', 'sin_week']
    data = df[feature_cols].values
    
    # Normalizacja
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    return df, data_scaled, scaler

# Data = ['close', 'sin_time', 'cos_time']
def create_sequences(data, target_col_idx=0, window_size=WINDOW_SIZE):
    """Tworzy sekwencje X (okno) -> y (następna wartość)."""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size, target_col_idx])
    return np.array(X), np.array(y)

def build_rnn_model(hp):
    """Model rekurencyjny dla Keras Tunera."""
    model = keras.Sequential()
    model.add(layers.InputLayer(shape=(WINDOW_SIZE, 3))) # 3 cechy: close, sin, cos
    
    # Wybór typu warstwy rekurencyjnej
    rnn_type = hp.Choice('rnn_type', ['LSTM', 'GRU'])
    units = hp.Int('units', min_value=32, max_value=128, step=32)
    
    if rnn_type == 'LSTM':
        model.add(layers.LSTM(units, return_sequences=False))
    else:
        model.add(layers.GRU(units, return_sequences=False))
        
    model.add(layers.Dropout(hp.Float('dropout', 0.1, 0.4, step=0.1)))
    model.add(layers.Dense(1)) # Przewidujemy jedną wartość (close)
    
    lr = hp.Float('lr', 1e-4, 1e-2, sampling='log')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss='mse',
                  metrics=['mae'])
    return model

def build_dense_model():
    """Prosty model w pełni połączony do porównania."""
    model = keras.Sequential([
        layers.InputLayer(shape=(WINDOW_SIZE, 3)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def main():
    args = parse_args()
    print(f"--- predict v1.0 ---")
    print(f"Pobieranie danych z: {args.history}")
    
    # 1. Przygotowanie danych
    df, data_scaled, scaler = prepare_data(args.history)
    
    # Tworzenie sekwencji treningowych
    X, y = create_sequences(data_scaled)
    
    # Podział na zbiór treningowy i testowy (chronologicznie)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Dane przygotowane. Treningowe: {X_train.shape}, Testowe: {X_test.shape}")

    # 2. Optymalizacja modelu RNN (Keras Tuner)
    print("\nRozpoczynanie poszukiwania najlepszych hiperparametrów (Keras Tuner)...")
    tuner = kt.RandomSearch(
        build_rnn_model,
        objective='val_loss',
        max_trials=5,
        executions_per_trial=1,
        directory='tuner_results',
        project_name='btc_prediction'
    )
    
    # patience = epochs, not trials
    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    tuner.search(X_train, y_train, epochs=15, validation_split=0.2, callbacks=[stop_early], verbose=1)
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"\nZnaleziono najlepsze parametry: {best_hps.values}")
    
    # Trening najlepszego modelu RNN
    print("Trenowanie zwycięskiego modelu RNN...")
    rnn_model = tuner.hypermodel.build(best_hps)
    history_rnn = rnn_model.fit(X_train, y_train, epochs=30, validation_split=0.2, verbose=1)
    
    # 3. Trening modelu Dense (Baseline)
    print("\nTrenowanie modelu referencyjnego (Dense)...")
    dense_model = build_dense_model()
    history_dense = dense_model.fit(X_train, y_train, epochs=30, validation_split=0.2, verbose=0)
    
    # 4. Ewaluacja i Porównanie
    loss_rnn, mae_rnn = rnn_model.evaluate(X_test, y_test, verbose=0)
    loss_dense, mae_dense = dense_model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nWyniki na zbiorze testowym:")
    print(f"RNN (Best): MAE = {mae_rnn:.4f}")
    print(f"Dense     : MAE = {mae_dense:.4f}")

    # 5. Generowanie przyszłych predykcji (Recursive Forecasting)
    print(f"\nGenerowanie {args.n} kolejnych predykcji...")
    
    # Bierzemy ostatnie dostępne okno danych
    current_seq = data_scaled[-WINDOW_SIZE:].copy() # shape (60, 3)
    last_time = df['date'].iloc[-1]
    
    # Obliczamy średni odstęp czasu w danych, żeby znać przyszłe daty
    time_diffs = df['date'].diff().dropna()
    avg_delta = time_diffs.mode()[0] if not time_diffs.empty else pd.Timedelta(days=1)
    
    future_preds_scaled = []
    future_times = []
    
    for _ in range(args.n):
        # Predykcja (wymaga kształtu (1, 60, 3))
        input_seq = current_seq.reshape(1, WINDOW_SIZE, 3)
        pred_val = rnn_model.predict(input_seq, verbose=0)[0][0]
        
        # Aktualizacja czasu
        last_time += avg_delta
        future_times.append(last_time)
        
        # Obliczenie harmonicznych dla nowego czasu
        ts = last_time.timestamp()
        day = 24 * 60 * 60
        year = 365.2425 * day
        next_sin = np.sin(ts * (2 * np.pi / year))
        next_cos = np.cos(ts * (2 * np.pi / year))
        
        # Tworzenie nowego wiersza [pred_close, sin, cos]
        new_row = np.array([pred_val, next_sin, next_cos])
        
        # Aktualizacja sekwencji (przesunięcie okna)
        current_seq = np.vstack([current_seq[1:], new_row])
        future_preds_scaled.append(pred_val)

    # Odwrócenie skalowania
    dummy = np.zeros((len(future_preds_scaled), 3))
    dummy[:, 0] = future_preds_scaled
    real_predictions = scaler.inverse_transform(dummy)[:, 0]
    
    # Zapis wyników
    res_df = pd.DataFrame({'date': future_times, 'predicted_close': real_predictions})
    res_df.to_csv(args.result, index=False)
    print(f"Zapisano predykcje do pliku: {args.result}")
    
    # 6. Wykresy i Raport
    plt.figure(figsize=(10, 6))
    plt.plot(history_rnn.history['loss'], label='RNN Train Loss')
    plt.plot(history_rnn.history['val_loss'], label='RNN Val Loss')
    plt.title('Krzywe uczenia (RNN)')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curve.png')

    plt.figure(figsize=(12, 6))
    plt.plot(df['date'].iloc[-100:], df['close'].iloc[-100:], label='Historia (ostatnie 100)')
    plt.plot(res_df['date'], res_df['predicted_close'], 'r--o', label='Predykcja (RNN)')
    plt.title(f"Prognoza na {args.n} kroków w przód")
    plt.legend()
    plt.grid(True)
    plt.savefig('prediction_chart.png')
    
    with open('raport_eksperymentu.md', 'w') as f:
        f.write("# Raport z eksperymentu predict\n\n")
        f.write("## 1. Porównanie modeli (Test Set)\n")
        f.write(f"- **RNN (Zoptymalizowany)**: MAE = {mae_rnn:.4f}, Loss (MSE) = {loss_rnn:.4f}\n")
        f.write(f"- **Dense (Baseline)**: MAE = {mae_dense:.4f}, Loss (MSE) = {loss_dense:.4f}\n\n")
        f.write("## 2. Najlepsze hiperparametry RNN\n")
        f.write(f"```json\n{best_hps.values}\n```\n\n")
        f.write("## 3. Wizualizacja\n")
        f.write("![Wykres predykcji](./prediction_chart.png)\n")
        f.write("![Krzywa uczenia](./learning_curve.png)\n")
    
    print("Wygenerowano raport: raport_eksperymentu.md")

if __name__ == "__main__":
    main()