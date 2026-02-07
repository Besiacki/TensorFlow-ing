import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras

CONFIG = {
    "STEPS": 300,
    "INTERVAL": 5,
    "EPOCHS": 30,
    "LOOK_BACK": 20
}

def sinus_function(steps: int = CONFIG['STEPS'], interval: int = CONFIG['INTERVAL']):
    x = np.linspace(0, interval * np.pi, steps)
    y = np.sin(x)
    return x, y

def create_dataset(y, lookback):
    X, Y = [], []
    
    for i in range(len(y) - lookback):
        X.append(y[i:i+lookback])
        Y.append(y[i+lookback])
        
    return np.array(X)[..., None], np.array(Y)

def create_model(look_back: int = CONFIG['LOOK_BACK']):
    model = keras.models.Sequential([
        keras.layers.LSTM(32, input_shape=(look_back, 1)),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    x, y = sinus_function()
    
    train_size = int(len(y) * 0.7)
    train_data = y[:train_size]
    
    x_train, y_train = create_dataset(train_data, CONFIG['LOOK_BACK'])
    
    model = create_model()
    model.fit(x_train, y_train, epochs=CONFIG['EPOCHS'])
    
    current_window = train_data[-CONFIG['LOOK_BACK']:]
    current_window = current_window.reshape(1, CONFIG['LOOK_BACK'], 1)

    future_steps = len(y) - train_size
    predictions = []

    for _ in range(future_steps):
        print(current_window.shape)
        pred = model.predict(current_window)[0, 0]
        predictions.append(pred)
        new_step = np.array([[[pred]]])
        current_window = np.append(current_window[:, 1:, :], new_step, axis=1)

    plt.plot(x[:train_size], train_data, label='Train')
    plt.plot(x[train_size:], y[train_size:], label='True Future')
    plt.plot(x[train_size:], predictions, label='Predicted Future')
    plt.legend(loc='lower center', ncol=3)
    plt.show()
    
    return

if __name__ == "__main__":
    main()