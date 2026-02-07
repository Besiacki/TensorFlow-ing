import pandas as pd
import sys
from sklearn.model_selection import train_test_split
import keras
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import argparse
import os

config = {
    'EPOCHS': 100,
    'LEARNING_RATE': 0.001,
    'BATCH_SIZE': 16,
    'MODEL_FILENAME': 'wine_quality_model_a.keras'
}


def load_wine_dataset(path: str = "wine_data.csv", frac: float = 1.0, test_size: float = 0.2, random_state: int = 42):
    """
    Loads the Wine Quality dataset based on the file https://archive.ics.uci.edu/dataset/109/wine
    Hot encodes the 'quality' column into multiple binary columns
    and splits the dataset into training and testing sets.
    """
    if not path:
        print("Please provide a valid path to the dataset CSV file.")
        return None
    
    try:
        # Load the dataset
        df = pd.read_csv(path)
        print(f"Success: Loaded {len(df)} rows from {path}.")
        
        # Shuffle the dataset
        df_shuffled = df.sample(frac=frac, random_state=random_state).reset_index(drop=True)
        
        # Separating data into features
        x = df_shuffled.drop(df_shuffled.columns[0], axis=1)
        
        # Separating data into labels
        y = df_shuffled.iloc[:, 0].values

        # Shifting labels into 1 less values as they have to start from index 0
        y_shifted = y - 1
        
        num_classes = len(set(y_shifted))
        
        # One-hot encoding the labels
        y_encoded = keras.utils.to_categorical(y_shifted, num_classes=num_classes)

        # Splitting the dataset into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=test_size, random_state=random_state)

        return x_train, x_test, y_train, y_test

    except FileNotFoundError:
        print(f"Error: The file '{path}' was not found.", file=sys.stderr)
        return None
    
# Building both models
    
def build_model_A(input_shape: int):
    model = keras.Sequential([
        keras.layers.InputLayer(shape=(input_shape,), name='input_layer'),
        keras.layers.Dense(64, activation='relu', name='dense_64'),
        keras.layers.Dense(32, activation='relu', name='dense_32'),
        keras.layers.Dense(3, activation='softmax', name='output_layer')],
        name="Model_A"
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config['LEARNING_RATE']), 
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    
    return model

def build_model_B(input_shape: int):
    model = keras.Sequential([
        keras.layers.InputLayer(shape=(input_shape,), name='input_layer'),
        keras.layers.Dense(128, activation='relu', name='dense_128'),
        keras.layers.Dropout(rate=0.3, name='1_dropout_0.3'),
        keras.layers.Dense(64, activation='relu', name='dense_64'),
        keras.layers.Dropout(rate=0.2, name='2_dropout_0.2'),
        keras.layers.Dense(32,activation='relu', name='dense_32'),
        keras.layers.Dense(3, activation='softmax', name='output_layer')],
        name="Model_B"
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config['LEARNING_RATE']), 
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    
    return model

def train_model(model, x_train, y_train, x_val, y_val):
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=config['EPOCHS'],
        batch_size=config['BATCH_SIZE'],
        verbose=1
    )
    return history

def plot_training_history(history_a, history_b):
    acc_a = history_a.history['accuracy']
    val_acc_a = history_a.history['val_accuracy']
    loss_a = history_a.history['loss']
    val_loss_a = history_a.history['val_loss']

    acc_b = history_b.history['accuracy']
    val_acc_b = history_b.history['val_accuracy']
    loss_b = history_b.history['loss']
    val_loss_b = history_b.history['val_loss']

    epochs = range(1, len(acc_a) + 1)

    plt.figure(figsize=(14, 6))

    # --- Plot 1: Accuracy ---
    plt.subplot(1, 2, 1)
    # Model A
    plt.plot(epochs, acc_a, 'b-', label='Model A Train')
    plt.plot(epochs, val_acc_a, 'b--', label='Model A Val')
    # Model B
    plt.plot(epochs, acc_b, 'r-', label='Model B Train')
    plt.plot(epochs, val_acc_b, 'r--', label='Model B Val')
    
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # --- Plot 2: Loss ---
    plt.subplot(1, 2, 2)
    # Model A
    plt.plot(epochs, loss_a, 'b-', label='Model A Train')
    plt.plot(epochs, val_loss_a, 'b--', label='Model A Val')
    # Model B
    plt.plot(epochs, loss_b, 'r-', label='Model B Train')
    plt.plot(epochs, val_loss_b, 'r--', label='Model B Val')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
def predict_wine_quality(model, features):
    
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    predicted_class = np.argmax(prediction, axis=1)[0] + 1  # Adjusting back to original class labels
    conf = np.max(prediction) * 100
    
    return predicted_class, conf


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Wine Quality Classification")
    parser.add_argument('--predict', type=float, nargs=13, help='Provide 13 feature values to predict wine quality')
    
    args = parser.parse_args()
    
    DATA_PATH = "wine_data.csv" 

    # Case 1: Prediction mode
    if args.predict:
        model = None
        
        # Looking for existing model
        if os.path.exists(config['MODEL_FILENAME']):
            print(f"Loading existing model from {config['MODEL_FILENAME']}...")
            model = keras.models.load_model(config['MODEL_FILENAME'])
        else:
            print(f"Model file '{config['MODEL_FILENAME']}' not found!")
            print("Training a new model before prediction...")
            
            data = load_wine_dataset(DATA_PATH)
            if data is None:
                sys.exit(1)
                
            x_train, x_test, y_train, y_test = data
            input_shape = x_train.shape[1]
            
            model = build_model_A(input_shape)
            train_model(model, x_train, y_train, x_test, y_test)
            
            model.save(config['MODEL_FILENAME'])
            print(f"Model saved to {config['MODEL_FILENAME']}")
        
        # Making prediction
        predicted_class, conf = predict_wine_quality(model, args.predict)
        
        print("\n" + "="*30)
        print(f"Input features: {args.predict}")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {conf:.2f}%")
        print("="*30 + "\n")
        
    # Case 2: Training mode
    else:
        print("No prediction arguments provided. Starting training mode...")
        data = load_wine_dataset(DATA_PATH)
        
        if data is not None:
            x_train, x_test, y_train, y_test = data
            input_shape = x_train.shape[1]
            
            # Training model and saving it
            model_A = build_model_A(input_shape)
            print("\n--- Training Model A ---")
            history_A = train_model(model_A, x_train, y_train, x_test, y_test)
            
            model_A.save(config['MODEL_FILENAME'])
            print(f"\nModel A saved successfully to '{config['MODEL_FILENAME']}'")
            # ----------------------------

            # Training Model B
            model_B = build_model_B(input_shape)
            print("\n--- Training Model B ---")
            history_B = train_model(model_B, x_train, y_train, x_test, y_test)
            
            print("\nFinal Validation Accuracy:")
            print(f"Model A: {history_A.history['val_accuracy'][-1]:.4f}")
            print(f"Model B: {history_B.history['val_accuracy'][-1]:.4f}")
            
            plot_training_history(history_A, history_B)
            
            #python classification.py --predict 14.23 1.71 2.43 15.6 127 2.8 3.06 0.28 2.29 5.64 1.04 3.92 1065 