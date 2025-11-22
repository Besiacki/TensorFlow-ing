import sys
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import Tuple
import os
import datetime

# ----- Data processing Functions -----

def load_dataset(dataset_filename: str ="wine_data.csv") -> pd.DataFrame | None:
    """
    Loads provided csv file into a pandas DataFrame
    
    Args:
        dataset_filename (str, optional): Path to csv file containing dataset. Defaults to "wine_data.csv".

    Returns:
        pd.DataFrame | None: 
    """
    try:
        df = pd.read_csv(dataset_filename)
        print(f"Success: Loaded {len(df)} rows from {dataset_filename}.")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{dataset_filename}' was not found.", file=sys.stderr)
        return None
    
def shuffle_dataset(dataset: pd.DataFrame ,frac: float = 1, random_state: int = 42, drop: bool = True) -> pd.DataFrame:
    """
    Shuffles the given dataset

    Args:
        dataset (pd.DataFrame): Input dataset
        frac (float, optional): Fraction of axis items to return. Defaults to 1.
        random_state (int, optional): Seed for random number generator. Defaults to 42.
        drop (bool, optional): Whether to drop previous index column. Defaults to True.

    Returns:
        pd.DataFrame: _description_
    """
    return dataset.sample(frac=frac, random_state=random_state).reset_index(drop=False)
    
    
def prepare_data(df: pd.DataFrame, target_column_index: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separates features (X) and targets (y), and performs One-Hot Encoding on the target.
    
    Args:
        df (pd.DataFrame): The shuffled dataframe.
        target_column_index (int, optional): Index of the column containing labels. Defaults to 0 (first column).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing (X_features, y_encoded_targets).
    """
    # 1. Separate Features (X)
    X = df.drop(df.columns[target_column_index], axis=1).values
    
    # 2. Separate Target (y)
    y = df.iloc[:, target_column_index].values
    
    # 3. Shift labels (Wine dataset is 1,2,3 -> needs to be 0,1,2)
    y_shifted = y - np.min(y)
    
    # 4. One-Hot Encode
    # We use len(unique) to determine how many columns we need
    num_classes = len(np.unique(y_shifted))
    y_encoded = tf.keras.utils.to_categorical(y_shifted, num_classes=num_classes)
    
    print(f"Data Prepared: Features shape: {X.shape}, Targets shape: {y_encoded.shape}")
    return X, y_encoded

def split_dataset(X: np.ndarray, y: np.ndarray, split_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the features and targets into Training and Testing sets manually.

    Args:
        X (np.ndarray): The feature matrix.
        y (np.ndarray): The target matrix (one-hot encoded).
        split_ratio (float, optional): The ratio of data to use for training. Defaults to 0.8 (80%).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, y_train, X_test, y_test
    """
    # Calculate the index where to cut the data
    train_size = int(len(X) * split_ratio)
    
    # Slice the arrays
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    print(f"Split Complete: {len(X_train)} Training samples, {len(X_test)} Test samples.")
    return X_train, y_train, X_test, y_test

# ----- Model and visualization functions -----

def build_model_v1(input_shape: int, num_classes: int) -> tf.keras.Model:
    """Builds a simple sequential neural network model (Version 1).

    Architecture: Input -> Dense(16, ReLU) -> Output(Softmax).

    Args:
        input_shape (int): The number of input features.
        num_classes (int): The number of output classes.

    Returns:
        tf.keras.Model: A compiled Keras model ready for training.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape,)),
        tf.keras.layers.Dense(16, activation='relu', name='Hidden_Layer_ReLu'),
        tf.keras.layers.Dense(num_classes, activation='softmax', name='Output_Softmax')
    ], name="Model_V1_Simple")
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def build_model_v2(input_shape: int, num_classes: int) -> tf.keras.Model:
    """Builds a deeper neural network model with He initialization (Version 2).

    Architecture: Input -> Dense(64, ReLU, He) -> Dense(32, Tanh) -> Output(Softmax).

    Args:
        input_shape (int): The number of input features.
        num_classes (int): The number of output classes.

    Returns:
        tf.keras.Model: A compiled Keras model ready for training.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform', name='Hidden_1_He'),
        tf.keras.layers.Dense(32, activation='tanh', name='Hidden_2_Tanh'),
        tf.keras.layers.Dense(num_classes, activation='softmax', name='Output')
    ], name="Model_V2_Deep")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(model: tf.keras.Model, 
                X_train: np.ndarray, y_train: np.ndarray, 
                X_test: np.ndarray, y_test: np.ndarray, 
                epochs: int = 50, batch_size: int = 16) -> tf.keras.callbacks.History:
    """Trains the Keras model and logs progress to TensorBoard.

    Args:
        model (tf.keras.Model): The compiled Keras model to train.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.
        X_test (np.ndarray): Validation features.
        y_test (np.ndarray): Validation targets.
        epochs (int): Number of epochs to train. Defaults to 50.
        batch_size (int): Number of samples per gradient update. Defaults to 16.

    Returns:
        tf.keras.callbacks.History: A History object containing training metrics.
    """
    print(f"\n--- Training {model.name} ---")
    
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + model.name
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[tensorboard_callback],
        verbose=1
    )
    return history


def plot_history(history: tf.keras.callbacks.History, title: str = "Model Training History") -> None:
    """Plots training and validation accuracy and loss over epochs.

    Args:
        history (tf.keras.callbacks.History): The history object returned by model.fit().
        title (str): The title for the plots. Defaults to "Model Training History".
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'{title} - Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'{title} - Loss')
    
    plt.show()


# --- MAIN EXECUTION ---

def main() -> None:
    """Main entry point for the script. Handles command-line arguments."""
    
    parser = argparse.ArgumentParser(description="Lab 03 - Wine Classification Neural Network")
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Mode: train or predict')
    
    # Train parser
    parser_train = subparsers.add_parser('train', help='Train models and save the best one.')
    parser_train.add_argument('--epochs', type=int, default=60, help='Number of epochs')
    parser_train.add_argument('--batch', type=int, default=16, help='Batch size')
    
    # Predict parser
    parser_predict = subparsers.add_parser('predict', help='Predict wine class based on features.')
    parser_predict.add_argument('--features', nargs='+', type=float, required=True, 
                                help='List of 13 float features for the wine.')
    
    args = parser.parse_args()

    if args.mode == 'train':
        data = load_dataset()
        if data is None: return

        data = shuffle_dataset(dataset=data)
        X, y, num_classes = prepare_data(data)
        X_train, y_train, X_test, y_test = split_dataset(X, y)
        
        input_shape = X_train.shape[1]

        # Model V1
        model_v1 = build_model_v1(input_shape, num_classes)
        history_v1 = train_model(model_v1, X_train, y_train, X_test, y_test, args.epochs, args.batch)
        plot_history(history_v1, title="Model V1 (Simple)")
        
        # Model V2
        model_v2 = build_model_v2(input_shape, num_classes)
        history_v2 = train_model(model_v2, X_train, y_train, X_test, y_test, args.epochs, args.batch)
        plot_history(history_v2, title="Model V2 (Deep)")
        
        # Save best
        acc_v1 = history_v1.history['val_accuracy'][-1]
        acc_v2 = history_v2.history['val_accuracy'][-1]
        
        print(f"\nValidation Accuracy -> V1: {acc_v1:.4f}, V2: {acc_v2:.4f}")
        
        best_model = model_v2 if acc_v2 > acc_v1 else model_v1
        best_model_name = "wine_model_best.keras"
        best_model.save(best_model_name)
        print(f"Best model saved as: {best_model_name}")

    elif args.mode == 'predict':
        model_filename = "wine_model_best.keras"
        
        if not os.path.exists(model_filename):
            print("Error: Model not found! Run 'train' mode first.")
            return

        model = tf.keras.models.load_model(model_filename)
        
        features = np.array(args.features)
        if len(features) != 13:
            print(f"Error: Expected 13 features, got {len(features)}.")
            return
        
        features_tensor = np.expand_dims(features, axis=0)
        prediction = model.predict(features_tensor)
        predicted_class_index = np.argmax(prediction)
        final_class = predicted_class_index + 1
        
        print(f"\n--- PREDICTION RESULT ---")
        print(f"Probabilities: {prediction}")
        print(f"Predicted Wine Class: {final_class}")

if __name__ == "__main__":
    main()