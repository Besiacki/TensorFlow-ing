import pandas as pd
import sys
import os
import argparse
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import keras_tuner as kt
from sklearn.model_selection import train_test_split

config = {
    'EPOCHS': 100,
    'BATCH_SIZE': 16,
    'MODEL_FILENAME': 'best_wine_model.keras',
    'BASELINE_FILENAME': 'baseline_model.keras'
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
        
        # Separating data into features (had to convert to float32 for keras_tuner compatibility)
        x = df_shuffled.drop(df_shuffled.columns[0], axis=1).values.astype('float32')
        
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
    
def get_normalization_layer(x_train):
    normalizer = keras.layers.Normalization(axis=-1)
    normalizer.adapt(x_train)
    return normalizer

def build_baseline_model(input_shape, normalizer):
    model = keras.Sequential([
        keras.layers.InputLayer(shape=(input_shape,)),
        normalizer,
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_tuner_model(hp, input_shape, normalizer):
    model = keras.Sequential()
    
    model.add(keras.layers.InputLayer(shape=(input_shape,)))
    model.add(normalizer)
    
    hp_units_1 = hp.Int('units_1', min_value=32, max_value=128, step=32)
    hp_activation = hp.Choice('activation', values=['relu', 'tanh'])
    
    model.add(keras.layers.Dense(units=hp_units_1, activation=hp_activation))
    
    if hp.Boolean('extra_layer'):
         hp_units_2 = hp.Int('units_2', min_value=16, max_value=64, step=16)
         model.add(keras.layers.Dense(units=hp_units_2, activation=hp_activation))

    model.add(keras.layers.Dense(3, activation='softmax'))
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def predict_wine_quality(model, features):
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    predicted_class = np.argmax(prediction, axis=1)[0] + 1
    conf = np.max(prediction) * 100
    return predicted_class, conf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wine Quality Classification with Keras Tuner")
    parser.add_argument('--predict', type=float, nargs=13, help='Provide 13 feature values to predict wine quality')
    args = parser.parse_args()
    
    DATA_PATH = "lab3/wine_data.csv"
    
    # Case 1: Prediction mode
    if args.predict:
        if os.path.exists(config['MODEL_FILENAME']):
            print(f"Loading best model from {config['MODEL_FILENAME']}...")
            model = keras.models.load_model(config['MODEL_FILENAME'])
            
            predicted_class, conf = predict_wine_quality(model, args.predict)
            print("\n" + "="*30)
            print(f"Input: {args.predict}")
            print(f"Class: {predicted_class}")
            print(f"Conf:  {conf:.2f}%")
            print("="*30 + "\n")
        else:
            print("Model file not found. Run without arguments first to train and tune.")

    # Case 2: Training and tuning mode
    else:
        print("--- Loading Data ---")
        data = load_wine_dataset(DATA_PATH)
        if data is None:
            sys.exit(1)
            
        x_train, x_test, y_train, y_test = data
        input_shape = x_train.shape[1]
        
        print("--- Adapting Normalization Layer ---")
        normalizer = get_normalization_layer(x_train)

        # 1. BASELINE MODEL
        print("\n--- Training Baseline Model (Fixed Params) ---")
        baseline_model = build_baseline_model(input_shape, normalizer)
        baseline_history = baseline_model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=config['EPOCHS'],
            batch_size=config['BATCH_SIZE'],
            verbose=0
        )
        
        baseline_val_acc = baseline_history.history['val_accuracy'][-1]
        print(f"BASELINE Validation Accuracy: {baseline_val_acc:.4f}")
        baseline_model.save(config['BASELINE_FILENAME'])

        # 2. KERAS TUNER
        print("\n--- Starting Keras Tuner ---")
        
        # Building the model wrapper for the tuner because it requires only one argument
        def model_builder_wrapper(hp):
            return build_tuner_model(hp, input_shape, normalizer)

        tuner = kt.RandomSearch(
            model_builder_wrapper,
            objective='val_accuracy',
            max_trials=10,
            executions_per_trial=1,
            directory='tuner_dir',
            project_name='wine_quality_tuning'
        )

        tuner.search_space_summary()

        tuner.search(x_train, y_train, 
                     epochs=30,
                     validation_data=(x_test, y_test),
                     callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)])

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        
        print(f"""
        The hyperparameter search is complete. 
        Best units_1: {best_hps.get('units_1')}
        Best activation: {best_hps.get('activation')}
        Best learning rate: {best_hps.get('learning_rate')}
        Extra layer used: {best_hps.get('extra_layer')}
        """)

        # 3. RETRAINING BEST MODEL
        print("\n--- Retraining Best Model ---")
        best_model = tuner.hypermodel.build(best_hps)
        history = best_model.fit(x_train, y_train, 
                                 epochs=config['EPOCHS'], 
                                 validation_data=(x_test, y_test),
                                 verbose=1)
        
        best_val_acc = history.history['val_accuracy'][-1]
        print(f"\nBASELINE Val Acc: {baseline_val_acc:.4f}")
        print(f"TUNED Val Acc:    {best_val_acc:.4f}")

        if best_val_acc >= baseline_val_acc:
            print("SUCCESS: Tuned model is equal or better than baseline.")
        else:
            print("NOTE: Baseline was better (might happen on small datasets).")

        # Saving the best model
        best_model.save(config['MODEL_FILENAME'])
        print(f"Best model saved to {config['MODEL_FILENAME']}")