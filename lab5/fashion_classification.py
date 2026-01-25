import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers
import keras_tuner as kt
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

CONFIG = {
    'EPOCHS': 10,
    'MAX_TRIALS': 5,
    'EXEC_PER_TRIAL': 1,
    'BATCH_SIZE': 64,
    'MODEL_FILE': 'best_clothes_model.keras',
    'METRICS_FILE': 'model_metrics.txt',
    'CONF_MATRIX_FILE': 'confusion_matrix.txt'
}

def load_data():
    fashion_mnist = keras.datasets.fashion_mnist
    
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    print(f"Training images shape:{train_images.shape}")
    print(f"Test images shape:{test_images.shape}")

    train_images = train_images.astype('float32') /255
    test_images = test_images.astype('float32') / 255

    return train_images, train_labels, test_images, test_labels

def show_image_with_colormap(image: np.ndarray):
    plt.figure(figsize=(5,5))
    plt.imshow(image)
    plt.colorbar()
    plt.show()

def build_model(hp):
    model = keras.Sequential()
    
    model.add(layers.InputLayer(shape=(28,28,1)))
    
    if hp.Boolean('use_augmentation', default=True):
        model.add(layers.RandomFlip("horizontal"))
        model.add(layers.RandomRotation(0.1))
        model.add(layers.RandomZoom(0.1))
    
    model_type = hp.Choice('model_type', ['dense', 'cnn'])
    
    if model_type == 'dense':
        model.add(layers.Flatten())
        
        for i in range(hp.Int('dense_layers', 1, 3)):
            model.add(layers.Dense(
                units=hp.Int(f'dense_units_{i}', 32, 256, step=32),
                activation='relu'
            ))
        
            if hp.Boolean('dropout'):
                model.add(layers.Dropout(0.2))
                
    else:
        for i in range(hp.Int('conv_layers', 1, 3)):
            model.add(layers.Conv2D(
                filters=hp.Int(f'filters_{i}', 16, 64, step=16),
                kernel_size=3, # 3x3 px
                activation='relu',
                padding='same'
            ))
            model.add(layers.MaxPooling2D(pool_size=2))
            
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))

    model.add(layers.Dense(10, activation='softmax'))
    
    hp_lr = hp.Choice('learning_rate', values=[1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def save_metrics(model, test_image, test_labels):
    
    loss, accuracy = model.evaluate(test_image, test_labels, verbose=0)    

    predictions = model.predict(test_image)
    predicted_label = np.argmax(predictions, axis=1)
    
    cm = confusion_matrix(test_labels, predicted_label)
    
    np.savetxt(CONFIG['CONF_MATRIX_FILE'], cm, fmt='%d') # fmt %d = int
    
    with open(CONFIG['METRICS_FILE'], 'w') as f:
        f.write(f"Final Test Loss: {loss:.4f}\n")
        f.write(f"Final Test Accuracy: {accuracy:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
        
    print(f"Metrics saved to {CONFIG['METRICS_FILE']}")
    print(f"Confusion Matrix saved to {CONFIG['CONF_MATRIX_FILE']}")
    return loss, accuracy
    
    pass

def main():
    train_images, train_labels, test_images, test_labels = load_data()

    #show_image_with_colormap(train_images[0])

    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=CONFIG['MAX_TRIALS'],
        executions_per_trial=CONFIG['EXEC_PER_TRIAL'],
        directory='clothes_tuner_dir'
    )

    tuner.search_space_summary()
    
    tuner.search(train_images, train_labels,
                 epochs=CONFIG['EPOCHS'], 
                 validation_split=0.2)

    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
        
    print(f"\nBest Hyperparameters found:")
    print(f"Model Type: {best_hyperparameters.get('model_type')}")
    print(f"Augmentation: {best_hyperparameters.get('use_augmentation')}")
    print(f"Learning Rate: {best_hyperparameters.get('learning_rate')}")
    
    print("\n--- Retraining Best Model from Scratch ---")
    model = tuner.hypermodel.build(best_hyperparameters)
    history = model.fit(train_images, train_labels, 
                        epochs=CONFIG['EPOCHS'], 
                        validation_split=0.2,
                        batch_size=CONFIG['BATCH_SIZE'])
    
    # 5. Zapis modelu i metryk
    model.save(CONFIG['MODEL_FILE'])
    print(f"Model saved to {CONFIG['MODEL_FILE']}")
    
    save_metrics(model, test_images, test_labels)


if __name__ == "__main__":
    main()