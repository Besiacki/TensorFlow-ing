import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models

CONFIG = {
    'IMG_HEIGHT': 256,
    'IMG_WIDTH': 256,
    'BATCH_SIZE': 8,
    'LATENT_DIM': 4,
    'EPOCHS': 100,
    'DATA_DIR': 'images'
}

def prepare_dataset(directory: str) -> tf.data.Dataset:
    """Ładuje zdjęcia, normalizuje je i aplikuje augmentację."""
    
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Katalog '{directory}' nie istnieje!")

    # Ładowanie surowych danych
    dataset = keras.utils.image_dataset_from_directory(
        directory,
        label_mode=None, # Autoenkoder nie potrzebuje etykiet
        image_size=(CONFIG['IMG_HEIGHT'], CONFIG['IMG_WIDTH']),
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=True
    )
    
    data_augmentation = keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    def process_data(images):
        images = tf.cast(images, tf.float32) / 255.0 # Normalizacja pod sigmoid
        aug_images = data_augmentation(images) # 
        #return images, images
        return aug_images, aug_images

    # Optymalizacja potoku danych (mapowanie + prefetch)
    augmented_dataset = (
        dataset
        .map(process_data, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    
    return augmented_dataset

class Autoencoder(keras.Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Obliczamy kształt przed spłaszczeniem, żeby dekoder wiedział jak odtwarzać.
        # Dla 256x256 i 4 warstw ze stride=2:
        # 256 -> 128 -> 64 -> 32 -> 16.
        # Więc końcowy kształt to 16x16x256 (przy 256 filtrach)
        self.shape_before_flatten = (16, 16, 256)

        # --- ENKODER ---
        self.encoder = keras.Sequential([
            layers.InputLayer(shape=(CONFIG['IMG_HEIGHT'], CONFIG['IMG_WIDTH'], 3)),
            
            layers.Conv2D(32, 3, activation='relu', strides=2, padding='same'),
            layers.Conv2D(64, 3, activation='relu', strides=2, padding='same'),
            layers.Conv2D(128, 3, activation='relu', strides=2, padding='same'),
            layers.Conv2D(256, 3, activation='relu', strides=2, padding='same'),

            
            layers.Flatten(),
            layers.Dense(latent_dim, name="latent_space"),
        ], name="encoder")

        # --- DEKODER ---
        self.decoder = keras.Sequential([
            layers.InputLayer(shape=(latent_dim,)),
            
            layers.Dense(int(np.prod(self.shape_before_flatten)), activation='relu'),
            layers.Reshape(self.shape_before_flatten),
            
            layers.Conv2DTranspose(256, 3, activation='relu', strides=2, padding='same'),
            layers.Conv2DTranspose(128, 3, activation='relu', strides=2, padding='same'),
            layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same'),
            layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same'),
            
            layers.Conv2D(3, 3, activation='sigmoid', padding='same')
        ], name="decoder")

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def plot_results(model, dataset, n=4):
    """Rysuje oryginały i rekonstrukcje dla jednego batcha."""
    # Pobieramy jeden batch danych
    for images, _ in dataset.take(1):
        reconstructed = model.predict(images)
        
        plt.figure(figsize=(10, 5))
        for i in range(min(n, len(images))):
            # Oryginał
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(images[i])
            plt.title("Original")
            plt.axis("off")
            
            # Rekonstrukcja
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(reconstructed[i])
            plt.title("Reconstructed")
            plt.axis("off")
        plt.tight_layout()
        plt.savefig('reconstruction_results.png')
        plt.show()
        break

if __name__ == "__main__":
    try:
        print(f"Ładowanie danych z katalogu: {CONFIG['DATA_DIR']}...")
        dataset = prepare_dataset(CONFIG['DATA_DIR'])
        
        print("Inicjalizacja modelu...")
        autoencoder = Autoencoder(CONFIG['LATENT_DIM'])
        
        # Kompilacja modelu
        autoencoder.compile(optimizer='adam', loss=keras.losses.MeanSquaredError)
        
        # Budowanie modelu, aby wyświetlić podsumowanie
        autoencoder.build(input_shape=(None, CONFIG['IMG_HEIGHT'], CONFIG['IMG_WIDTH'], 3))
        autoencoder.summary()
        
        print("Rozpoczynanie treningu...")
        # Trenujemy model. Augmentacja jest zaszyta w dataset.
        history = autoencoder.fit(
            dataset,
            epochs=CONFIG['EPOCHS']
        )
        
        print("Zapisywanie wag modelu...")
        autoencoder.save_weights('autoencoder_weights.weights.h5')
        
        print("Generowanie wizualizacji wyników...")
        plot_results(autoencoder, dataset)
        
    except Exception as e:
        print(f"Wystąpił błąd: {e}")