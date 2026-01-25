import argparse
import os
import sys
import numpy as np
import cv2
import keras
import tensorflow as tf

MODEL_PATH = 'best_clothes_model.keras'

#Fashion MNIST (Index -> Name)
CLASS_NAMES = [
    "T-shirt/top",   # 0
    "Trouser",       # 1
    "Pullover",      # 2
    "Dress",         # 3
    "Coat",          # 4
    "Sandal",        # 5
    "Shirt",         # 6
    "Sneaker",       # 7
    "Bag",           # 8
    "Ankle boot"     # 9
]

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error reading file: {image_path}")
        sys.exit(1)
    
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    
    # if background is in light colours then we have to invert it as fashion mnist is white clothes on black bg
    if np.mean(img_resized) > 127:
        img_processed = cv2.bitwise_not(img_resized)
    else:
        img_processed = img_resized

    # (0-255 -> 0.0-1.0)
    img_normalized = img_processed.astype('float32') / 255.0
    
    # (1- batch size, 28 - height, 28 - width, 1 - channel)
    img_ready = img_normalized.reshape(1, 28, 28, 1)
    
    return img_ready, img_resized, img_processed
def main():
    parser = argparse.ArgumentParser(description="Clothes classification based on the provided image")
    parser.add_argument('filename', type=str, help='Path to an image file')
    
    args = parser.parse_args()
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found '{MODEL_PATH}'.")
        print("Run fashion_classification.py to train the model.")
        sys.exit(1)
        
    try:
        print(f"Loading model from {MODEL_PATH}")
        model = keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error in loading model: {e}")
        sys.exit(1)

    print(f"Classifying image: {args.filename}")
    img_tensor, img_original_small, img_inverted = preprocess_image(args.filename)
    
    predictions = model.predict(img_tensor, verbose=0)
    
    class_idx = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    
    print("\n" + "="*40)
    print(f"Classification score:")
    print("="*40)
    print(f"Class:   {class_idx}")
    print(f"Name:   {CLASS_NAMES[class_idx]}")
    print(f"Confidence: {confidence:.2f}%")
    print("="*40 + "\n")
    
if __name__ == "__main__":
    main()