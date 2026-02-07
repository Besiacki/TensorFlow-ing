import tensorflow as tf
import argparse as arg
import numpy as np


def load_model(model_filename="first_mnist_model.keras"):
    """Load a pre-trained MNIST model."""
    try:
        return tf.keras.models.load_model(model_filename)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def load_image(image_filename="3.png"):
    """Load and preprocess a single image for MNIST digit recognition."""
    try:
        image = tf.keras.utils.load_img(
            image_filename,
            color_mode="grayscale",
            target_size=(28, 28),
            interpolation="nearest",
        )

        input_arr = tf.keras.utils.img_to_array(image)
        input_arr = input_arr[:, :, 0]
        input_arr = input_arr / 255.0
        input_arr = 1.0 - input_arr
        input_arr = np.expand_dims(input_arr, axis=0)

        return input_arr
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def image_recognition(model_path, image_path):
    model = load_model(model_path)
    image = load_image(image_path)
    print(image_path)

    predictions = model.predict(image)

    predicted_digit = np.argmax(predictions)

    confidence = predictions[0][predicted_digit]

    print(f"Predicted digit: {predicted_digit}")
    print(f"Confidence in prediction: {confidence}")
    return 0


if __name__ == "__main__":

    parser = arg.ArgumentParser("Digit recognization tool")
    parser.add_argument(
        "--model", type=str, required=True, help="Filepath to previously trained MNIST model."
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Filepath to image containing handwritten digit."
    )
    args = parser.parse_args()

    image_recognition(args.model, args.image)
