import cv2
import keras
import numpy as np
import tensorflow as tf
from keras import layers

from config import UCMLU_FULL_FINETUNED_MODEL_PATH, UCMLU_HEAD_FINETUNED_MODEL_PATH, UCMLU_PARTIAL_FINETUNED_MODEL_PATH, UCMLU_TRAINED_MODEL_PATH


UCML_UNIQUE_LABELS = [
    'agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral', 'denseresidential',
    'forest', 'freeway', 'golfcourse', 'harbor', 'intersection', 'mediumresidential', 'mobilehomepark',
    'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential', 'storagetanks', 'tenniscourt'
]


def preprocess_image(image: np.ndarray, size: tuple[int, int] | None = (100, 100)) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406]) # ImageNet mean values for normalization
    std = np.array([0.229, 0.224, 0.225]) # ImageNet standard deviation values for normalization
    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if size is not None:
        processed_image = cv2.resize(processed_image, size)
    processed_image = processed_image.astype(np.float32) / 255.0
    processed_image = (processed_image - mean) / std
    return processed_image


def residual_block(x: tf.Tensor, filters: int, kernel_size: int = 3) -> layers.Layer:
    shortcut = x
    # First convolutional layer
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # Second convolutional layer
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    # Skip connection
    if shortcut.shape[-1] != filters:
        # Adjust the number of channels in the shortcut to match the output of the convolutional layers
        # (H x W x C) -> (H x W x filters)
        shortcut = layers.Conv2D(filters, kernel_size=1, padding='same')(shortcut) # 1x1 convolution
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


def build_model(input_shape: tuple[int, int, int], classes_number: int, print_summary: bool = False) -> keras.Model:
    inputs = keras.Input(shape=input_shape) # (height, width, channels)
    x = layers.Conv2D(filters=32, kernel_size=3, padding='same')(inputs) # (height, width, 32)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x) # (height/2, width/2, 32)
    x = residual_block(x, filters=32) # (height/2, width/2, 32)
    x = residual_block(x, filters=32) # (height/2, width/2, 32)
    x = layers.MaxPooling2D(pool_size=2)(x) # (height/4, width/4, 32)
    x = residual_block(x, filters=64) # (height/4, width/4, 64), 1x1 convolution in the skip connection
    x = residual_block(x, filters=64) # (height/4, width/4, 64)
    x = layers.MaxPooling2D(pool_size=2)(x) # (height/8, width/8, 64)
    x = residual_block(x, filters=128) # (height/8, width/8, 128), 1x1 convolution in the skip connection
    x = residual_block(x, filters=128) # (height/8, width/8, 128)
    x = layers.MaxPooling2D(pool_size=2)(x) # (height/16, width/16, 128)
    x = layers.Flatten()(x) # (height/16 * width/16 * 128,)
    x = layers.Dense(units=64)(x) # (64,)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(rate=0.3)(x)
    x = layers.Dense(units=64)(x) # (64,)
    x = layers.ReLU()(x)
    x = layers.Dropout(rate=0.3)(x)
    x = layers.Dense(units=classes_number)(x) # (classes_number,)
    outputs = layers.Softmax()(x)
    model = keras.Model(inputs, outputs)
    if print_summary:
        model.summary()
    return model


def load_model(weights_path: str, input_shape: tuple[int, int, int] = (100, 100, 3),
               classes_number: int = 21) -> keras.Model:
    model = build_model(input_shape, classes_number)
    model.load_weights(weights_path)
    return model


def classify_image(model: keras.Model, image: np.ndarray, unique_labels: list[str], print_probabilities: bool = False) -> str:
    processed_image = preprocess_image(image)
    input_tensor = np.expand_dims(processed_image, axis=0) # (1, height, width, channels)
    predictions = model.predict(input_tensor) # (1, classes_number), probabilities for each class
    if print_probabilities:
        print(f'Predicted probabilities: {sorted(zip(unique_labels, predictions[0]), key=lambda x: x[1], reverse=True)}')
    predicted_index = np.argmax(predictions[0])
    predicted_label = unique_labels[predicted_index]
    return predicted_label


def classify_image_on_ucmlu(image_path: str, model_name: str = 'full_finetuned') -> str:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f'Image not read: {image_path}')
    weights_path = get_weights_path_from_model_name(model_name)
    model = load_model(weights_path=weights_path)
    predicted_label = classify_image(model, image, UCML_UNIQUE_LABELS)
    return predicted_label


def get_weights_path_from_model_name(model_name: str) -> str:
    if model_name == 'head_finetuned':
        return UCMLU_HEAD_FINETUNED_MODEL_PATH
    elif model_name == 'partial_finetuned':
        return UCMLU_PARTIAL_FINETUNED_MODEL_PATH
    elif model_name == 'full_finetuned':
        return UCMLU_FULL_FINETUNED_MODEL_PATH
    elif model_name == 'trained':
        return UCMLU_TRAINED_MODEL_PATH
    raise ValueError(f'Invalid model name: {model_name}')
