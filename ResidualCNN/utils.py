import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path

from classification import preprocess_image


def load_images(paths: list[str], max_images_number: int | None = None):
    # Generator that yields images loaded from the given paths
    if max_images_number is not None:
        paths = paths[:max_images_number]
    for path in paths:
        image = cv2.imread(str(path))
        if image is not None:
            yield image


def load_image_paths_and_labels(dataset_path: str, extension: str = 'jpg') -> tuple:
    base_folder = Path(dataset_path)
    if not base_folder.exists() or not base_folder.is_dir():
        raise FileNotFoundError(f'Base folder not found: {base_folder}')
    unique_labels = sorted([folder.name for folder in base_folder.iterdir() if folder.is_dir()])
    image_paths = []
    image_labels = []
    for label in unique_labels:
        for image_path in (base_folder / label).glob(f'*.{extension}'):
            image_paths.append(str(image_path))
            image_labels.append(label)
    return image_paths, image_labels


class SampleGenerator:

    def __init__(self, image_paths: list[str], labels: list[str], unique_labels: list[str],
                 augmented_number: int = 0, image_size: tuple[int, int] | None = (100, 100)):
        self.image_paths = image_paths
        self.labels = labels
        self.unique_labels = unique_labels
        self.augmented_number = augmented_number
        self.image_size = image_size

    def __call__(self):
        for image, label in zip(load_images(self.image_paths), self.labels):
            label_index = self.unique_labels.index(label)
            label_one_hot = keras.utils.to_categorical(label_index, num_classes=len(self.unique_labels))
            yield preprocess_image(image, size=self.image_size), label_one_hot
            for _ in range(self.augmented_number):
                augmented_image = augment_image(image)
                yield preprocess_image(augmented_image, size=self.image_size), label_one_hot


def augment_image(image: np.ndarray) -> np.ndarray:
    # Random horizontal flip
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
    # Random vertical flip
    if random.random() < 0.5:
        image = cv2.flip(image, 0)
    # Random rotation
    angle = random.uniform(-10, 10)
    height, width = image.shape[:2]
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    image = cv2.warpAffine(image, M, (width, height))
    return image


def show_training_history_over_epochs(history: keras.callbacks.History, suptitle: str) -> None:
    plt.figure()
    plt.suptitle(suptitle)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
