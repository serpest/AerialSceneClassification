import cv2
import math
import random
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from config import AID_DATASET_PATH, AID_PRETRAINED_MODEL_PATH, UCMLU_DATASET_PATH, UCMLU_HEAD_FINETUNED_MODEL_PATH, UCMLU_PARTIAL_FINETUNED_MODEL_PATH, UCMLU_FULL_FINETUNED_MODEL_PATH, UCMLU_TRAINED_MODEL_PATH


def load_images(paths: list[str], shuffle: bool = False, max_images_number: int | None = None):
    # Generator that yields images loaded from the given paths
    if shuffle:
        random_generator = random.Random(42)
        random_generator.shuffle(paths)
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


def load_aid_image_paths_and_labels(dataset_path: str = AID_DATASET_PATH, train_size: float = 0.7) -> tuple:
    image_paths, image_labels = load_image_paths_and_labels(dataset_path, extension='jpg')
    return split_aid_paths_and_labels(image_paths, image_labels, train_size)


def load_ucmlu_image_paths_and_labels(dataset_path: str = UCMLU_DATASET_PATH, train_size: float = 0.7,
                                      validation_size: float = 0.15) -> tuple:
    image_paths, image_labels =  load_image_paths_and_labels(f'{dataset_path}/Images', extension='tif')
    return split_ucmlu_paths_and_labels(image_paths, image_labels, train_size, validation_size)


def split_aid_paths_and_labels(image_paths: list[str], image_labels: list[str], train_size: float) -> tuple:
    train_image_paths, validation_image_paths, train_labels, validation_labels = train_test_split(
        image_paths,
        image_labels,
        train_size=train_size,
        random_state=42,
        shuffle=True,
        stratify=image_labels
    )
    return train_image_paths, validation_image_paths, train_labels, validation_labels


def split_ucmlu_paths_and_labels(image_paths: list[str], image_labels: list[str],
                                 train_size: float, validation_size: float) -> tuple:
    train_image_paths, temp_image_paths, train_labels, temp_labels = train_test_split(
        image_paths,
        image_labels,
        train_size=train_size,
        random_state=42,
        shuffle=True,
        stratify=image_labels
    )
    validation_image_paths, test_image_paths, validation_labels, test_labels = train_test_split(
        temp_image_paths,
        temp_labels,
        train_size=validation_size,
        random_state=42+1,
        shuffle=True,
        stratify=temp_labels
    )
    return train_image_paths, train_labels, validation_image_paths, validation_labels, test_image_paths, test_labels


class SampleGenerator:

    def __init__(self, image_paths: list[str], labels: list[str], unique_labels: list[str],
                 augmented_number: int = 0, image_size: tuple[int, int] | None = (200, 200)):
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


def preprocess_image(image: np.ndarray, size: tuple[int, int] | None = (200, 200)) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406]) # ImageNet mean values for normalization
    std = np.array([0.229, 0.224, 0.225]) # ImageNet standard deviation values for normalization
    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if size is not None:
        processed_image = cv2.resize(processed_image, size)
    processed_image = processed_image.astype(np.float32) / 255.0
    processed_image = (processed_image - mean) / std
    return processed_image


def build_aid_train_validation_datasets(image_size: tuple[int, int] = (200, 200), train_size: float = 0.7,
                                        batch_size: int = 32, train_augmented_number: int = 1) -> tuple:
    train_image_paths, validation_image_paths, train_labels, validation_labels = load_aid_image_paths_and_labels(train_size=train_size)
    unique_labels = sorted(set(train_labels) | set(validation_labels))
    image_shape = (image_size[0], image_size[1], 3)
    dataset_output_signature = (
        tf.TensorSpec(shape=image_shape, dtype=tf.float32), # Shape of the preprocessed image
        tf.TensorSpec(shape=(len(unique_labels),), dtype=tf.float32) # Shape of the one-hot encoded label vector
    )
    train_dataset = tf.data.Dataset.from_generator(
        SampleGenerator(train_image_paths, train_labels, unique_labels, augmented_number=train_augmented_number, image_size=image_size),
        output_signature=dataset_output_signature
    )
    validation_dataset = tf.data.Dataset.from_generator(
        SampleGenerator(validation_image_paths, validation_labels, unique_labels, augmented_number=0, image_size=image_size),
        output_signature=dataset_output_signature
    )
    train_dataset = train_dataset.batch(batch_size).repeat() # Repeat the training dataset indefinitely for multiple epochs
    validation_dataset = validation_dataset.batch(batch_size)
    train_samples = len(train_image_paths) * (1 + train_augmented_number)
    validation_samples = len(validation_image_paths)
    train_steps = math.ceil(train_samples / batch_size) # Number of steps per epoch for training
    validation_steps = math.ceil(validation_samples / batch_size)
    return train_dataset, validation_dataset, unique_labels, image_shape, train_steps, validation_steps


def build_ucmlu_train_validation_test_datasets(image_size: tuple[int, int] = (100, 100), train_size: float = 0.7,
                                               validation_size: float = 0.15, batch_size: int = 32,
                                               train_augmented_number: int = 1) -> tuple:
    train_image_paths, train_labels, validation_image_paths, validation_labels, test_image_paths, test_labels = load_ucmlu_image_paths_and_labels(
        train_size=train_size, validation_size=validation_size
    )
    unique_labels = sorted(set(train_labels) | set(validation_labels) | set(test_labels))
    image_shape = (image_size[0], image_size[1], 3)
    dataset_output_signature = (
        tf.TensorSpec(shape=image_shape, dtype=tf.float32), # Shape of the preprocessed image
        tf.TensorSpec(shape=(len(unique_labels),), dtype=tf.float32) # Shape of the one-hot encoded label vector
    )
    train_dataset = tf.data.Dataset.from_generator(
        SampleGenerator(train_image_paths, train_labels, unique_labels, augmented_number=train_augmented_number, image_size=image_size),
        output_signature=dataset_output_signature
    )
    validation_dataset = tf.data.Dataset.from_generator(
        SampleGenerator(validation_image_paths, validation_labels, unique_labels, augmented_number=0, image_size=image_size),
        output_signature=dataset_output_signature
    )
    test_dataset = tf.data.Dataset.from_generator(
        SampleGenerator(test_image_paths, test_labels, unique_labels, augmented_number=0, image_size=image_size),
        output_signature=dataset_output_signature
    )
    train_dataset = train_dataset.batch(batch_size).repeat() # Repeat the training dataset indefinitely for multiple epochs
    validation_dataset = validation_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)
    train_samples = len(train_image_paths) * (1 + train_augmented_number)
    validation_samples = len(validation_image_paths)
    test_samples = len(test_image_paths)
    train_steps = math.ceil(train_samples / batch_size)
    validation_steps = math.ceil(validation_samples / batch_size)
    test_steps = math.ceil(test_samples / batch_size)
    return (
        train_dataset, validation_dataset, test_dataset, unique_labels,
        image_shape, train_steps, validation_steps, test_steps
    )


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


def augment_image(image: np.ndarray) -> np.ndarray:
    # Random horizontal flip
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
    # Random rotation
    angle = random.uniform(-10, 10)
    height, width = image.shape[:2]
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    image = cv2.warpAffine(image, M, (width, height))
    return image


def build_data_augmentation_pipeline() -> keras.Sequential:
    data_augmentation = keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.1)
    ])
    return data_augmentation


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
    x = layers.Flatten()(x) # (height/8 * width/8 * 128,)
    x = layers.Dense(units=128)(x) # (128,)
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


def pretrain_model_on_aid(image_size: tuple[int, int] = (100, 100), epochs: int = 30, train_augmented_number: int = 1,
                          learning_rate: float = 1e-3, weights_path: str = AID_PRETRAINED_MODEL_PATH) -> keras.callbacks.History:
    train_dataset, validation_dataset, unique_labels, image_shape, train_steps, validation_steps = build_aid_train_validation_datasets(
        image_size=image_size,
        train_augmented_number=train_augmented_number
    )
    classes_number = len(unique_labels)
    model = build_model(image_shape, classes_number, print_summary=True)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    model_checkpoint = keras.callbacks.ModelCheckpoint( # Save the best model weights during training
        filepath=weights_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True
    )
    callbacks = [early_stopping, model_checkpoint]
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        steps_per_epoch=train_steps,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=callbacks
    )
    return history


def load_aid_pretrained_model(weights_path: str = AID_PRETRAINED_MODEL_PATH, input_shape: tuple[int, int, int] = (100, 100, 3),
                              classes_number: int = 30) -> keras.Model:
    model = build_model(input_shape, classes_number)
    model.load_weights(weights_path)
    return model


def build_finetune_model(classes_number: int, train_strategy: str = 'full') -> keras.Model:
    pretrained_model = load_aid_pretrained_model()
    inputs = pretrained_model.inputs
    x = pretrained_model.layers[-3].output
    x = layers.Dense(units=classes_number)(x)
    outputs = layers.Softmax()(x)
    model = keras.Model(inputs, outputs)
    if train_strategy == 'head': # Freeze convolutional layers and train only dense layers
        for layer in model.layers:
            if isinstance(layer, layers.Flatten):
                break
            layer.trainable = False
    elif train_strategy == 'partial': # Freeze first convolutional layers and train the rest of the model
        last_maxpooling_index = max(
            index for index, layer in enumerate(model.layers) if isinstance(layer, layers.MaxPooling2D)
        )
        for index, layer in enumerate(model.layers):
            layer.trainable = False
            if index == last_maxpooling_index:
                break
    elif train_strategy == 'full': # Train all layers of the model
        pass # All layers are trainable by default
    else:
        raise ValueError(f'Invalid train strategy: {train_strategy}')
    return model


def finetune_model_on_ucmlu(train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset,
                            train_steps: int, validation_steps: int, unique_labels: list[str],
                            epochs: int = 30, learning_rate: float = 1e-4,
                            train_strategy: str = 'full') -> keras.callbacks.History:
    classes_number = len(unique_labels)
    model = build_finetune_model(classes_number=classes_number, train_strategy=train_strategy)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    weights_path = UCMLU_FULL_FINETUNED_MODEL_PATH
    if train_strategy == 'head':
        weights_path = UCMLU_HEAD_FINETUNED_MODEL_PATH
    elif train_strategy == 'partial':
        weights_path = UCMLU_PARTIAL_FINETUNED_MODEL_PATH
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=weights_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True
    )
    callbacks = [early_stopping, model_checkpoint]
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        steps_per_epoch=train_steps,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=callbacks
    )
    return history


def finetune_model_on_ucmlu_all_strategies(train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset,
                                           train_steps: int, validation_steps: int, unique_labels: list[str],
                                           epochs: int = 30, learning_rate: float = 1e-4) -> dict[str, keras.callbacks.History]:
    histories = {}
    for train_strategy in ['head', 'partial', 'full']:
        print(f'Starting finetuning on UCMLU with strategy {train_strategy}')
        history = finetune_model_on_ucmlu(
            train_dataset, validation_dataset, train_steps, validation_steps, unique_labels,
            epochs=epochs, learning_rate=learning_rate, train_strategy=train_strategy
        )
        print(f'Finished finetuning on UCMLU with strategy {train_strategy}')
        histories[train_strategy] = history
    return histories


def load_finetuned_ucmlu_model(train_strategy: str = 'full', classes_number: int = 21) -> keras.Model:
    if train_strategy not in ['head', 'partial', 'full']:
        raise ValueError(f'Invalid train strategy: {train_strategy}')
    weights_path = UCMLU_FULL_FINETUNED_MODEL_PATH
    if train_strategy == 'head':
        weights_path = UCMLU_HEAD_FINETUNED_MODEL_PATH
    elif train_strategy == 'partial':
        weights_path = UCMLU_PARTIAL_FINETUNED_MODEL_PATH
    model = build_finetune_model(classes_number=classes_number)
    model.load_weights(weights_path)
    return model


def train_model_on_ucmlu(train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset, train_steps: int,
                         validation_steps: int, unique_labels: list[str], image_shape: tuple,
                         epochs: int = 100, learning_rate: float = 1e-3) -> keras.callbacks.History:
    classes_number = len(unique_labels)
    model = build_model(input_shape=image_shape, classes_number=classes_number)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=UCMLU_TRAINED_MODEL_PATH,
        monitor='val_loss',
        save_weights_only=True
    )
    callbacks = [early_stopping, model_checkpoint]
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        steps_per_epoch=train_steps,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=callbacks
    )
    return history


def load_trained_ucmlu_model(classes_number: int = 21) -> keras.Model:
    model = build_model(input_shape=(100, 100, 3), classes_number=classes_number)
    model.load_weights(UCMLU_TRAINED_MODEL_PATH)
    return model


def show_training_history_over_epochs(history: keras.callbacks.History, suptitle: str) -> None:
    plt.figure()
    plt.suptitle(suptitle)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()


def evaluate_model(model: keras.Model, test_dataset: tf.data.Dataset, test_steps: int) -> dict[str, float]:
    y_true = []
    for _, batch_labels in test_dataset.take(test_steps):
        # Convert one-hot encoded labels to label indexes
        y_true.append(np.argmax(batch_labels.numpy(), axis=1))
    y_true = np.array(y_true)
    y_pred_proba = model.predict(test_dataset, steps=test_steps, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }


def show_confusion_matrix(model: keras.Model, test_dataset: tf.data.Dataset, test_steps: int, unique_labels: list[str]) -> None:
    y_true = []
    for _, batch_labels in test_dataset.take(test_steps):
        # Convert one-hot encoded labels to label indexes
        y_true.append(np.argmax(batch_labels.numpy(), axis=1))
    y_true = np.array(y_true)
    y_pred_proba = model.predict(test_dataset, steps=test_steps, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    plt.figure()
    plt.imshow(confusion_matrix(y_true, y_pred), cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(unique_labels))
    plt.xticks(tick_marks, unique_labels, rotation=45)
    plt.yticks(tick_marks, unique_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()
