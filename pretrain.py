import math
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split

from classification import build_model
from config import AID_DATASET_PATH, AID_PRETRAINED_MODEL_PATH
from utils import SampleGenerator, load_image_paths_and_labels, show_training_history_over_epochs


def load_aid_image_paths_and_labels(dataset_path: str = AID_DATASET_PATH, train_size: float = 0.7) -> tuple:
    image_paths, image_labels = load_image_paths_and_labels(dataset_path, extension='jpg')
    return split_aid_paths_and_labels(image_paths, image_labels, train_size)


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


def build_aid_train_validation_datasets(image_size: tuple[int, int] = (100, 100), train_size: float = 0.7,
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


def pretrain_model_on_aid(image_size: tuple[int, int] = (100, 100), epochs: int = 100, train_augmented_number: int = 1,
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
        patience=10,
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


def load_aid_pretrained_model(input_shape: tuple[int, int, int] = (100, 100, 3), classes_number: int = 30,
                              weights_path: str = AID_PRETRAINED_MODEL_PATH) -> keras.Model:
    model = build_model(input_shape, classes_number)
    model.load_weights(weights_path)
    return model


def main(image_size: tuple[int, int] = (100, 100), epochs: int = 100, train_augmented_number: int = 1,
         show_history: bool = True) -> None:
    history = pretrain_model_on_aid(image_size, epochs, train_augmented_number)
    if show_history:
        show_training_history_over_epochs(history, 'Pretraining History')


if __name__ == '__main__':
    main()
