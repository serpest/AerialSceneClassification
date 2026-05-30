import math
import keras
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers

from classification import build_model
from config import UCMLU_DATASET_PATH, UCMLU_FULL_FINETUNED_MODEL_PATH, UCMLU_HEAD_FINETUNED_MODEL_PATH, UCMLU_PARTIAL_FINETUNED_MODEL_PATH, UCMLU_TRAINED_MODEL_PATH
from pretrain import load_aid_pretrained_model
from utils import SampleGenerator, load_image_paths_and_labels, show_training_history_over_epochs


def load_ucmlu_image_paths_and_labels(dataset_path: str = UCMLU_DATASET_PATH, train_size: float = 0.7,
                                      validation_size: float = 0.15) -> tuple:
    image_paths, image_labels =  load_image_paths_and_labels(f'{dataset_path}/Images', extension='tif')
    return split_ucmlu_paths_and_labels(image_paths, image_labels, train_size, validation_size)


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
        random_state=42,
        shuffle=True,
        stratify=temp_labels
    )
    return train_image_paths, train_labels, validation_image_paths, validation_labels, test_image_paths, test_labels


def build_ucmlu_train_validation_test_datasets(image_size: tuple[int, int] = (100, 100), train_size: float = 0.7,
                                               validation_size: float = 0.15, batch_size: int = 32,
                                               train_augmented_number: int = 2) -> tuple:
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


def build_finetuned_model(classes_number: int, train_strategy: str = 'full') -> keras.Model:
    if train_strategy not in ['head', 'partial', 'full']:
        raise ValueError(f'Invalid train strategy: {train_strategy}')
    pretrained_model = load_aid_pretrained_model()
    inputs = pretrained_model.inputs
    x = pretrained_model.layers[-3].output # Just before layers.Dense(units=classes_number)(x)
    x = layers.Dense(units=classes_number)(x)
    outputs = layers.Softmax()(x)
    model = keras.Model(inputs, outputs)
    if train_strategy == 'head':
        # Freeze convolutional layers and residual blocks, train dense layers
        for layer in model.layers:
            layer.trainable = False
            if isinstance(layer, layers.Flatten):
                break
    elif train_strategy == 'partial':
        # Freeze convolutional layer and most residual blocks, train remaining layers
        last_non_trainable_index = [index for index, layer in enumerate(model.layers) if isinstance(layer, layers.MaxPooling2D)][2]
        # The first trainable layer is residual_block(x, filters=128)
        for index, layer in enumerate(model.layers):
            layer.trainable = False
            if index == last_non_trainable_index:
                break
    elif train_strategy == 'full':
        # Train all layers of the model
        pass # All layers are trainable by default
    return model


def finetune_model_on_ucmlu(train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset,
                            train_steps: int, validation_steps: int, unique_labels: list[str],
                            epochs: int = 100, learning_rate: float = 1e-4,
                            train_strategy: str = 'full') -> tuple[keras.Model, keras.callbacks.History, float]:
    print(f'Starting finetuning with strategy {train_strategy} on UCMLU')
    classes_number = len(unique_labels)
    model = build_finetuned_model(classes_number=classes_number, train_strategy=train_strategy)
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
    weights_path = get_weights_path(train_strategy)
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
    validation_metrics = evaluate_model(model, validation_dataset, validation_steps)
    print(f'Validation F1-score (macro) at the end of finetuning with strategy {train_strategy}: {validation_metrics["f1_macro"]:.4f}')
    return model, history, validation_metrics['f1_macro']


def get_weights_path(train_strategy: str) -> str:
    if train_strategy == 'head':
        return UCMLU_HEAD_FINETUNED_MODEL_PATH
    elif train_strategy == 'partial':
        return UCMLU_PARTIAL_FINETUNED_MODEL_PATH
    elif train_strategy == 'full':
        return UCMLU_FULL_FINETUNED_MODEL_PATH
    else:
        raise ValueError(f'Invalid train strategy: {train_strategy}')


def finetune_model_on_ucmlu_all_strategies(train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset,
                                           train_steps: int, validation_steps: int, unique_labels: list[str],
                                           epochs: int = 100, learning_rate: float = 1e-4) -> dict[str, tuple[keras.Model, keras.callbacks.History, float]]:
    results = {}
    for train_strategy in ['head', 'partial', 'full']:
        model, history, f1_macro = finetune_model_on_ucmlu(
            train_dataset, validation_dataset, train_steps, validation_steps,
            unique_labels, epochs, learning_rate, train_strategy
        )
        results[train_strategy] = model, history, f1_macro
    return results


def load_finetuned_ucmlu_model(train_strategy: str = 'full', classes_number: int = 21) -> keras.Model:
    weights_path = get_weights_path(train_strategy)
    model = build_finetuned_model(classes_number=classes_number)
    model.load_weights(weights_path)
    return model


def train_model_on_ucmlu(train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset, train_steps: int,
                         validation_steps: int, unique_labels: list[str], image_shape: tuple,
                         epochs: int = 100, learning_rate: float = 1e-3) -> tuple[keras.Model, keras.callbacks.History, float]:
    print('Starting training on UCMLU')
    classes_number = len(unique_labels)
    model = build_model(input_shape=image_shape, classes_number=classes_number)
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
    validation_metrics = evaluate_model(model, validation_dataset, validation_steps)
    print(f'Validation F1-score (macro) at the end of training: {validation_metrics["f1_macro"]:.4f}')
    return model, history, validation_metrics['f1_macro']


def evaluate_model(model: keras.Model, dataset: tf.data.Dataset, steps: int) -> dict[str, float]:
    y_true = []
    for _, batch_labels in dataset.take(steps):
        # Convert one-hot encoded labels to label indexes
        y_true.append(np.argmax(batch_labels.numpy(), axis=1))
    y_true = np.array(y_true)
    y_pred_proba = model.predict(dataset, steps=steps, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    return metrics


def show_confusion_matrix(matrix: np.ndarray, unique_labels: list[str]) -> None:
    plt.figure()
    plt.imshow(matrix, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()


def main(image_size: tuple[int, int] = (100, 100), train_augmented_number: int = 1) -> None:
    # Load data and build datasets
    train_dataset, validation_dataset, test_dataset, unique_labels, image_shape, train_steps, validation_steps, test_steps = build_ucmlu_train_validation_test_datasets(
        image_size=image_size, train_augmented_number=train_augmented_number
    )
    # Train and finetune models
    finetune_results = finetune_model_on_ucmlu_all_strategies(
        train_dataset, validation_dataset, train_steps, validation_steps, unique_labels
    )
    train_results = train_model_on_ucmlu(
        train_dataset, validation_dataset, train_steps, validation_steps, unique_labels, image_shape
    )
    # Show loss and accuracy curves over epochs for all training and finetuning processes
    for finetune_type, (_, history, _) in finetune_results.items():
        show_training_history_over_epochs(history, f'{finetune_type.capitalize()} Finetuning History')
    show_training_history_over_epochs(train_results[1], 'Training History')
    # Identify the best model based on validation F1-score (macro) and evaluate it on the test set
    best_finetune_type = max(finetune_results, key=lambda key: finetune_results[key][2])
    best_finetune_f1_macro = finetune_results[best_finetune_type][2]
    train_f1_macro = train_results[2]
    print(f'Best finetuning strategy: {best_finetune_type} with validation F1-score (macro): {best_finetune_f1_macro:.4f}')
    print(f'Training validation F1-score (macro): {train_f1_macro:.4f}')
    if best_finetune_f1_macro > train_f1_macro:
        best_model = finetune_results[best_finetune_type][0]
        print(f'Best model is the one finetuned with strategy {best_finetune_type}')
    else:
        best_model = train_results[0]
        print('Best model is the one trained from scratch')
    test_metrics = evaluate_model(best_model, test_dataset, test_steps)
    print(f'Test Accuracy: {test_metrics["accuracy"]:.4f}')
    print(f'Test Precision (macro): {test_metrics["precision_macro"]:.4f}')
    print(f'Test Recall (macro): {test_metrics["recall_macro"]:.4f}')
    print(f'Test F1-score (macro): {test_metrics["f1_macro"]:.4f}')
    show_confusion_matrix(test_metrics['confusion_matrix'], unique_labels)


if __name__ == '__main__':
    main()
