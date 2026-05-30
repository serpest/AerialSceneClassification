from utils import build_ucmlu_train_validation_test_datasets, finetune_model_on_ucmlu_all_strategies, train_model_on_ucmlu, show_training_history_over_epochs


def main(image_size: tuple[int, int] = (100, 100), train_augmented_number: int = 1) -> None:
    train_dataset, validation_dataset, test_dataset, unique_labels, image_shape, train_steps, validation_steps, test_steps = build_ucmlu_train_validation_test_datasets(
        image_size=image_size, train_augmented_number=train_augmented_number
    )
    finetune_histories = finetune_model_on_ucmlu_all_strategies(
        train_dataset, validation_dataset, train_steps, validation_steps, unique_labels, image_shape
    )
    train_history = train_model_on_ucmlu(
        train_dataset, validation_dataset, train_steps, validation_steps, unique_labels, image_shape
    )
    for finetune_type, history in finetune_histories.items():
        show_training_history_over_epochs(history, f'{finetune_type} Finetuning History')
    show_training_history_over_epochs(train_history, 'Training History')


if __name__ == '__main__':
    main()
