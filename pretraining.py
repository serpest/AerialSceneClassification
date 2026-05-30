from utils import pretrain_model_on_aid, show_training_history_over_epochs


def main() -> None:
    history = pretrain_model_on_aid()
    show_training_history_over_epochs(history, 'Pretraining History')


if __name__ == '__main__':
    main()
