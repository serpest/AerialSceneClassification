import os
from pathlib import Path
import warnings
import pandas as pd
import cv2
import joblib
from matplotlib import pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from bow_classification import DescriptorsExtractor, VisualWordsHistogramComputer
from build_vocabularies import VISUAL_WORDS_CONFIGS, load_visual_words


UCMLU_DATASET_PATH = 'DATASET/UCMerced_LandUse'

CLASSIFIERS_PATH = 'classifiers' # Directory for storing trained classifiers

STRATIFIED_K_FOLD = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

CLASSIFIERS = {
    'SVM_Linear': SVC(kernel='linear'),
    'SVM_RBF': SVC(kernel='rbf'),
    'Random_Forest': RandomForestClassifier(random_state=42),
    'k-NN': KNeighborsClassifier(),
    'Logistic_Regression': LogisticRegression()
}

HISTOGRAM_NORMALIZATIONS = [None, 'L1', 'L2']


def load_ucmlu_images(dataset_path: str = UCMLU_DATASET_PATH):
    base_folder = Path(dataset_path) / 'Images'
    if not base_folder.exists() or not base_folder.is_dir():
        raise FileNotFoundError(f'Base folder not found: {base_folder}')
    for image_path in base_folder.glob('*/*.tif'):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        label = image_path.parent.name
        yield (label, image)


def convert_classifier_config_to_file_path(classifier_name: str, bow_method: str,
                                           bow_normalization: str, bow_clusters: int,
                                           hi_normalization: str) -> str:
    file_name = f'{classifier_name}_{bow_method}_{bow_normalization}_{bow_clusters}_{hi_normalization}.pkl'
    return f'{CLASSIFIERS_PATH}/{file_name}'


def save_classifier(classifier: object, classifier_name: str, bow_method: str, bow_normalization: str,
                    bow_clusters: int, hi_normalization: str) -> None:
    file_path = convert_classifier_config_to_file_path(classifier_name, bow_method, bow_normalization, bow_clusters, hi_normalization)
    joblib.dump(classifier, file_path)


def load_classifier(classifier_name: str, bow_method: str, bow_normalization: str,
                    bow_clusters: int, hi_normalization: str) -> object:
    file_path = convert_classifier_config_to_file_path(classifier_name, bow_method, bow_normalization, bow_clusters, hi_normalization)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Classifier file not found: {file_path}. Run build_classifiers.py to build the classifiers first')
    return joblib.load(file_path)


def evaluate_classifier_fold(classifier: object, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = classifier.predict(X_test)
    evaluation = {
        'accuracy': classifier.score(X_test, y_test),
        'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    return evaluation


def print_classifier_evaluations(classifier_name: str, evaluations: list) -> None:
    mean_accuracy = np.mean([e['accuracy'] for e in evaluations])
    std_accuracy = np.std([e['accuracy'] for e in evaluations])
    mean_precision = np.mean([e['precision'] for e in evaluations])
    std_precision = np.std([e['precision'] for e in evaluations])
    mean_recall = np.mean([e['recall'] for e in evaluations])
    std_recall = np.std([e['recall'] for e in evaluations])
    mean_f1_score = np.mean([e['f1_score'] for e in evaluations])
    std_f1_score = np.std([e['f1_score'] for e in evaluations])
    print('----------------------------------------')
    print(f'Classifier: {classifier_name}')
    print(f'Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}')
    print(f'Precision: {mean_precision:.4f} ± {std_precision:.4f}')
    print(f'Recall: {mean_recall:.4f} ± {std_recall:.4f}')
    print(f'F1-score: {mean_f1_score:.4f} ± {std_f1_score:.4f}')
    print('----------------------------------------')


def build_classifiers(bow_method: str, bow_normalization: str, bow_clusters: int, hi_normalization: str,
                      save_classifiers: bool = True, show_evaluation: bool = True, show_confusion_matrices: bool = True) -> pd.DataFrame:
    if save_classifiers:
        Path(CLASSIFIERS_PATH).mkdir(parents=True, exist_ok=True)
    descriptors_extractor = DescriptorsExtractor(method=bow_method, normalization=bow_normalization)
    visual_words = load_visual_words(
        method=bow_method, normalization=bow_normalization, clusters_number=bow_clusters
    )
    histogram_computer = VisualWordsHistogramComputer(
        descriptors_extractor=descriptors_extractor, visual_words=visual_words, normalization=hi_normalization
    )
    X = [] # Feature matrix
    y = [] # Labels
    for label, image in load_ucmlu_images():
        histogram = histogram_computer.compute_histogram(image=image)
        if histogram is None: # No descriptors found in the image
            continue
        X.append(histogram)
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    classifier_performances = pd.DataFrame()
    for classifier_name, classifier in CLASSIFIERS.items():
        evaluations = []
        for train_index, test_index in STRATIFIED_K_FOLD.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            fold_classifier = clone(classifier)
            fold_classifier.fit(X_train, y_train)
            evaluations.append(evaluate_classifier_fold(fold_classifier, X_test, y_test))
        classifier_performances = pd.concat([classifier_performances, pd.DataFrame([{
            'Classifier': classifier_name,
            'BoW Method': bow_method,
            'BoW Normalization': bow_normalization if bow_normalization is not None else 'None',
            'BoW Clusters': bow_clusters,
            'Histogram Normalization': hi_normalization if hi_normalization is not None else 'None',
            'Accuracy': np.mean([e['accuracy'] for e in evaluations]),
            'Precision': np.mean([e['precision'] for e in evaluations]),
            'Recall': np.mean([e['recall'] for e in evaluations]),
            'F1-score': np.mean([e['f1_score'] for e in evaluations])
        }])], ignore_index=True)
        if save_classifiers:
            final_classifier = clone(classifier)
            final_classifier.fit(X, y)
            save_classifier(final_classifier, classifier_name, bow_method, bow_normalization, bow_clusters, hi_normalization)
        if show_evaluation:
            print_classifier_evaluations(classifier_name, evaluations)
        if show_confusion_matrices:
            classifier_index = list(CLASSIFIERS.keys()).index(classifier_name)
            subfig = plt.subplot(2, 3, classifier_index + 1)
            subfig.imshow(np.mean([e['confusion_matrix'] for e in evaluations], axis=0))
            subfig.set_title(classifier_name)
            subfig.set_xlabel('Predicted label')
            subfig.set_ylabel('True label')
    if show_confusion_matrices:
        plt.show()
    return classifier_performances


def build_classifiers_wrapped(bow_method: str, bow_normalization: str, bow_clusters: int, hi_normalization: str,
                              save_classifiers: bool = True, show_evaluation: bool = True,
                              show_confusion_matrices: bool = True) -> pd.DataFrame:
    # Function used for parallel computation
    print(f'Building classifiers for bow_method={bow_method}, bow_normalization={bow_normalization}.'
          f'bow_clusters={bow_clusters}, hi_normalization={hi_normalization}...')
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    classifier_performances =  build_classifiers(
        bow_method=bow_method,
        bow_normalization=bow_normalization,
        bow_clusters=bow_clusters,
        hi_normalization=hi_normalization,
        save_classifiers=save_classifiers,
        show_evaluation=show_evaluation,
        show_confusion_matrices=show_confusion_matrices
    )
    return classifier_performances


def main() -> None:
    classifier_performances = Parallel(n_jobs=-1, prefer='processes')(
        delayed(build_classifiers_wrapped)(
            bow_method=bow_method,
            bow_normalization=bow_normalization,
            bow_clusters=bow_clusters,
            hi_normalization=hi_normalization,
            save_classifiers=True,
            show_evaluation=True,
            show_confusion_matrices=True
        )
        for hi_normalization in HISTOGRAM_NORMALIZATIONS
        for bow_method, bow_normalization, bow_clusters in VISUAL_WORDS_CONFIGS
    )
    classifier_performances = pd.concat(classifier_performances)
    classifier_performances = classifier_performances.sort_values(by='Accuracy', ascending=False)
    classifier_performances.to_csv(f'{CLASSIFIERS_PATH}/classifier_performances.csv', index=False)


if __name__ == '__main__':
    main()
