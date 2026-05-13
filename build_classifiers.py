from pathlib import Path

import cv2
import joblib
from matplotlib import pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from bow_classification import DescriptorsExtractor, DescriptorsExtractor, VisualWordsHistogramComputer
from build_vocabularies import load_saved_visual_words


CLASSIFIERS_PATH = 'classifiers'

UCMLU_DATASET_PATH = 'DATASET/UCMerced_LandUse'

STRATIFIED_K_FOLD = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

CLASSIFIERS = {
    'SVM_linear': SVC(kernel='linear'),
    'SVM_RBF': SVC(kernel='rbf'),
    'Random_Forest': RandomForestClassifier(random_state=42),
    'k-NN': KNeighborsClassifier(),
    'Logistic_Regression': LogisticRegression()
}


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


def convert_config_to_file_path(classifier_name: str, bow_method: str, bow_normalization: str, bow_clusters: int, hi_normalization: str) -> str:
    file_name = f'{classifier_name}_{bow_method}_{bow_normalization}_{bow_clusters}_{hi_normalization}.pkl'
    return f'{CLASSIFIERS_PATH}/{file_name}'


def load_classifier(classifier_name: str, bow_method: str, bow_normalization: str, bow_clusters: int, hi_normalization: str):
    file_path = convert_config_to_file_path(classifier_name, bow_method, bow_normalization, bow_clusters, hi_normalization)
    if not Path(file_path).exists():
        raise FileNotFoundError(f'Classifier not found: {file_path}. Please run build_classifiers.py to build the classifiers.')
    return joblib.load(file_path)


def evaluate_classifier(classifier, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = classifier.predict(X_test)
    evaluation = {
        'accuracy': classifier.score(X_test, y_test),
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro'),
        'f1_score': f1_score(y_test, y_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    return evaluation

def compute_mean_evaluation(evaluations: list) -> dict:
    mean_evaluation = {
        'accuracy': np.mean([e['accuracy'] for e in evaluations]),
        'precision': np.mean([e['precision'] for e in evaluations]),
        'recall': np.mean([e['recall'] for e in evaluations]),
        'f1_score': np.mean([e['f1_score'] for e in evaluations]),
        'confusion_matrix': np.mean([e['confusion_matrix'] for e in evaluations], axis=0)
    }
    return mean_evaluation

def print_mean_evaluation(classifier_name: str, evaluations: list) -> None:
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


def build_classifiers_for_config(bow_method: str = 'SIFT', bow_normalization: str = 'L2', bow_clusters: int = 100, hi_normalization: str = 'L2',
                                 show_evaluation: bool = True, show_confusion_matrices: bool = True) -> None:
    Path(CLASSIFIERS_PATH).mkdir(parents=True, exist_ok=True)
    descriptors_extractor = DescriptorsExtractor(method=bow_method, normalization=bow_normalization)
    visual_words = load_saved_visual_words(
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
    for classifier_name, classifier in CLASSIFIERS.items():
        evaluations = []
        for train_index, test_index in STRATIFIED_K_FOLD.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            classifier.fit(X_train, y_train)
            evaluations.append(evaluate_classifier(classifier, X_test, y_test))
        joblib.dump(
            classifier,
            convert_config_to_file_path(classifier_name, bow_method, bow_normalization, bow_clusters, hi_normalization)
        )
        if show_evaluation:
            print_mean_evaluation(classifier_name, evaluations)
        if show_confusion_matrices:
            subfig = plt.subplot(2, 3, list(CLASSIFIERS.keys()).index(classifier_name) + 1)
            subfig.imshow(np.mean([e['confusion_matrix'] for e in evaluations], axis=0))
            subfig.set_title(classifier_name)
            subfig.set_xlabel('Predicted label')
            subfig.set_ylabel('True label')
    if show_confusion_matrices:
        plt.show()


def main() -> None:
    for bow_method in ['SIFT', 'ORB']:
        for bow_normalization in [None, 'L1', 'L2']:
            for bow_clusters in [50, 100, 500]:
                for hi_normalization in [None, 'L1', 'L2']:
                    build_classifiers_for_config(
                        bow_method=bow_method,
                        bow_normalization=bow_normalization,
                        bow_clusters=bow_clusters,
                        hi_normalization=hi_normalization
                    )


if __name__ == '__main__':
    main()
