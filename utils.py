import os
import joblib
import numpy as np

from config import CLASSIFIERS_PATH, VOCABULARIES_PATH


def convert_classifier_config_to_file_path(classifier_name: str, bow_method: str,
                                           bow_normalization: str, bow_clusters: int,
                                           hi_normalization: str) -> str:
    file_name = f'{classifier_name}_{bow_method}_{bow_normalization}_{bow_clusters}_{hi_normalization}.pkl'
    return f'{CLASSIFIERS_PATH}/{file_name}'


def load_classifier(classifier_name: str, bow_method: str, bow_normalization: str,
                    bow_clusters: int, hi_normalization: str) -> object:
    file_path = convert_classifier_config_to_file_path(classifier_name, bow_method, bow_normalization, bow_clusters, hi_normalization)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Classifier file not found: {file_path}. Run build_classifiers.py to build the classifiers first')
    return joblib.load(file_path)


def convert_visual_words_config_to_file_path(method: str, normalization: str | None, clusters_number: int) -> str:
    file_name = f'visual_words_{method}_{normalization}_{clusters_number}.npy'
    return f'{VOCABULARIES_PATH}/{file_name}'


def load_visual_words(method: str, normalization: str | None, clusters_number: int) -> np.ndarray:
    file_path = convert_visual_words_config_to_file_path(method, normalization, clusters_number)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Visual words file not found: {file_path}. Run build_vocabularies.py to build the visual words first')
    visual_words = np.load(file_path)
    return visual_words
