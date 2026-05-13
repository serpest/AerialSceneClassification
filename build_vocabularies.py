import os
from pathlib import Path
import random

import cv2
from joblib import Parallel, delayed
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from bow_classification import DescriptorsExtractor, preprocess_image


AID_DATASET_PATH = 'DATASET/AID'


VOCABULARIES_PATH = 'vocabularies'

VISUAL_WORDS_COMPUTATION_METHODS = ['SIFT', 'ORB']
VISUAL_WORDS_COMPUTATION_NORMALIZATIONS = [None, 'L1', 'L2']
VISUAL_WORDS_COMPUTATION_CLUSTERS_NUMBERS = [50, 100, 500]
VISUAL_WORDS_COMPUTATION_CONFIGS = [
    (method, normalization, clusters_number)
    for method in VISUAL_WORDS_COMPUTATION_METHODS
    for normalization in VISUAL_WORDS_COMPUTATION_NORMALIZATIONS
    for clusters_number in VISUAL_WORDS_COMPUTATION_CLUSTERS_NUMBERS
]


def load_aid_images(dataset_path: str = AID_DATASET_PATH, max_images_number: int | None = None):
    base_folder = Path(dataset_path)
    if not base_folder.exists() or not base_folder.is_dir():
        raise FileNotFoundError(f'Base folder not found: {base_folder}')
    image_paths = list(base_folder.glob('**/*.jpg'))
    if max_images_number is not None:
        image_paths = random.sample(image_paths, k=min(max_images_number, len(image_paths)))
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        yield image


def compute_visual_words(descriptors_extractor: DescriptorsExtractor, clusters_number: int,
                         max_images_number: int | None = None) -> None:
    kmeans = MiniBatchKMeans( # TODO
        n_clusters=clusters_number, batch_size=1024, random_state=42
    )
    for i, image in enumerate(load_aid_images(max_images_number=max_images_number)):
        processed_image = preprocess_image(image)
        descriptors = descriptors_extractor.extract(processed_image)
        if descriptors is not None:
            kmeans.partial_fit(descriptors)
        if (i + 1) % 10 == 0:
            print(f'Processed {i + 1} images...')
    return kmeans.cluster_centers_


def convert_config_to_file_path(method: str, normalization: str | None, clusters_number: int,
                                vocabularies_path: str) -> str:
    file_name = f'visual_words_{method}_{normalization}_{clusters_number}.npy'
    return f'{vocabularies_path}/{file_name}'


def save_visual_words(visual_words: np.ndarray, method: str, normalization: str | None, clusters_number: int,
                      vocabularies_path: str = VOCABULARIES_PATH) -> None:
    file_path = convert_config_to_file_path(method, normalization, clusters_number, vocabularies_path)
    np.save(file_path, visual_words)


def decode_visual_words_filename(filename: str):
    parts = filename.split('_')
    method = parts[2]
    normalization = parts[3] if parts[3] != 'None' else None
    clusters_number = int(parts[4].split('.')[0])
    return method, normalization, clusters_number


def load_saved_visual_words(method: str, normalization: str | None, clusters_number: int,
                            vocabularies_path: str = VOCABULARIES_PATH) -> np.ndarray:
    if method not in VISUAL_WORDS_COMPUTATION_METHODS:
        raise ValueError(f'Unsupported method: {method}. Use {VISUAL_WORDS_COMPUTATION_METHODS}')
    if normalization not in VISUAL_WORDS_COMPUTATION_NORMALIZATIONS:
        raise ValueError(f'Unsupported normalization: {normalization}. Use {VISUAL_WORDS_COMPUTATION_NORMALIZATIONS}')
    if clusters_number not in VISUAL_WORDS_COMPUTATION_CLUSTERS_NUMBERS:
        raise ValueError(f'Invalid clusters number: {clusters_number}. Use {VISUAL_WORDS_COMPUTATION_CLUSTERS_NUMBERS}')
    file_path = convert_config_to_file_path(method, normalization, clusters_number, vocabularies_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Visual words file not found: {file_path}')
    visual_words = np.load(file_path)
    return visual_words


def compute_visual_words_for_config(method: str, normalization: str | None, clusters_number: int) -> tuple:
    print(f'Computing visual words for method={method}, normalization={normalization}, clusters_number={clusters_number}...')
    descriptors_extractor = DescriptorsExtractor(method=method, normalization=normalization)
    visual_words = compute_visual_words(descriptors_extractor=descriptors_extractor, clusters_number=clusters_number)
    return visual_words, method, normalization, clusters_number


def main() -> None:
    results = Parallel(n_jobs=-1, prefer='processes')(
        delayed(compute_visual_words_for_config)(method, normalization, clusters_number)
        for method, normalization, clusters_number in VISUAL_WORDS_COMPUTATION_CONFIGS
    )
    for visual_words, method, normalization, clusters_number in results:
        save_visual_words(visual_words, method, normalization, clusters_number)


if __name__ == '__main__':
    main()
