import random
import cv2
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans

from bow_classification import DescriptorsExtractor, preprocess_image
from config import AID_DATASET_PATH, VOCABULARIES_PATH
from utils import convert_visual_words_config_to_file_path

VISUAL_WORDS_METHODS = ['SIFT', 'ORB']
VISUAL_WORDS_NORMALIZATIONS = [None, 'L1', 'L2']
VISUAL_WORDS_CLUSTERS_NUMBERS = [50, 100, 500]

VISUAL_WORDS_CONFIGS = [
    (method, normalization, clusters_number)
    for method in VISUAL_WORDS_METHODS
    for normalization in VISUAL_WORDS_NORMALIZATIONS
    for clusters_number in VISUAL_WORDS_CLUSTERS_NUMBERS
]


def load_aid_images(dataset_path: str = AID_DATASET_PATH, max_images_number: int | None = None):
    base_folder = Path(dataset_path)
    if not base_folder.exists() or not base_folder.is_dir():
        raise FileNotFoundError(f'Base folder not found: {base_folder}')
    random_generator = random.Random(42)
    image_paths = list(base_folder.glob('**/*.jpg'))
    random_generator.shuffle(image_paths)
    if max_images_number is not None:
        image_paths = image_paths[:max_images_number]
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        yield image


def compute_visual_words(descriptors_extractor: DescriptorsExtractor, clusters_number: int,
                         max_images_number: int | None = None, passes_number: int = 1) -> np.ndarray:
    kmeans = MiniBatchKMeans(
        n_clusters=clusters_number, batch_size=1024, random_state=42
    )
    for _ in range(passes_number):
        for i, image in enumerate(load_aid_images(max_images_number=max_images_number)):
            processed_image = preprocess_image(image)
            descriptors = descriptors_extractor.extract(processed_image)
            if descriptors is not None:
                kmeans.partial_fit(descriptors)
            if (i + 1) % 1000 == 0:
                print(f'Processed {i + 1} images...')
    return kmeans.cluster_centers_


def save_visual_words(visual_words: np.ndarray, method: str, normalization: str | None, clusters_number: int) -> None:
    Path(VOCABULARIES_PATH).mkdir(parents=True, exist_ok=True)
    file_path = convert_visual_words_config_to_file_path(method, normalization, clusters_number)
    np.save(file_path, visual_words)


def compute_visual_words_wrapped(method: str, normalization: str | None, clusters_number: int) -> None:
    # Function used for parallel computation
    print(f'Computing visual words for method={method}, normalization={normalization}, clusters_number={clusters_number}...')
    descriptors_extractor = DescriptorsExtractor(method=method, normalization=normalization)
    visual_words = compute_visual_words(descriptors_extractor=descriptors_extractor, clusters_number=clusters_number)
    save_visual_words(visual_words, method, normalization, clusters_number)


def main() -> None:
    Parallel(n_jobs=-1, prefer='processes')(
        delayed(compute_visual_words_wrapped)(method, normalization, clusters_number)
        for method, normalization, clusters_number in VISUAL_WORDS_CONFIGS
    )


if __name__ == '__main__':
    main()
