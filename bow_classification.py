import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

from utils import load_classifier, load_visual_words


class DescriptorsExtractor:

    def __init__(self, method: str, normalization: str | None = None, descriptors_number: int | None = None) -> None:
        if method not in ['SIFT', 'ORB']:
            raise ValueError(f'Unsupported method: {method}. Use "SIFT" or "ORB"')
        if normalization not in [None, 'L1', 'L2']:
            raise ValueError(f'Unsupported normalization: {normalization}. Use None, "L1" or "L2"')
        if method == 'SIFT':
            self.detector = cv2.SIFT_create(nfeatures=descriptors_number)
        elif method == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=descriptors_number)
        self.normalization = normalization
        
    def extract(self, image: np.ndarray) -> np.ndarray | None:
        # Image should be already preprocessed
        _, descriptors = self.detector.detectAndCompute(image, None)
        if descriptors is not None and self.normalization is not None:
            if self.normalization == 'L1':
                descriptors = descriptors / np.maximum(np.sum(descriptors, axis=1, keepdims=True), 1e-8)
            elif self.normalization == 'L2':
                descriptors = descriptors / np.maximum(np.linalg.norm(descriptors, axis=1, keepdims=True), 1e-8)
        return descriptors


class VisualWordsHistogramComputer:

    def __init__(self, descriptors_extractor: DescriptorsExtractor, visual_words: np.ndarray,
                 normalization: str | None = None) -> None:
        if normalization not in [None, 'L1', 'L2']:
            raise ValueError(f'Unsupported normalization: {normalization}. Use None, "L1" or "L2"')
        self.descriptors_extractor = descriptors_extractor
        self.visual_words = visual_words
        self.normalization = normalization
        self.nearest_neighbors = NearestNeighbors(n_neighbors=1).fit(visual_words)

    def compute_histogram(self, image: np.ndarray) -> np.ndarray | None:
        processed_image = preprocess_image(image)
        descriptors = self.descriptors_extractor.extract(processed_image)
        if descriptors is None:
            # raise ValueError('No descriptors found in the image')
            return None
        _, centroids = self.nearest_neighbors.kneighbors(descriptors)
        centroids = centroids.ravel()
        # histogram = np.zeros(self.visual_words.shape[0])
        # for centroid in centroids:
        #     histogram[centroid] += 1
        histogram = np.bincount(centroids, minlength=self.visual_words.shape[0])
        if self.normalization is not None:
            if self.normalization == 'L1':
                histogram = histogram / np.maximum(np.sum(histogram), 1e-8)
            elif self.normalization == 'L2':
                histogram = histogram / np.maximum(np.linalg.norm(histogram), 1e-8)
        return histogram


def preprocess_image(image: np.ndarray) -> np.ndarray:
    # TODO: Skipping preprocessing is ok?
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


def classify_image(image_path: str, bow_method: str = 'SIFT', bow_normalization: str | None = None, bow_clusters: int = 500,
                   hi_normalization: str | None = 'L2', classifier: str = 'SVM_RBF') -> str:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f'Image not read: {image_path}')
    descriptors_extractor = DescriptorsExtractor(method=bow_method, normalization=bow_normalization)
    visual_words = load_visual_words(method=bow_method, normalization=bow_normalization, clusters_number=bow_clusters)
    histogram_computer = VisualWordsHistogramComputer(
        descriptors_extractor=descriptors_extractor, visual_words=visual_words, normalization=hi_normalization
    )
    histogram = histogram_computer.compute_histogram(image)
    if histogram is None:
        raise ValueError('cannot compute histogram for the image')
    histogram = histogram.reshape(1, -1) # Convert shape from (n,) to (1, n)
    classifier = load_classifier(
        classifier_name=classifier, bow_method=bow_method, bow_normalization=bow_normalization,
        bow_clusters=bow_clusters, hi_normalization=hi_normalization
    )
    predicted_label = classifier.predict(histogram)[0]
    return predicted_label
