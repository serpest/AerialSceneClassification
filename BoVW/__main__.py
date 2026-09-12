import argparse

from bow_classification import classify_image
from build_classifiers import CLASSIFIERS
from build_vocabularies import VISUAL_WORDS_CLUSTERS_NUMBERS, VISUAL_WORDS_METHODS

NORMALIZATION_CHOICES = ['None', 'L1', 'L2']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run image classification using BoW.')
    parser.add_argument('image_path', help='Path to the image to classify.')
    parser.add_argument('--bow_method', choices=VISUAL_WORDS_METHODS, default='SIFT', help='Method for computing visual words (default: SIFT).')
    parser.add_argument('--bow_normalization', choices=NORMALIZATION_CHOICES, default='None', help='Normalization for visual words histograms (default: None).')
    parser.add_argument('--bow_clusters', choices=VISUAL_WORDS_CLUSTERS_NUMBERS, type=int, default=500, help='Number of clusters for visual words (default: 500).')
    parser.add_argument('--hi_normalization', choices=NORMALIZATION_CHOICES, default='L2', help='Normalization for histograms of visual words (default: L2).')
    parser.add_argument('--classifier', choices=list(CLASSIFIERS.keys()), default='SVM_RBF', help='Classifier to use for prediction (default: SVM_RBF).')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bow_normalization = args.bow_normalization if args.bow_normalization != 'None' else None
    hi_normalization = args.hi_normalization if args.hi_normalization != 'None' else None
    result = classify_image(
        image_path=args.image_path,
        bow_method=args.bow_method,
        bow_normalization=bow_normalization,
        bow_clusters=args.bow_clusters,
        hi_normalization=hi_normalization,
        classifier=args.classifier
    )
    print(f'Predicted label: {result}')


if __name__ == '__main__':
    main()
