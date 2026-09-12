import argparse

from bovw_classification import classify_image
from build_classifiers import CLASSIFIERS
from build_vocabularies import VISUAL_WORDS_CLUSTERS_NUMBERS, VISUAL_WORDS_METHODS

NORMALIZATION_CHOICES = ['None', 'L1', 'L2']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run image classification using BoVW.')
    parser.add_argument('image_path', help='Path to the image to classify.')
    parser.add_argument('--bovw_method', choices=VISUAL_WORDS_METHODS, default='SIFT', help='Method for computing visual words (default: SIFT).')
    parser.add_argument('--bovw_normalization', choices=NORMALIZATION_CHOICES, default='None', help='Normalization for visual words histograms (default: None).')
    parser.add_argument('--bovw_clusters', choices=VISUAL_WORDS_CLUSTERS_NUMBERS, type=int, default=500, help='Number of clusters for visual words (default: 500).')
    parser.add_argument('--hi_normalization', choices=NORMALIZATION_CHOICES, default='L2', help='Normalization for histograms of visual words (default: L2).')
    parser.add_argument('--classifier', choices=list(CLASSIFIERS.keys()), default='SVM_RBF', help='Classifier to use for prediction (default: SVM_RBF).')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bovw_normalization = args.bovw_normalization if args.bovw_normalization != 'None' else None
    hi_normalization = args.hi_normalization if args.hi_normalization != 'None' else None
    result = classify_image(
        image_path=args.image_path,
        bovw_method=args.bovw_method,
        bovw_normalization=bovw_normalization,
        bovw_clusters=args.bovw_clusters,
        hi_normalization=hi_normalization,
        classifier=args.classifier
    )
    print(f'Predicted label: {result}')


if __name__ == '__main__':
    main()
