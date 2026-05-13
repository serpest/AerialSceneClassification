import argparse

from bow_classification import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run image classification using BoW.')
    parser.add_argument('image_path', help='Path to the image to classify.')
    parser.add_argument('--bow_method', choices=['SIFT', 'ORB'], default='SIFT', help='Method for computing visual words (default: SIFT).')
    parser.add_argument('--bow_normalization', choices=[None, 'L1', 'L2'], default='L2', help='Normalization for visual words histograms (default: L2).')
    parser.add_argument('--bow_clusters', choices=[50, 100, 500], type=int, default=100, help='Number of clusters for visual words (default: 100).')
    parser.add_argument('--hi_normalization', choices=[None, 'L1', 'L2'], default='L2', help='Normalization for histograms of visual words (default: L2).')
    parser.add_argument('--classifier', choices=['SVM_linear', 'SVM_RBF', 'Random_Forest', 'k-NN', 'Logistic_Regression'], default='Random_Forest', help='Classifier to use for prediction (default: Random Forest).')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        image_path=args.image_path,
        bow_method=args.bow_method,
        bow_normalization=args.bow_normalization,
        bow_clusters=args.bow_clusters,
        hi_normalization=args.hi_normalization,
        classifier=args.classifier
    )
    print(f'Predicted label: {result}')


if __name__ == '__main__':
    main()
