import argparse
from pathlib import Path
import cv2

from classification import classify_image_on_ucmlu


MODEL_CHOICES = ['head_finetuned', 'partial_finetuned', 'full_finetuned', 'trained']


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description='Run CNN image classification.')
	parser.add_argument('image_path', help='Path to the image to classify.')
	parser.add_argument(
		'--model',
		'-m',
		choices=MODEL_CHOICES,
		default='partial_finetuned',
		help='Model weights to use (default: partial_finetuned).'
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	result = classify_image_on_ucmlu(image_path=args.image_path, model_name=args.model)
	print(f'Predicted label: {result}')


if __name__ == '__main__':
	main()
