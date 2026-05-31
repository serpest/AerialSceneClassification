# CNN Image Classification

## Usage
Run image classification by passing the path of a single image:

python __main__.py <image_path>

Optional arguments:
--model, -m Model weights: head_finetuned, partial_finetuned, full_finetuned, trained (default: full_finetuned)

Example:
python __main__.py DATASET/UCMerced_LandUse/Images/airplane/airplane00.tif --model partial_finetuned
