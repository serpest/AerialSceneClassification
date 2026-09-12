# Aerial Scene Classification: Residual CNN

## Usage

Run image classification by passing the path of a single image:

```bash
python __main__.py <image_path>
```

Optional arguments:

- `--model` Model weights: `head_finetuned`, `partial_finetuned`, `full_finetuned` or `trained` (default: `full_finetuned`)

Example:

```bash
python __main__.py DATASET/UCMerced_LandUse/Images/airplane/airplane00.tif --model partial_finetuned
```
