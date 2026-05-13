# BoW Image Classification

## Usage

Run image classification by passing the path to a single image:

```bash
python __main__.py <image_path>
```

Optional arguments:

- `--bow_method` Visual words method: `SIFT` or `ORB` (default: `SIFT`)
- `--bow_normalization` Normalization for visual words: `None`, `L1` or `L2` (default: `L2`)
- `--bow_clusters` Number of visual word clusters: `50`, `100` or `500` (default: `100`)
- `--hi_normalization` Normalization for histogram of visual words: `None`, `L1` or `L2` (default: `L2`)
- `--classifier` Classifier: `SVM_Linear`, `SVM_RBF`, `Random_Forest`, `k-NN` or `Logistic_Regression` (default: `Random_Forest`)

Example:

```bash
python __main__.py DATASET/AID/Airport/image_1.jpg --bow_method SIFT --bow_clusters 100 --classifier Random_Forest
```
