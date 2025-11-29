# Cyst vs Tumor Classifier

A radiomics-based classifier for distinguishing cysts from solid tumors in CT scans.

## Installation

```bash
pip install -e .
```

## Usage

### Generate train/test split
```python
from cyst_classifier.data_utils import generate_train_test_split
import pandas as pd

df = pd.read_csv("data.csv")
train_idx, test_idx = generate_train_test_split(df, test_size=0.2, random_state=42)
df.iloc[train_idx].to_csv("train.csv", index=False)
df.iloc[test_idx].to_csv("test.csv", index=False)
```

### Pre-extract features (RECOMMENDED for fast experimentation)
```bash
# Extract features once (slow)
python -m cyst_classifier.extract_features_script --data train.csv --output features_train.csv
python -m cyst_classifier.extract_features_script --data test.csv --output features_test.csv

# Train on cached features (fast - seconds instead of hours!)
python -m cyst_classifier.main train --data features_train.csv --model logistic --output model.pkl

# Evaluate on cached features (fast)
python -m cyst_classifier.main eval --data features_test.csv --model model.pkl --output-dir results/
```

### Train a model (direct from images - slower)
```bash
python -m cyst_classifier.main train --data train.csv --model logistic --output model.pkl
```

### Inference on single lesion
```bash
python -m cyst_classifier.main infer --image ct.nii.gz --seg mask.nii.gz --model model.pkl
```

### Inference on multi-lesion scan
```bash
python -m cyst_classifier.main infer --image ct.nii.gz --seg mask.nii.gz --model model.pkl --multi-lesion --output result.nii.gz
```

### Evaluate model
```bash
# From cached features (fast)
python -m cyst_classifier.main eval --data features_test.csv --model model.pkl --output-dir results/

# Or directly from images (slower)
python -m cyst_classifier.main eval --data test.csv --model model.pkl --output-dir results/
```

## Performance Tips

**For rapid experimentation**: Pre-extract features with `extract_features_script.py`
- Feature extraction: ~1-2 hours (one time)
- Model training: ~seconds (can repeat many times)
- Space efficient: ~72 bytes per lesion vs ~500KB for image data

**The main script auto-detects** whether you're using feature CSVs or image CSVs, so you can switch seamlessly!

## Data Format

Input CSV should have columns: `seg_path`, `image_path`

Segmentation labels:
- 1: Kidney (ignored)
- 2: Tumor (solid)
- 3: Cyst

## Features

The classifier uses the following radiomics features:
- Mean HU (intensity)
- Standard deviation HU
- Coefficient of variation (std/mean)
- 10th and 90th percentiles
- Entropy (histogram-based)
- GLCM contrast (texture)
- Mean gradient magnitude (edge characteristics)
- Sphericity (shape regularity)
- Fraction of voxels < 20 HU (fluid detection)
