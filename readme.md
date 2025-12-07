<h1 align="center">
  <img src="./images/icon.svg" alt="Renal Vision Logo" width="50" style="vertical-align: middle; margin-right: 10px;" />
  Renal Vision
</h1>

<h3 align="center">The Modular Lesion Analysis Platform</h3>


<div align="center">
<a href="https://github.com/hhaentze/CystClassifier/actions/workflows/ci.yaml"><img alt="Continuous Integration" src="https://github.com/hhaentze/CystClassifier/actions/workflows/ci.yaml/badge.svg"></a>
<a href="https://github.com/hhaentze/RenalVision/blob/main/License.txt"><img alt="License: Apache" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

<a href="https://www.comfort-ai.eu/for-patients/kidney-cancer">
  <img alt="Classification Paper" src="https://img.shields.io/badge/paper-classification-red.svg">
</a>

</div>

RenalVision is a modular, high-performance platform for quantifying and classifying medical imaging lesions. Its architecture is completely modality-agnostic, separating the [**Data Engine**](src/renal_vision/features) from the [**Machine Learning Logic**](src/renal_vision/modeling).

This platform allows researchers to decouple the heavy lifting of image processing (Radiomics, Neural Embeddings) from the rapid iteration of model training.

## 🌟 Core Features

* **Modular Architecture:** Explicit separation between Feature Extraction (`renal_vision/features`) and Model Training (`renal_vision/modeling`).
* **Offline Feature Store:** Converts heavy NIfTI/NRRD/MHA datasets into lightweight, efficient Parquet feature stores.
* **Self-Contained Models:** Trained models (`ModelBundle`) store their own preprocessing configuration, class mappings, and scaling logic, ensuring reproducible inference.
* **Robust Inference:** Classify single or multiple lesions in a scan at once without any additional configurations. `LesionPredictor` handles it for you.
* **Explainable-Ready:** Built-in support for classical ML (Logistic Regression, Decision Trees) and Gradient Boosting (XGBoost).

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/hhaentze/RenalVision.git
cd RenalVision

# Install in editable mode
pip install -e .`
```

## 🚀 Quick Start (CLI)
The platform exposes a unified command-line interface: rv.

### 1. Extract Features (Data Engine)
Convert raw images and masks into a feature table. Mask values (excl. 0) are use to deduct class_id's that are assigned to the feature vectors.

**Important:**
* Indexing of classes in the segmentation masks starts at 1
* Indexing of classes in the extracted feature vectors starts at 0

```bash
rv extract \
    --data ./data/dataset.csv \
    --output ./data/features/radiomics_v1.parquet \
    --extractor radiomics \
    --augment 3
```


### 2. Train & Evaluate Model (Logic Engine)
Utilize the extracted features.

```bash
rv train \
    --data ./data/features/radiomics_v1.parquet \
    --model xgboost \
    --output-dir ./models/v1

rv eval \
    --data ./data/features/radiomics_v2.parquet \
    --model ./models/v1/model.pkl \
    --output-dir ./models/v1
```

### 3. Run Inference
Predict on new scans using the trained model bundle.

```bash
rv infer \
    --image ./new_data/scan_001.nii.gz \
    --seg ./new_data/mask_001.nii.gz \
    --model ./models/v1/model.pkl \
    --output ./results/prediction_001.nii.gz
```

##

## 🐍 Python API
RenalVision is designed to be used programmatically for custom pipelines.

### Loading Features for Custom Training
```python

from src.features.dataset import FeatureDatasetProcessor

# Load the Parquet store as a Pandas DataFrame
# Contains metadata (case_id, lesion_id) + feature columns
df = FeatureDatasetProcessor.load_features("./data/features/radiomics_v1.parquet")

print(df.head())
Running Inference in Your Script
Python

from src.modeling.inference import LesionPredictor

# 1. Initialize Predictor (Auto-loads extractor config from the model)
predictor = LesionPredictor(model_path="./models/v1/model.pkl")

# 2. Predict a single lesion
result = predictor.infer_lesion(image="scan.nii.gz", seg="lesion_mask.nii.gz")
print(f"Prediction: {result['class_name']} ({result['confidence']:.1%})")

# 3. Predict full mask (multi-lesion)
predictor.infer_mask(
    image="scan.nii.gz",
    seg="full_mask.nii.gz",
    output_path="predictions.nii.gz"
)
```

## 🔌 Expandability
The platform is designed for extension.

### Adding New Feature Extractors:

1. Create a new class in `src/features/` inheriting from `BaseFeatureExtractor`.

2. Implement `_extract_single_lesion` (logic) and `get_config` (serialization).

3. Register it in the `_reconstruct_extractor factory` in `src/modeling/inference.py`.

### Adding New Models:

1. Add the architecture to `src/modeling/models.py` inside ModelFactory.

2. Update the `ModelType` enum.

### Adding Custom Preprocessing:

1. Create a `CustomPreprocessor` in `src/features/preprocessing.py` inheriting from `BasePreprocessor`.

2. Pass this preprocessor to your Extractor.


## 📝 How to Contribute

To mantain hiqh quality code please adhere to our coding guidelines. You can run the full Continuous Integration pipeline locally. This ensures your code is clean, typed correctly, and fully tested.

  * **Run All Checks:**

    ```bash
    make ci
    ```

    This single command executes the workflow in the following order: **Linting** (Ruff/Black) $\rightarrow$ **Type Checking** (Mypy) $\rightarrow$ **Unit Tests & Coverage** (Pytest).

  * **Fix Code Style Automatically:**
    If the `make ci` command reports style or formatting errors, you can fix them instantly using:

    ```bash
    make format
    ```

Once your code is formatted correctly and passes `make ci` locally, push your changes and open a Pull Request. Our GitHub Actions pipeline will mirror your local `make ci` run and upload a final test coverage report.
