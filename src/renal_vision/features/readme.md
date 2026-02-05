<h1 align="left">
  <img src="../../../images/feature_icon.svg" alt="Feature Logo" width="40" style="vertical-align: middle; margin-right: 10px; margin-bottom: 5px" />
  Renal Vision - Data Engine
</h1>

The **Data Engine** is the foundation of the RenalVision platform. It is responsible for the ingestion, preprocessing, and quantification of medical images. Its primary role is to convert raw volumetric data (e.g., NIfTI, NRRD, MHA) into structured, machine-learnable feature vectors stored in **Parquet** format.

This module is stateless and completely decoupled from the downstream modeling logic.

## 🌟 Key Capabilities

* **Modality Agnostic:** Built on top of MONAI, enabling support for typical biomedical imaging formats like NIfTI, NRRD, and MHA.
* **Lesion Detection:** Automatically scans segmentation masks to identify and isolate individual connected components (lesions) for analysis.
* **MetaTensor Powered:** Relies exclusively on MONAI's MetaTensor architecture. This ensures that spatial metadata (affine matrices, spacing) is carried implicitly through the pipeline without manual management, reducing the risk of incorrect resampling.
* **Augmentation Pipeline:** Supports generating $N$ augmented feature vectors per lesion (via random spatial and intensity transformations) to upsample rare classes before training.

## 📐 Architecture

### 1. The Preprocessor (`preprocessing.py`)
Handles the "Pixels".
* **Input:** File paths or MetaTensors.
* **Operations:** Spacing normalization, Intensity Windowing, and standardizing inputs into MONAI MetaTensor objects.
* **Configuration:** Stores exact windowing parameters to ensure the Inference phase replicates the Training phase 1:1.

### 2. The Extractor (`base.py`, `radiomics.py`)
Handles the "Math".
* **Radiomics:** Implements standard features (Shape, Intensity, GLCM Texture).
* **Foundation Models (TODO):** The `BaseFeatureExtractor` interface is designed to support Deep Learning extractors (e.g., FMCIB) that map lesion crops to embedding vectors.

### 3. The Batch Processor (`dataset.py`)
Handles the "Scale".
* Orchestrates the iteration over dataset manifests (CSVs).
* Manages memory-efficient writing to **Parquet** files.

## 🚀 Usage

Access the Data Engine via the unified `rv` command.

### Basic Extraction
Extract a feature vector for each connected component in the segmentation mask. The the values of the segmentation masks (minus one) are stored as class ids. If you want to configure this you can pass your own label mapping. For example in the KiTS data we would like to ignore kidney masks and include tumors and cysts, so we create a custom label mapping: `{1:0,2:1,3:2}`.

**Important:**
* Indexing of classes in the segmentation masks starts at 1
* Indexing of classes in the extracted features starts at 0
```bash
rv extract \
    --data ./data/train_data.csv \
    --output ./data/features/train_features.parquet \
    --extractor radiomics \
    --label-map label-map.json \
    --min-volume 400 \ # exclude lesions smaller than 400 mm^3 (default)
    --augment 5 # generate 5 (additional) synthetic variations for every lesion
```
### Output Format
The resulting Parquet file contains flat rows with metadata and features:
| case_id | filepath | lesion_id | class_id | ... | mean_hu | sphericity | augment |
|---------|----------|-----------|----------|-----|---------|------------|---------|
| case_01 | image1.nii.gz| 1         | 1        | ... | 85.4    | 0.65       |True      |
| case_01 | image1.nii.gz| 2         | 2        | ... | 4.2     | 0.98       |True      |

Additionally, a copy of all extractor settings is saved next to the feature file.
