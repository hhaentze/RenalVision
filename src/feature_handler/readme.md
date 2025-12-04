<h1 align="left">
  <img src="../../images/feature_icon.svg" alt="Feature Logo" width="40" style="vertical-align: middle; margin-right: 10px; margin-bottom: 5px" />
  Renal Vision - Feature Engine
</h1>

This module is the **data preparation engine** for the Renal Vision project, responsible for converting raw medical images into standardized, quantifiable feature vectors ($\text{Parquet}$ files) suitable for Machine Learning model training.

The core goal of this module is to decouple the slow, I/O-intensive task of **feature extraction** from the fast, iterative process of **model training and evaluation**.

-----

## ✨ Core Features

  * **Decoupled Architecture:** Features are calculated offline, ensuring models are trained only on features, not pixels. This enables rapid experimentation.
  * **Per-Lesion Analysis:** Automatically identifies all individual connected components (lesions) within a segmentation mask, filters out components smaller than the specified $\text{minimum voxel}$ threshold, and extracts features for each one separately.
  * **$\text{Global Lesion ID}$:** Lesions are assigned an ID based on size, with the largest lesion in the scan always designated as $\text{Lesion 1}$.
  * **Flexible Feature Extraction:** Designed with a `BaseFeatureExtractor` to support different feature types:
      * **Radiomics (Current):** Extracts $\text{HU}$-based intensity, shape, and texture metrics.
      * **Neural Embeddings (Future):** Easily allows integration of foundation models ($\text{ResNet}$, $\text{DINOv2}$) for advanced semantic feature extraction.
  * **Data Augmentation:** Supports generating augmented feature vectors via random spatial and intensity transformations during extraction.
  * **Efficient Storage:** Uses the **$\text{Parquet}$** format for high-speed loading and efficient storage of both scalar radiomics and high-dimensional embeddings.

-----

## 🏗️ Structure and Components

The module is built on three main classes that work together in a pipeline:

### 1\. $\text{BasePreprocessor}$ ($\text{Preprocessing.py}$)

  * **Role:** Handles all image I/O and spatial normalization.
  * **Key Functions:**
      * $\text{Loading}$: Reads $\text{NIfTI}$ files or wraps $\text{NumPy}$ arrays.
      * $\text{Resampling/Windowing}$: Normalizes spacing and clips intensity to $\text{HU}$ window.
      * $\text{Augmentation}$: Applies random spatial and intensity transformations if requested.
      * *Note*: Preserves $\text{HU}$ units by default for radiomics compatibility.

### 2\. $\text{BaseFeatureExtractor}$ ($\text{Base.py}$)

  * **Role:** Orchestrator and abstraction layer.
  * **Key Functions:**
      * $\text{Orchestration}$: Manages the flow from $\text{Preprocessor}$ output to final feature lists.
      * $\text{Component Splitting}$: Splits the segmentation mask into individual lesion components, assigns a unique $\text{lesion\_id}$, and derives the $\text{class\_id}$ for each.
      * $\text{\_extract\_single\_lesion}$: Abstract method implemented by subclasses to define the actual calculation (e.g., radiomics math or neural network forward pass).

### 3\. $\text{FeatureDatasetProcessor}$ ($\text{Dataset.py}$)

  * **Role:** Batch processing and data management.
  * **Key Functions:**
      * $\text{Batch Execution}$: Iterates over the input $\text{CSV}$ of paths, calls the $\text{Extractor}$ for each image, and manages the augmentation loop.
      * $\text{I/O}$: Merges extracted features with original metadata and saves the consolidated data to a $\text{Parquet}$ file.
      * $\text{Loading}$: Provides a static utility function to easily load the final $\text{Parquet}$ dataset for use in the $\text{cyst\_classifier}$ module.

-----

## 🚀 Usage

The primary entry point for batch feature extraction is the command-line interface: $\text{cli.py}$.

### Prerequisites

Your input data must be a $\text{CSV}$ file containing at least the following columns:

  * $\text{image\_path}$
  * $\text{seg\_path}$
  * Any additional metadata columns ($\text{case}$, $\text{patient\_id}$, etc.) you wish to include in the output.

### Extraction Command

```bash
# General Syntax:
# python -m feature_handler.cli --data [INPUT_CSV] --output [OUTPUT_FILE] [OPTIONS]

# Example: Extract radiomics features and generate 3 augmented copies per lesion
python -m feature_handler.cli \
    --data ./data/raw_data.csv \
    --output ./features/radiomics_v1.parquet \
    --extractor radiomics \
    --augment 3 \
    --min-voxels 20 \
    --label-map ./config/label_map.json
```

### Loading Features for Training

Your model training script should use the static loader utility:

```python
from feature_handler.dataset import FeatureDatasetProcessor

# Load the entire feature table as a Pandas DataFrame
df_features = FeatureDatasetProcessor.load_features("./features/radiomics_v1.parquet")

# df_features contains all original metadata + lesion_id, class_id, and feature columns.
# It is now ready for model training!
```

-----
