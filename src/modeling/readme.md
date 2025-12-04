

<h1 align="left">
  <img src="../../images/model_icon.svg" alt="Model Logo" width="40" style="vertical-align: middle; margin-right: 10px; margin-bottom: 5px" />
  Renal Vision - Feature Engine
</h1>

This module contains the "Brain" of the RenalVision platform. It is responsible for learning patterns from feature vectors and generating predictions on new data.

Unlike the `features` module, this module is **modality-agnostic**. It operates primarily on tabular data (Parquet/Numpy) and does not handle image I/O directly, except during the final Inference stage.

## 📦 The `ModelBundle`

We do not simply save `sklearn` objects. Instead, we wrap the entire training context into a `ModelBundle` (persisted as `model.pkl`). This ensures the model is self-documenting and portable.

A `ModelBundle` contains:
1.  **The Classifier:** (e.g., `XGBClassifier`, `LogisticRegression`).
2.  **The Scaler:** (e.g., `StandardScaler`, if used).
3.  **Extractor Configuration:** A complete dictionary describing *how* features were extracted (e.g., "Radiomics with window width 400"). This allows inference to reconstruction the pipeline automatically.
4.  **Class Mappings:** Stores `{0: "Tumor", 1: "Cyst"}` internally, ensuring output is always human-readable.
5.  **Transform Logic:** Knows which features require log-transformation (e.g., `gradient_magnitude`) before prediction.

## 🧠 Inference Logic: World Coordinate Matching

The most complex component of this module is `LesionPredictor` (`inference.py`). It bridges the gap between **Image Space** and **Feature Space**.

**The Problem:**
Preprocessing (in `src/features`) often resamples, crops, or rotates images. This means voxel indices $(i, j, k)$ in the feature map do not match voxel indices in the user's original segmentation mask.

**The Solution:**
We implement **World Coordinate Matching**.
1.  **Extraction:** When features are extracted, we calculate the centroid of the lesion in physical space (millimeters) using the image's affine matrix.
2.  **Prediction:** We calculate the physical centroids of components in the user's original mask.
3.  **Matching:** We map predictions to original components by finding the nearest neighbor in physical space (Euclidean distance).

This ensures robust predictions even if the preprocessor drastically changes the image grid.

## 🤖 Supported Algorithms

The `ModelFactory` currently supports:

| Model | Type | Use Case |
| :--- | :--- | :--- |
| **Logistic Regression** | `logistic` | Baseline, high interpretability (coefficients). |
| **Decision Tree** | `tree` | Simple rules, non-linear boundaries. |
| **XGBoost** | `xgboost` | High performance, gradient boosting state-of-the-art. |

## 📊 Evaluation

Use `eval.py` (via `rv eval`) to generate standard performance reports from a test set.

**Outputs:**
* `metrics.json`: Raw values (Accuracy, F1, Sensitivity, Specificity).
* `confusion_matrix.png`: Seaborn visualization of class confusion.
* `roc_curves.png`: Multi-class ROC curves (One-vs-Rest) with Micro/Macro averages.
