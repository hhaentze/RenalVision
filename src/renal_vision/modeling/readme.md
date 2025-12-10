

<h1 align="left">
  <img src="../../../images/model_icon.svg" alt="Model Logo" width="40" style="vertical-align: middle; margin-right: 10px; margin-bottom: 5px" />
  Renal Vision - Logic Engine
</h1>

This module contains the "Brain" of the RenalVision platform. It is responsible for learning patterns from feature vectors and generating predictions on new data.

Unlike the [features](../features) module, this module is **preprocessing-agnostic**. It operates primarily on tabular data (Parquet/Numpy) and does not handle image I/O directly, except during the final Inference stage.

## 📦 The ModelBundle

We do not simply save sklearn objects. Instead, we wrap the entire training context into a `ModelBundle` (persisted as model.pkl). This ensures the model is self-documenting and portable.

A ModelBundle contains:
1.  **The Classifier:** (e.g., XGBClassifier, LogisticRegression).
2.  **The Scaler:** (e.g., StandardScaler, if used).
3.  **Extractor Configuration:** A complete dictionary describing *how* features were extracted (e.g., "Radiomics with window width 400"). This allows inference to reconstruction the pipeline automatically.
4.  **Class Mappings:** Stores `{0: "Tumor", 1: "Cyst"}` internally, ensuring output is always human-readable.
5.  **Transform Logic:** Knows which features require log-transformation (e.g., gradient_magnitude) before prediction.

## 🧠 Inference Logic: All In One

Inference sounds easy, but you may ask yourself: Which spacing and orientation do I need? Do I need to normalize the images? Which class names correspond to the predicted ids?

All of this is stored in the model bundle! In the exact same configurations as  used during training. All you need is an image and a segmentation mask of a typical medical format and the LesionPredictor class (`inference.py`) handles the rest.

## 🤖 Supported Algorithms

The ModelFactory currently supports:

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
* `pr_curves.png`: Multi-class Precision-Recall curves (One-vs-Rest) with Micro/Macro averages.
