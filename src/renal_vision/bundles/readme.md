# Renal Vision - Pretrained Bundles

We trained these models on scans from both Charité and KiTS.
In total 4315 lesions were extracted from 1309 patients.

You can use them either in the CLI:
```bash
rv infer
```

or with our python API:
```python
from renal_vision.modeling.inference import LesionPredictor
from renal_vision.bundles import ImplementedModels

# Either use our enum class
predictor1 = LesionPredictor(model_identifier=ImplementedModels.HISTOLOGY_SUBTYPE)

# Or just pass the string directly
predictor2 =  LesionPredictor(model_identifier="RADIOMICS_BINARY")
```

All models are xgboost classifer trained on KiTS & Charité (10-fold cross-validated)

### 1. Radiomics based Tumor/Cyst classifier
- identifier: **RADIOMICS_BINARY**
Classes:
```python
{
    0: "Tumor",
    1: "Cyst",
}
```

<div align="center">
  <img src="radiomics_binary/cv_confusion_matrix.webp" alt="Description of Image 1" width="30%">
  <img src="radiomics_binary/cv_roc_curve.webp" alt="Description of Image 2" width="30%">
  <img src="radiomics_binary/cv_pr_curve.webp" alt="Description of Image 3" width="30%">
</div>

### 2 Histology-subtype classifier
Classes:
```python
{
    0: "ccRCC",
    1: "pRCC",
    2: "chrRCC",
    3: "Oncocytoma",
    4: "Cyst",
    5: "Other",
}
```
- identifier: **RADIOMICS**
<div align="center">
  <img src="radiomics/cv_roc_curve.webp" alt="Description of Image 2" width="30%">
  <img src="radiomics/cv_pr_curve.webp" alt="Description of Image 3" width="30%">
</div>

- identifier: **MEVIS**
<div align="center">
  <img src="mevis/cv_roc_curve.webp" alt="Description of Image 2" width="30%">
  <img src="mevis/cv_pr_curve.webp" alt="Description of Image 3" width="30%">
</div>

- identifier: **FMCIB**
<div align="center">
  <img src="fmcib/cv_roc_curve.webp" alt="Description of Image 2" width="30%">
  <img src="fmcib/cv_pr_curve.webp" alt="Description of Image 3" width="30%">
</div>

- identifier: **CTFM**
<div align="center">
  <img src="ctfm/cv_roc_curve.webp" alt="Description of Image 2" width="30%">
  <img src="ctfm/cv_pr_curve.webp" alt="Description of Image 3" width="30%">
</div>
