<h1 align="center">
  <img src="./images/icon.svg" alt="Renal Vision Logo" width="50" style="vertical-align: middle; margin-right: 10px;" />
  Renal Vision
</h1>

<h3 align="center">The Modular Lesion Analysis Platform</h3>


<div align="center">
<a href="https://github.com/hhaentze/CystClassifier/actions/workflows/ci.yaml"><img alt="Continuous Integration" src="https://github.com/hhaentze/CystClassifier/actions/workflows/ci.yaml/badge.svg"></a>
<a href="https://github.com/hhaentze/RenalVision/blob/main/License.txt"><img alt="License: Apache" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
<a href="https://www.comfort-ai.eu/for-patients/kidney-cancer">
  <img alt="website" src="https://img.shields.io/badge/Website-COMFORT-darkblue.svg"></a>
<a href="https://github.com/astral-sh/ruff"><img alt="Code style: ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
<a href="https://arxiv.org/abs/2605.07749"><img alt="Preprint" src="https://img.shields.io/badge/Preprint-arXiv-b31b1b"></a>

</div>

RenalVision is a modular platform for quantifying and classifying renal lesions. It keeps feature extraction (Radiomics, Neural Embeddings) cleanly separated from model training and evaluation, so swapping in a different backbone or classifier is straightforward. Trained models are packaged as self-contained bundles that carry their own preprocessing config and class mappings, making inference portable and reproducible.
The platform ships with eight pre-trained models out of the box: `radiomics_binary`, `radiomics`, `mevis`, `mevis_unicorn`, `fmcib`, `ctfm`, `spectre`, and `renalclip`. Read more about the bundles in their [readme.md](src/renal_vision/bundles/readme.md).

![Workflow](images/workflow.webp)

## 🛠️ Installation
We recommend Python 3.10. The base install covers radiomics support. If you want to run foundation models please specify these as additional dependencies or use the `install-all` option.
```bash
# Clone or download the repository
git clone https://github.com/hhaentze/RenalVision.git
cd RenalVision

# Base install (radiomics only)
make install

# [Optional] Full install including all FM extractors
make install-all
```

## 🔬 Inference
The platform exposes a unified command-line interface: rv.
```bash
rv infer \
    --image scan_001.nii.gz \
    --seg mask_001.nii.gz \
    --output prediction_001.nii.gz \
    --model "RADIOMICS"

```
Alternatively, use our python interface! You can choose between image-level classification that returns an updated segmentation mask, or lesion-level, which returns details information about a specific lesion prediction.
```python
from renal_vision.modeling.inference import LesionPredictor

# 1. Initialize Predictor
predictor = LesionPredictor(model_identifier="RADIOMICS_BINARY")

# 2. Predict full mask (multi-lesion)
mask = predictor.infer_mask(
    image="scan.nii.gz", seg="full_mask.nii.gz", output_path="predictions.nii.gz"
)

# 3. Predict a single lesion
result = predictor.infer_lesion(image="scan.nii.gz", seg="lesion_mask.nii.gz")
print(result)
```

```json
{
  'class_id': 0,                    # predicted class
  'class_name': 'Tumor',            # predicted class name
  'confidence': 0.997,              # proability of predicted class
  'probability': [0.997, 0.003],    # proabilities of all classes
  'volume': 8726                    # volume of target lesions in mm^3
 }
```

## 🔧 Useful Tools from our Lab:

- Automatic segmentation of kidneys and masses: [RenalNet](https://github.com/DIAGNijmegen/oncology-kidney-abnormality-segmentation)
- Pre-annotated TCGA kidney tumour data: [RCC-AID](https://zenodo.org/records/20719257)



## 📦 Demo Workflow
The [KiTS23 dataset](https://github.com/neheller/kits23) is publicly available and can be downloaded via their official starter kit:
```bash
git clone https://github.com/neheller/kits23
cd kits23
pip3 install .
kits23_download_data
```
This will download the CT scans and segmentation masks to ./dataset/ (~46GB). The masks include kidney, tumor, and cyst labels. (If you don't have 46GB of available storage you can simply stop the download earlier.)

Check out how to use this dataset to train and evaluate your own classification model. No GPU needed! [Tutorial.ipynb](notebooks/tutorial.ipynb)

Alternatively, test our own model directly on the KiTS data: [Demo.ipynb](notebooks/demo.ipynb)



## 📋 Citation
Paper under Review. Please cite this preprint: [arXiv](https://arxiv.org/abs/2605.07749).

If you used any of the FM based classifiers in your research please refer to their GitHub for citation guidelines:
- [FMCIB](https://github.com/AIM-Harvard/foundation-cancer-image-biomarker)
- [CTFM](https://github.com/project-lighter/CT-FM)
- [MMM](https://github.com/FraunhoferMEVIS/MedicalMultitaskModeling)
- [SPECTRE](https://github.com/cclaess/SPECTRE)
- [RenalCLIP](https://github.com/dt-yuhui/RenalCLIP)
