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
We recommend Python 3.10. The base install covers radiomics support. If you want to run foundation models please specify these as additional dependencies: run `make install` once, then any number of the commands below.
Run `make list-models` for the current list.
(Our implementations of M3 default and M3 unicorn are incompatible with each other. If you want to run both, please specify independent virtual environments.)

| Model         | Command                    | Notes                          |
|---------------|----------------------------|--------------------------------|
| Radiomics     | `make install`             | included in base               |
| M3 (default)  | `make model-mevis`         | ⚠️ conflicts with M3 UNICORN   |
| M3 (UNICORN)  | `make model-mevis_unicorn` | ⚠️ conflicts with M3 default   |
| RenalCLIP     | `make install`             | included in base     |
| SPECTRE       | `make model-spectre`       |                                |
| CT-FM         | `make model-ctfm`          | installed `--no-deps` |
| FMCIB         | `make model-fmcib`         | installed `--no-deps` |

```bash
# Clone or download the repository
git clone https://github.com/hhaentze/RenalVision.git
cd RenalVision

# Base install (radiomics only)
make install

# [Optional] install other dependencies, for example the M3 (unicorn) model
make model-mevis_unicorn
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

Test the pretrained model directly on data from RCC-AID: [Inference.ipynb](notebooks/inference.ipynb)

Train and evaluate your own classification model on the KiTS23 data. No GPU needed! [Training.ipynb](notebooks/training.ipynb)


## 📋 Citation
Paper under Review. Please cite this preprint: [arXiv](https://arxiv.org/abs/2605.07749).

If you used any of the FM based classifiers in your research please refer to their GitHub for citation guidelines:
- [FMCIB](https://github.com/AIM-Harvard/foundation-cancer-image-biomarker)
- [CTFM](https://github.com/project-lighter/CT-FM)
- [MMM](https://github.com/FraunhoferMEVIS/MedicalMultitaskModeling)
- [SPECTRE](https://github.com/cclaess/SPECTRE)
- [RenalCLIP](https://github.com/dt-yuhui/RenalCLIP)

## ⚖️ Licensing

RenalVision's own code and trained classifier heads are licensed Apache-2.0.


Feature extractors are third-party works with their own licenses. RenalVision distributes no third-party weights — they are fetched at runtime from the original providers. The terms of the extractor you choose apply to your use of that pipeline:

| Bundle | Code used | Weights license | Commercial use |
|---|---|---|---|
| radiomics, radiomics_binary | PyRadiomics (BSD-3-Clause) | — | ✅ |
| ctfm | ctfm pip pkg (MIT) | Apache-2.0 | ✅ |
| renalclip | reimplementation | MIT | ✅ |
| fmcib | fmcib pip pkg + adapted snippets (MIT) | CC-BY-4.0 | ✅ with attribution |
| spectre | spectre-fm pip pkg (MIT) | CC-BY-NC-SA-4.0 | ❌ non-commercial |
| mevis, mevis_unicorn | medicalmultitaskmodeling pip pkg (Fraunhofer non-commercial research license) | same | ❌ non-commercial, no redistribution, research institutions only |

> **Note:** This table is provided for orientation only. It reflects our reading of the upstream licenses at the time of writing and carries no  guarantee of accuracy or completeness.
Before relying on any pipeline — especially commercially or clinically — consult the upstream license texts directly and seek your own legal advice.

> RenalVision is research software. It is not CE-marked or FDA-cleared, and no
conformity assessment has been performed for it or for any of the included
feature extractors. The M3 (`mevis`, `mevis_unicorn`) license additionally prohibits the use of any results obtained from that model for diagnostic or therapeutic purposes.
