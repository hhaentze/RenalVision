import numpy as np
import pytest
from monai.data import MetaTensor

from renal_vision.bundles import ImplementedModels
from renal_vision.modeling.inference import LesionPredictor


# Mark as integration so you can exclude it if needed (pytest -m "not integration")
@pytest.mark.integration
class TestEndToEndInference:
    @pytest.fixture(scope="class")
    def real_predictor(self):
        """
        Instantiates the REAL predictor with the binary radiomics model.
        This might take time to load, so we scope it to 'class' to load once.
        """
        return LesionPredictor(model_identifier=ImplementedModels.RADIOMICS_BINARY)

    def test_smoke_infer_mask(self, real_predictor, mock_two_lesions):
        """
        Smoke Test: Does it run without crashing on synthetic data?
        We don't care if the prediction is medically accurate (it's random noise),
        we just care that the pipeline completes.
        """
        img, seg = mock_two_lesions

        # Act
        output_mask = real_predictor.infer_mask(img, seg)

        # Assert
        assert isinstance(output_mask, MetaTensor)
        assert output_mask.shape == seg.shape
        # Ensure it didn't just return NaNs
        assert not output_mask.isnan().any()

    def test_smoke_infer_lesion(self, real_predictor, mock_single_lesion):
        """
        Smoke Test for single lesion inference.
        """
        img, seg = mock_single_lesion
        result = real_predictor.infer_lesion(img, seg)

        assert isinstance(result, dict)
        assert "class_id" in result
        assert "class_name" in result
        assert "confidence" in result
        assert "probability" in result
        assert "volume" in result
        assert isinstance(result["class_id"], (int, np.integer))
