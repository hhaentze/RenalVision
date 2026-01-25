import importlib

import pytest

# Centralized configuration: (name, import_path, class_name, expected_len)
EXTRACTOR_DATA = [
    pytest.param(
        "radiomics", "renal_vision.features.embeddings_radiomics", "RadiomicsExtractor", 88
    ),
    pytest.param("mevis", "renal_vision.features.embeddings_mevis", "MevisExtractor", 1024),
    pytest.param("fmcib", "renal_vision.features.embeddings_fmcib", "FMCIBExtractor", 4096),
    pytest.param("ctfm", "renal_vision.features.embeddings_ctfm", "CTFMExtractor", 512),
]


def get_extractor(module_path, class_name, **kwargs):
    """Dynamically imports and instantiates the extractor."""
    module = importlib.import_module(module_path)
    extractor_class = getattr(module, class_name)
    return extractor_class(**kwargs)


class TestFeatureExtraction:
    @pytest.mark.local_only
    @pytest.mark.integration
    @pytest.mark.parametrize("is_augmented", [True, False])
    @pytest.mark.parametrize(
        "name, mod_path, cls_name, expected_emb_len",
        EXTRACTOR_DATA,
        ids=[item[0][0] for item in EXTRACTOR_DATA],
    )
    def test_end_to_end_extraction(
        self, name, mod_path, cls_name, expected_emb_len, is_augmented, mock_single_lesion
    ):
        # 1. Setup
        image, seg = mock_single_lesion
        extractor = get_extractor(mod_path, cls_name, min_volume=10)

        # 2. Execution
        embeddings = extractor.extract(image, seg, augment=is_augmented)

        # 3. Validation
        assert len(embeddings) == 1, f"{name} should return exactly one embedding for one lesion"

        result = embeddings[0]
        assert result["augmented"] == is_augmented

        # The +5 represents metadata fields (ID, label, etc.)
        expected_total_len = expected_emb_len + 5
        actual_total_len = len(result)

        assert actual_total_len == expected_total_len, (
            f"Feature length mismatch for {name}. "
            f"Expected {expected_total_len} (inc. metadata), got {actual_total_len}"
        )
