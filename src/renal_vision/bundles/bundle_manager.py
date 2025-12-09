import pickle
from difflib import get_close_matches
from enum import Enum
from importlib import resources
from typing import Optional, Type, TypeVar

from renal_vision.modeling.models import ModelBundle

T = TypeVar("T", bound=Enum)


class ImplementedModels(str, Enum):
    TUMOR_CYST = "tumor_cyst"
    HISTOLOGY_SUBTYPE = "histology_subtype"


def load_model_bundle(model_identifier: ImplementedModels) -> ModelBundle:
    if isinstance(model_identifier, str):
        try:
            model_identifier = ImplementedModels(model_identifier)
        except ValueError:
            raise ValueError(
                f"Unknown model identifier: '{model_identifier}'. "
                f"Valid options: {[m.value for m in ImplementedModels]}"
            )

    try:
        pkg_files = resources.files(__package__)
        model_name = model_identifier.value
        model_path = pkg_files.joinpath(model_name, "model.pkl")

        if not model_path.is_file():
            raise FileNotFoundError(f"Model file '{model_name}' missing from package.")

        with model_path.open("rb") as f:
            return pickle.load(f)

    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_identifier}: {e}")


def suggest_similar_enum(name: str, enum_cls: Type[T], cutoff: float = 0.6) -> Optional[T]:
    """
    Suggest the closest matching enum member (case-insensitive, fuzzy).
    Returns the enum member or None.
    """
    # Build a lowercase lookup
    lookup = {member.name.lower(): member for member in enum_cls}

    # Perform fuzzy matching on lowercase names
    matches = get_close_matches(name.lower(), list(lookup.keys()), n=1, cutoff=cutoff)

    return lookup[matches[0]] if matches else None
