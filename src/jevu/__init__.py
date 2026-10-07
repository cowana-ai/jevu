"""jevu -- erase a target concept from text embeddings.

Define a concept in plain English, label it zero-shot with laya (a local, free, open-weights
calibrated scorer -- or bring your own labels), and remove its linear signal from off-the-shelf
embeddings with INLP or LEACE.
"""
import logging as _logging

from .audit import concept_auc, erasure_report, tpr_gap
from .concepts import LLMConceptLabeler
from .erasers import InlpEraser, LeaceEraser
from .laya_labeler import LayaLabeler
from .scrubber import ConceptScrubber

# Library best practice: attach a NullHandler so importing jevu never emits logs unless the
# application configures logging (e.g. logging.basicConfig(level=logging.INFO)).
_logging.getLogger("jevu").addHandler(_logging.NullHandler())

__version__ = "0.1.0"
__all__ = [
    "ConceptScrubber",
    "InlpEraser",
    "LeaceEraser",
    "LayaLabeler",
    "LLMConceptLabeler",
    "concept_auc",
    "erasure_report",
    "tpr_gap",
    "__version__",
]
