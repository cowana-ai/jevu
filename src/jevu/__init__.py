"""jevu -- erase a target concept from text embeddings.

Define a concept in plain English, label it zero-shot with JEV (or bring your own labels),
and remove its linear signal from off-the-shelf embeddings with INLP or LEACE.
"""
from .audit import concept_auc, erasure_report, tpr_gap
from .embedders import OpenAIEmbedder
from .erasers import InlpEraser, LeaceEraser
from .labelers import JevLabeler
from .scrubber import ConceptScrubber

__version__ = "0.1.0"
__all__ = [
    "ConceptScrubber",
    "InlpEraser",
    "LeaceEraser",
    "JevLabeler",
    "OpenAIEmbedder",
    "concept_auc",
    "erasure_report",
    "tpr_gap",
    "__version__",
]
