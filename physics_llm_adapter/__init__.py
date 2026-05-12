"""
physics_llm_adapter

Exposes `PhysicsLLMAdapterV2` + Stage-2 training entry points, plus
`PhysicsLLMAdapterV3`, a subclass of V2 that adds mixed-format training
(Tier 1a from `ADAPTER_GENERALIZATION_PLAN.md`). V2 remains the paper's
canonical checkpoint class; V3 is the follow-up research variant.

Included: `adapter_v2.py`, `adapter_v3.py`, `adapter_heads.py`,
`mcq_head.py`, `train_adapter_v2.py`, `colab_train_adapter.ipynb`,
`language_normalizer.py`, `clevrer_qa_generator.py`.
Excluded: baseline scripts, older adapter variants, and diagnostic
artifacts from the full repo that are not part of the paper pipeline.
"""

from .adapter_v2 import PhysicsLLMAdapterV2, create_adapter_v2
from .adapter_v3 import PhysicsLLMAdapterV3, create_adapter_v3

__all__ = [
    "PhysicsLLMAdapterV2",
    "create_adapter_v2",
    "PhysicsLLMAdapterV3",
    "create_adapter_v3",
]
