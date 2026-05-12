"""Reusable scoring helpers shared across CLEVRER eval & audit scripts.

Until this module existed, scoring logic was duplicated byte-for-byte across
``paraphrase_audit.py``, ``semantic_equivalence_audit.py``,
``free_form_transfer_test.py``, ``paraphrased_mcq_test.py``, and
``free_form_prefix_ablation.py``. Each duplication was a maintenance liability:
a fix to ``_substring_correct`` in one script would silently miss the four
other copies, producing inconsistent strict/lenient numbers across the paper.

The submodules here are organised by *what kind of agreement* they measure:

  - ``text_match``      : surface-form scoring (substring, verbatim choice,
                         CLEVRER-template detection, paraphrase-fallback
                         detection). No model dependency.
  - ``nli_paraphrase``  : bidirectional-entailment NLI scoring with the
                         flip-rule used by the paraphrase audit. Imports
                         transformers lazily so ``--help`` stays fast.
  - ``referent_equiv``  : scene-aware referent-set equivalence (the rule
                         behind ``semantic_equivalence_audit``). Resolves
                         CLEVRER MCQ event templates against the scene's
                         object inventory.
  - ``categorical_match``: ordinal-bucket synonym match for the Phase 9
                         held-out QUESTION TYPES (kinetic_energy,
                         collision_prediction, mass_comparison,
                         speed_comparison, time_to_event). The substring
                         scorer alone misses paraphrases like
                         ``"medium"`` <-> ``"moderate"`` or
                         ``"the third object"`` <-> ``"3"``.
  - ``io``              : Phase-9 eval-JSON <-> legacy details.jsonl
                         schema conversion so the legacy audits run
                         unchanged on the new outputs.

Every helper has the same behaviour as the pre-refactor copy in its
original script. The refactored callers are regression-tested in
``clevrer_benchmark/scripts/scoring_regression_test.py``.
"""

from __future__ import annotations

# Re-export the most common helpers at the package level so consumers can
# do ``from clevrer_benchmark.scoring import substring_correct`` instead
# of having to remember which submodule each lives in.
from .text_match import (  # noqa: F401
    norm,
    substring_correct,
    verbatim_choice_match,
    looks_clevrer_template,
    detects_original_template_fallback,
    bucket_wrong_pred,
)
from .nli_paraphrase import (  # noqa: F401
    nli_setup,
    paraphrase_score,
    nli_correct,
    run_nli_batched,
    build_pairs,
    evaluate_flips,
)
from .referent_equiv import (  # noqa: F401
    parse_descriptor,
    parse_event,
    matches_scene_object,
    referent_set,
    is_semantic_match,
    load_scene_objects,
    scene_id_to_num,
    scene_object_inventory,
)
from .categorical_match import (  # noqa: F401
    HELDOUT_TYPE_BUCKETS,
    categorical_correct,
    extract_object_index,
)
from .answer_space import (  # noqa: F401
    ANSWER_SPACE_CUES,
    TRAINED_TYPE_CUES,
    IN_CONTEXT_DEMOS,
    PromptConfig,
    build_prompt,
    build_first_token_mask,
    make_constrained_processor,
)

__all__ = [
    'norm',
    'substring_correct',
    'verbatim_choice_match',
    'looks_clevrer_template',
    'detects_original_template_fallback',
    'bucket_wrong_pred',
    'nli_setup',
    'paraphrase_score',
    'nli_correct',
    'run_nli_batched',
    'build_pairs',
    'evaluate_flips',
    'parse_descriptor',
    'parse_event',
    'matches_scene_object',
    'referent_set',
    'is_semantic_match',
    'load_scene_objects',
    'scene_id_to_num',
    'scene_object_inventory',
    'HELDOUT_TYPE_BUCKETS',
    'categorical_correct',
    'extract_object_index',
    'ANSWER_SPACE_CUES',
    'TRAINED_TYPE_CUES',
    'IN_CONTEXT_DEMOS',
    'PromptConfig',
    'build_prompt',
    'build_first_token_mask',
    'make_constrained_processor',
]
