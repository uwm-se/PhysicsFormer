"""
Phase 9 canonical train/eval splits.

This module is the SINGLE SOURCE OF TRUTH for the splits used by:
  - synthetic-QA generation (Lever 1, ``generate_phase9_diverse_qa.py``)
  - paraphrase generation (Lever 2, ``generate_phase9_paraphrased_qa.py``)
  - the three held-out eval scripts (Lever 4, ``eval_phase9_*``)
  - the Phase 9 training notebook itself

Any code that filters CLEVRER scenes, picks question types, or partitions
records by paraphrase tier MUST import its boundaries from here so the
training set and the eval set stay disjoint by construction. Edits to
this file are research-significant and must be reviewed before any
Phase 9 training run.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple

# ---------------------------------------------------------------------
# Scene splits
# ---------------------------------------------------------------------
# CLEVRER's validation set spans scene indices 10000-14999 (5000 scenes).
# Phases 3-8 used a 90/10 cut: scenes 10000-14498 for training,
# 14499-14999 for evaluation (the canonical 501-scene held-out subset
# at ``compsac_2026_code/clevrer_benchmark/results/heldout_scenes.json``).
# Phase 9 keeps the same partition byte-identically -- mixing the
# scene boundary now would make Phase 9 results non-comparable with the
# Phase 3-8 baselines and would also contaminate the 501-scene held-out
# eval that the paper reports against.
TRAIN_SCENE_RANGE: Tuple[int, int] = (10000, 14499)  # half-open [start, end)
HELDOUT_SCENE_RANGE: Tuple[int, int] = (14499, 15000)  # half-open

HELDOUT_SCENES_JSON = (
    Path(__file__).resolve().parent.parent
    / 'clevrer_benchmark' / 'results' / 'heldout_scenes.json'
)


def is_train_scene(scene_index: int) -> bool:
    """Return True iff scene_index falls in the Phase 9 training range."""
    lo, hi = TRAIN_SCENE_RANGE
    return lo <= int(scene_index) < hi


def is_heldout_scene(scene_index: int) -> bool:
    """Return True iff scene_index is in the canonical 501-scene held-out set."""
    lo, hi = HELDOUT_SCENE_RANGE
    return lo <= int(scene_index) < hi


def load_heldout_scene_set() -> FrozenSet[int]:
    """Return the canonical held-out scene IDs from the committed sidecar.

    Falls back to a range-based set if the sidecar is missing (e.g. fresh
    checkout without any past results). The sidecar wins when present
    because it is the exact set Phase 3-7 evals were graded against.
    """
    if HELDOUT_SCENES_JSON.exists():
        with open(HELDOUT_SCENES_JSON, 'r') as f:
            return frozenset(int(s) for s in json.load(f))
    lo, hi = HELDOUT_SCENE_RANGE
    return frozenset(range(lo, hi))


# ---------------------------------------------------------------------
# Question-type splits (Lever 4 eval cut #1: held-out-type)
# ---------------------------------------------------------------------
# Five PhysicsQAGenerator question types are reserved for evaluation
# only. Phase 9 training MUST exclude these types from the synthetic
# diverse pool so we can measure whether the model generalizes from the
# 25-ish types it did see to types it never saw during training.
#
# Selection criteria for the held-out five:
#   1. Cover distinct cognitive skills (numerical regression vs.
#      categorical classification vs. boolean prediction).
#   2. Easy to score automatically (single answer, no
#      free-form narrative).
#   3. NOT structurally derivable from another type by simple template
#      substitution (e.g. we keep OBJECT_VELOCITY in train but reserve
#      KINETIC_ENERGY because energy depends on both velocity and mass
#      -- harder to fake from velocity alone).
#   4. Distinct surface-form vocabularies so a paraphrase-style
#      generalization isn't enough to crack them.
#
# Strings here are the ``QuestionType.value`` of the generator enum
# (frozen here as plain strings so this module has no torch / generator
# import at top level -- avoids a circular-import surface in tests).
HELDOUT_QUESTION_TYPES: FrozenSet[str] = frozenset({
    'kinetic_energy',        # numerical, requires velocity AND mass
    'collision_prediction',  # boolean, requires future-state reasoning
    'speed_comparison',      # relational, picks a specific object
    'mass_comparison',       # relational, picks a specific object
    'time_to_event',         # numerical, predictive
})


# Subset of generator types we will INCLUDE in the diverse training
# pool. Picked to:
#   - cover the four PhysicsQAGenerator categories that work on
#     CLEVRER tensors out of the box (basic properties, physical
#     quantities, predictive, relational)
#   - SKIP categories 5 (evaluative/safety) and 6 (intentional/agency)
#     because their answers depend on free-floating concepts ("safe",
#     "agent", "cooperating") that a 4-object CLEVRER scene cannot
#     ground -- including them would teach the LoRA to hallucinate.
#   - SKIP all metaphor types -- not relevant to physics QA and
#     introduces narrative templates that would re-create exactly the
#     overfitting failure mode Phase 9 is designed to escape.
#
# Held-out types (above) are explicitly removed below so the training
# pool is closed under Phase 9 invariants.
TRAIN_QUESTION_TYPES_ALL: Tuple[str, ...] = (
    # Category 1: Basic Physical Properties
    'object_count',
    'object_position',
    'object_velocity',
    'object_mass',
    'motion_direction',
    # Category 2: Physical Quantities (held-out kinetic/mass/speed types
    # filtered below)
    'total_momentum',
    'relative_velocity',
    'spatial_distance',
    # Category 3: Predictive Reasoning (held-out collision/time types
    # filtered below)
    'trajectory_extrapolation',
    'reachability',
    'path_obstruction',
    # Category 4: Relational Reasoning
    'proximity',
    'spatial_containment',
    'relative_position',
    'contact_state',
)

TRAIN_QUESTION_TYPES: Tuple[str, ...] = tuple(
    qt for qt in TRAIN_QUESTION_TYPES_ALL if qt not in HELDOUT_QUESTION_TYPES
)


def is_train_question_type(qa_type: str) -> bool:
    """Return True iff this qa_type is in the Phase 9 train allowlist."""
    return str(qa_type).lower() in set(TRAIN_QUESTION_TYPES)


def is_heldout_question_type(qa_type: str) -> bool:
    """Return True iff this qa_type is reserved for held-out evaluation."""
    return str(qa_type).lower() in HELDOUT_QUESTION_TYPES


# ---------------------------------------------------------------------
# Paraphrase splits (Lever 4 eval cut #2: held-out-paraphrase)
# ---------------------------------------------------------------------
# The paraphrase audit at
# ``compsac_2026_code/clevrer_benchmark/scripts/paraphrase_audit.py``
# emits four tiers of question rewriting:
#   t0 = canonical CLEVRER phrasing (same as training)
#   t1 = mild paraphrase (synonym swap, voice change)
#   t2 = structural paraphrase (clause reorder, embedded question)
#   t3 = aggressive paraphrase (full rewrite, may shift register)
#
# Phase 9 trains on TIER 0 + Lever 2 GPT-4-paraphrased records of training
# scenes (which sit between t0 and t1 in distribution). Eval grades on
# TIERS 1-3 of the held-out scenes, which the model has never seen in
# any form. This cleanly separates "the model learned to handle one
# alternative phrasing" from "the model can handle arbitrary
# phrasings".
TRAIN_PARAPHRASE_TIERS: FrozenSet[str] = frozenset({'t0'})
HELDOUT_PARAPHRASE_TIERS: FrozenSet[str] = frozenset({'t1', 't2', 't3'})


# ---------------------------------------------------------------------
# Mix ratios for the Phase 9 training pool
# ---------------------------------------------------------------------
# These are the realised fractions in the final training set, NOT the
# fractions of source files. The dataloader concatenates the three
# pools and shuffles; sample-without-replacement is applied if any
# pool's natural size doesn't match the target ratio.
#
# Rationale:
#   - 50% diverse synthetic -- breaks the question-template prior
#     (the core lever; 25 question types vs Phase 7/8's 3 means no
#     single template can be memorised as ``the`` answer template).
#   - 30% existing CLEVRER causal QA -- keeps the model fluent in
#     CLEVRER's surface form; without this the Phase 7/8 baselines
#     stop being comparable.
#   - 20% GPT-4-paraphrased CLEVRER causal QA -- forces invariance to
#     question phrasing for the same scene/answer pair.
PHASE9_MIX_RATIOS: Dict[str, float] = {
    'diverse_synthetic': 0.50,
    'clevrer_canonical': 0.30,
    'clevrer_paraphrased': 0.20,
}
assert abs(sum(PHASE9_MIX_RATIOS.values()) - 1.0) < 1e-9, (
    'Phase 9 mix ratios must sum to 1.0'
)


# ---------------------------------------------------------------------
# Canonical artifact paths
# ---------------------------------------------------------------------
# Where Phase 9 generators write their outputs. These paths are the
# arguments the training notebook expects on Drive:
#   /content/drive/MyDrive/physics_llm/data/phase9/
# but the local-disk paths used by the generators / unit tests live
# under ``compsac_2026_code/data/phase9/``. Kept relative to the
# snapshot root so Colab and local stay in sync.
PHASE9_DATA_DIR_LOCAL = Path(__file__).resolve().parent.parent / 'data' / 'phase9'
PHASE9_DIVERSE_QA_FILE = PHASE9_DATA_DIR_LOCAL / 'physics_diverse_qa_dataset.json'
PHASE9_PARAPHRASED_QA_FILE = PHASE9_DATA_DIR_LOCAL / 'causal_qa_dataset_paraphrased.json'
PHASE9_TRAIN_SCENE_CACHE = PHASE9_DATA_DIR_LOCAL / 'phase9_scene_cache.pt'


# ---------------------------------------------------------------------
# Convenience: human-readable split summary, useful for log preambles.
# ---------------------------------------------------------------------
def describe() -> str:
    """One-line summary of the splits, for logging and the training banner."""
    return (
        f'Phase 9 splits: '
        f'train_scenes=[{TRAIN_SCENE_RANGE[0]}, {TRAIN_SCENE_RANGE[1]}) '
        f'({TRAIN_SCENE_RANGE[1] - TRAIN_SCENE_RANGE[0]} scenes), '
        f'heldout_scenes=[{HELDOUT_SCENE_RANGE[0]}, {HELDOUT_SCENE_RANGE[1]}) '
        f'({HELDOUT_SCENE_RANGE[1] - HELDOUT_SCENE_RANGE[0]} scenes); '
        f'train_qtypes={len(TRAIN_QUESTION_TYPES)}, '
        f'heldout_qtypes={len(HELDOUT_QUESTION_TYPES)}; '
        f'train_paraphrase_tiers={sorted(TRAIN_PARAPHRASE_TIERS)}, '
        f'heldout_paraphrase_tiers={sorted(HELDOUT_PARAPHRASE_TIERS)}; '
        f'mix={PHASE9_MIX_RATIOS}'
    )


if __name__ == '__main__':
    # Standalone smoke test: ``python phase9_splits.py``.
    print(describe())
    print()
    print('TRAIN question types:')
    for qt in TRAIN_QUESTION_TYPES:
        print(f'  - {qt}')
    print()
    print('HELDOUT question types (eval-only):')
    for qt in sorted(HELDOUT_QUESTION_TYPES):
        print(f'  - {qt}')
    print()
    heldout = load_heldout_scene_set()
    print(f'Heldout scene set: n={len(heldout)}, '
          f'range [{min(heldout)}, {max(heldout)}]')
    print(f'is_train_scene(10000)  -> {is_train_scene(10000)}')
    print(f'is_train_scene(14499)  -> {is_train_scene(14499)}')
    print(f'is_train_scene(14999)  -> {is_train_scene(14999)}')
    print(f'is_heldout_scene(14999)-> {is_heldout_scene(14999)}')
