"""Per-qa_type answer-space prompt cues, demos, and constrained-decoding masks.

The Phase 9 held-out-type eval revealed three output-format failure modes:

  1. **Yes-bias** on ``collision_prediction`` (model emits ``Yes`` ~95%
     of the time regardless of physics signal).
  2. **Vocabulary leakage** on ``speed_comparison`` (model emits
     kinetic-energy buckets like ``high`` instead of speed buckets like
     ``very fast``).
  3. **Schema unseen** on ``time_to_event`` (model emits gibberish like
     ``Centre`` because the answer vocabulary
     ``{imminent, soon, late, never}`` was never observed at training
     time).

All three failures share one root cause: the eval prompt
``f"{scene_text} {question} Answer:"`` provides no information about
what the valid answer space is. The Phase 9 base LLM (Qwen2.5-1.5B)
falls back to its pretraining priors, which generate plausible English
but not the categorical bucket the gold answer expects.

This module provides three layers of prompt-side intervention:

  - **Layer 1 (cue)**: ``ANSWER_SPACE_CUES`` -- a one-line hint per
    held-out qa_type using the EXACT bucket vocabulary the answer
    function in ``qa_generator.py`` emits. Insert via ``build_prompt``
    with ``answer_cue=True``. Zero-training-cost; ~30 second change.

  - **Layer 2 (in-context demo)**: ``IN_CONTEXT_DEMOS`` -- one canned
    Q/A demonstration per qa_type. Hand-crafted, NEVER drawn from any
    eval scene to avoid leakage. Insert via ``build_prompt`` with
    ``in_context_shots=1``. Zero-training-cost; complementary to L1.

  - **Layer 3 (constrained decoding)**: ``HeldoutAnswerSpaceProcessor``
    -- a ``transformers.LogitsProcessor`` that masks all but the
    allowed first tokens at the first generation step. Force-commits
    the model to a valid bucket; especially effective on ``time_to_event``
    (gold vocab fully outside the model's prior). Pass via
    ``adapter.forward(..., logits_processor=...)``. Zero-training-cost.

All three layers share the same vocabulary contract anchored in
``categorical_match.HELDOUT_TYPE_BUCKETS`` (which the lenient rescore
already uses for held-out-type scoring). Edits to bucket vocab MUST
update both files in lockstep.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# ``categorical_match`` is the source of truth for held-out bucket vocab;
# we re-use it here so the cues and the lenient scoring agree.
from .categorical_match import HELDOUT_TYPE_BUCKETS


# ---------------------------------------------------------------------------
# Layer 1: per-qa_type answer-space cue
# ---------------------------------------------------------------------------
# Phrased as parenthetical hints, identical in spirit to the demonstration
# cues used by ARC, BIG-Bench, and MMLU evaluators. Wording is deliberately
# terse so it costs ~10-20 prompt tokens (Qwen2.5 BPE) while still naming
# every legal bucket. The exact word casing matches the gold-answer
# generator in qa_generator.py (see _answer_kinetic_energy_calc,
# _answer_max_speed, _answer_time_to_collision, etc.).
# ---------------------------------------------------------------------------
# Phase 10 (Lever 4 Fix 4): cues for the TRAIN qa_types so the LoRA
# learns to attend to answer-space hints during training. Without this
# the eval-time cue is OOD prompt territory and HURTS accuracy
# (Fix 1 alone = -2pp on held-out types). Phase 10 retraining injects
# these into a configurable fraction of training prompts so the model
# treats cued-and-uncued prompts as in-distribution. Vocabulary verified
# against qa_generator.py answer-fn return values.
TRAINED_TYPE_CUES: Dict[str, str] = {
    # _answer_counting -> str(count). CLEVRER scenes have at most 8
    # objects so the integer space is bounded.
    'object_count': (
        'Answer with an integer between 0 and 8.'
    ),
    # _answer_moving_count -> str(count). Same range as object_count.
    'object_velocity': (
        'Answer with an integer between 0 and 8.'
    ),
    # _answer_heaviest_object_num returns the bare 1-based index ('1',
    # '2', ...) so the cue says "answer with a number 1-8". The held-
    # out mass cue uses the more explicit "object index: 1, 2, 3, ..."
    # phrasing -- both substring-match the bare digit gold so cross-
    # regime vocab transfer works either way.
    'object_mass': (
        'Answer with a number from 1 to 8.'
    ),
    # _answer_largest_position -> position descriptor like 'top-left',
    # 'right', 'center'. Free-form structured -- list the canonical
    # axes so the model has a vocabulary anchor.
    'object_position': (
        'Choose one: top-left, top, top-right, left, center, right, '
        'bottom-left, bottom, bottom-right.'
    ),
    # _answer_direction_count -> str(count). The QUESTION mentions a
    # direction ("How many objects are moving rightward?") but the
    # ANSWER is the count. Cue must reflect the integer answer space.
    'motion_direction': (
        'Answer with an integer between 0 and 8.'
    ),
    # _answer_momentum_calc -> negligible/very low/low/moderate/high/
    # very high. Same bucket family as held-out kinetic_energy so the
    # cue text is identical -- key transfer channel: Phase 10 trains
    # the model to associate this cue text with picking from these
    # buckets, and the eval-time KE cue then lands in-distribution.
    'total_momentum': (
        'Choose one: negligible, very low, low, moderate, high, very high.'
    ),
    # _answer_relative_motion -> 'approaching' / 'moving apart' / 'stationary'.
    'relative_velocity': (
        'Choose one: approaching, moving apart, stationary.'
    ),
    # _answer_min_distance_calc -> touching/very close/close/moderate/
    # far/very far (verified against qa_generator.py:991-1002).
    'spatial_distance': (
        'Choose one: touching, very close, close, moderate, far, very far.'
    ),
    # _answer_trajectory_prediction -> position description ('upper
    # left', etc.). Free-form, intentionally omitted from this dict.

    # _answer_can_reach_target / _answer_is_path_blocked: yes/no.
    'reachability': (
        'Answer Yes or No.'
    ),
    'path_obstruction': (
        'Answer Yes or No.'
    ),
    # _answer_objects_in_range -> str(count_in_range). Integer answer.
    'spatial_containment': (
        'Answer with an integer between 0 and 8.'
    ),
    # _answer_nearest_object -> 'Object N' (1-based). Index branch.
    'proximity': (
        'Answer with an object index: 1, 2, 3, ...'
    ),
    # _answer_largest_position (used by RELATIVE_POSITION too) ->
    # position descriptor. Free-form, intentionally omitted.

    # _answer_is_grounded -> 'yes' / 'no' (verified against
    # qa_generator.py:1159-1162).
    'contact_state': (
        'Answer Yes or No.'
    ),
}


ANSWER_SPACE_CUES: Dict[str, str] = {
    # _answer_kinetic_energy_calc: 6 buckets including 'negligible'.
    'kinetic_energy': (
        'Choose one: negligible, very low, low, moderate, high, very high.'
    ),
    # _answer_collision returns "Yes"/"No"; _answer_will_collide_soon
    # returns "yes"/"no". Cue uses the canonical capitalization since
    # both casings substring-match either way.
    'collision_prediction': (
        'Answer Yes or No.'
    ),
    # _answer_max_speed bucket branch (stationary..very fast) OR index
    # branch (Object N). Cue surfaces both since the question template
    # selects one or the other at generation time.
    'speed_comparison': (
        'Choose one: stationary, very slow, slow, moderate, fast, very fast '
        '-- or an object index like 1, 2, 3.'
    ),
    # _answer_heaviest_object_num returns "Object N" (1-based) but the
    # eval substring rule accepts bare "N" too.
    'mass_comparison': (
        'Answer with an object index: 1, 2, 3, ...'
    ),
    # _answer_time_to_collision: 6 ordinal buckets (verified against
    # qa_generator.py:1035-1046).
    'time_to_event': (
        'Choose one: imminent, soon, moderate, long, very long, never.'
    ),
}


# ---------------------------------------------------------------------------
# Layer 2: per-qa_type in-context demonstration
# ---------------------------------------------------------------------------
# One canned Q+A pair per held-out type. Question phrasings mirror the
# generator's templates so the model sees a realistic format match;
# answers use the correct bucket vocabulary so the format channel is
# fully grounded by the demonstration.
#
# ANTI-LEAKAGE: Every demo answer is HAND-PICKED to be consistent with
# the qa_type's bucket family but not derived from any specific eval
# scene. The demos are static across the entire eval pool so the
# information they leak is the *answer space*, not any *answer*. The
# n=200 eval pool is sampled from 501 held-out scenes that the model
# has never seen, so the per-question physics signal still has to
# come from the prefix tokens the encoder emits.
IN_CONTEXT_DEMOS: Dict[str, List[Dict[str, str]]] = {
    'kinetic_energy': [{
        'question': 'What is the total kinetic energy of the system?',
        'answer': 'moderate',
    }],
    'collision_prediction': [{
        'question': 'Will any objects collide in the near future?',
        'answer': 'No',
    }],
    'speed_comparison': [{
        'question': 'Which object has the highest speed?',
        'answer': '2',
    }],
    'mass_comparison': [{
        'question': 'Which object has the largest mass?',
        'answer': '1',
    }],
    'time_to_event': [{
        'question': 'When will the closest objects make contact?',
        'answer': 'soon',
    }],
}


# ---------------------------------------------------------------------------
# Layer 1+2: prompt assembly
# ---------------------------------------------------------------------------

@dataclass
class PromptConfig:
    """Per-call config for ``build_prompt``.

    Attributes:
        answer_cue: If True, insert ``(<ANSWER_SPACE_CUES[qa_type]>)``
            between the question and ``Answer:``. False = legacy Phase 9
            behaviour.
        in_context_shots: Number of canned demos from
            ``IN_CONTEXT_DEMOS[qa_type]`` to prepend. 0 = no demos
            (legacy). 1 = first demo. Currently only 0/1 are supported
            because IN_CONTEXT_DEMOS only carries one demo per type.
        answer_marker: Trailing token. Almost always ``"Answer:"`` to
            match training distribution.
    """
    answer_cue: bool = False
    in_context_shots: int = 0
    answer_marker: str = 'Answer:'


def build_prompt(
    qa_type: str,
    scene_text: str,
    question: str,
    cfg: Optional[PromptConfig] = None,
) -> str:
    """Assemble the eval-time prompt string for a held-out qa_type.

    Layered construction (left to right, in the final string):

        [demo_q1] {marker} [demo_a1]\n[demo_q2] {marker} [demo_a2]\n
        {scene_text} {question} ({cue}) {marker}

    Pieces are omitted when their layer is disabled.

    For the BASELINE (cfg.answer_cue=False, cfg.in_context_shots=0)
    this collapses to the legacy ``f"{scene_text} {question} Answer:"``
    string byte-identically -- so passing ``cfg=None`` reproduces Phase
    9's eval prompt.
    """
    cfg = cfg or PromptConfig()
    parts: List[str] = []

    # In-context demos first so the prompt narrative is
    # ``[demos] [scene] [question] [cue] Answer:`` -- the demos read
    # like prior turns of a multi-turn conversation, which Qwen2.5
    # handles naturally because it's an instruction-tuned model.
    if cfg.in_context_shots > 0:
        demos = IN_CONTEXT_DEMOS.get(qa_type, [])
        for d in demos[:cfg.in_context_shots]:
            parts.append(f'{d["question"]} {cfg.answer_marker} {d["answer"]}')

    # Scene + question + optional cue.
    body = f'{scene_text} {question}' if scene_text else question
    if cfg.answer_cue:
        cue = ANSWER_SPACE_CUES.get(qa_type)
        if cue:
            body = f'{body} ({cue})'
    parts.append(f'{body} {cfg.answer_marker}')

    return '\n'.join(parts)


# ---------------------------------------------------------------------------
# Layer 3: constrained-decoding LogitsProcessor
# ---------------------------------------------------------------------------
# Lazy imports for torch / transformers so this module is cheap to
# import in unit tests that don't need the processor.

def _allowed_first_words(qa_type: str) -> List[str]:
    """Return the set of first WORDS the model may emit as the answer.

    Pulls from ``HELDOUT_TYPE_BUCKETS[qa_type]['buckets']`` (NOT the dict
    keys -- that was the bug in the first version). Uses just the first
    whitespace-separated word of each bucket so that multi-word buckets
    like ``very fast`` and ``very low`` share their leading ``very``
    token without the mask listing it twice.

    Index-branch handling for ``mass_comparison`` and the index
    half of ``speed_comparison``:

      - The qa_generator returns ``f"Object {idx + 1}"``, so the FIRST
        emitted token is ``' Object'`` (Qwen2.5 token id 3002). Adding
        ``Object`` to the allowed list covers that branch.
      - The model also emits bare-digit answers like ``" 1"``, which
        Qwen tokenizes as ``[' ', '1']`` -- token id 220 then 16. Token
        220 is just the leading space, so masking on it is uninformative
        (allowing it lets the model emit ANY digit on the second step,
        which is exactly what we want for the index branch). We surface
        a sentinel ``__LEADING_SPACE__`` so ``build_first_token_mask``
        can add token 220 to the allowed mask without listing every
        digit individually.
    """
    qa_type = (qa_type or '').lower()
    spec = HELDOUT_TYPE_BUCKETS.get(qa_type)
    if not spec:
        return []
    raw_buckets = list(spec.get('buckets', []) or [])
    # ordinal_or_index types carry their numeric branch under a separate
    # key; surface it via the leading-space sentinel so the digit branch
    # remains open when the LogitsProcessor is active.
    has_index_branch = (
        qa_type in {'mass_comparison'}
        or spec.get('kind') == 'ordinal_or_index'
    )
    # Modifiers we explicitly DROP from the allowed-first-token set
    # because masking only the first token + leaving them allowed leads
    # to a known failure: the model commits to "very" then completes
    # with "very high" (KE-vocabulary bias from Phase 9 training)
    # regardless of qa_type. That ruins time_to_event ("very long" gold
    # -> "very high" pred) and speed_comparison ("very fast" gold ->
    # "very high" pred). The substring scoring credits "fast" for gold
    # "very fast" and "high" for "very high", so we lose nothing by
    # forcing the model onto a non-modifier bucket-word.
    _MODIFIERS = {'very', 'extremely'}
    words: List[str] = []
    for b in raw_buckets:
        b = (b or '').strip()
        if not b:
            continue
        # Surface every whitespace-separated token of the bucket EXCEPT
        # the modifiers above. For "very fast" this yields ["fast"];
        # for "moderate" -> ["moderate"]; for "very long" -> ["long"].
        for part in b.split(' '):
            if part.lower() in _MODIFIERS:
                continue
            words.append(part)
    if has_index_branch:
        words.append('Object')
        words.append('__LEADING_SPACE__')
    if qa_type == 'collision_prediction':
        for w in ('Yes', 'No', 'yes', 'no'):
            if w not in words:
                words.append(w)
    seen = set()
    out: List[str] = []
    for w in words:
        if w and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _first_token_id(tokenizer, word: str) -> Optional[int]:
    """Return the token id for the first sub-token of ``word`` when it
    appears at the start of an answer (i.e. preceded by a space).

    Different tokenizers handle the leading-space case differently:

      - GPT-2 / Qwen byte-level BPE: ``" word"`` becomes one or more
        tokens beginning with the metaspace ``" "`` byte. Encoding
        ``" word"`` and taking ``[0]`` is correct.
      - SentencePiece (LLaMA): ``"word"`` and ``"_word"`` are different
        tokens; we encode with leading space too.

    Returns ``None`` if the tokenizer fails to produce any tokens for
    the input (shouldn't happen for non-empty ``word``).
    """
    try:
        ids = tokenizer.encode(' ' + word, add_special_tokens=False)
    except Exception:
        ids = []
    if not ids:
        return None
    return int(ids[0])


def build_first_token_mask(
    qa_type: str,
    tokenizer,
) -> Optional['torch.BoolTensor']:
    """Construct a 1D bool tensor of length ``vocab_size`` where True
    marks tokens that are valid as the FIRST generated token for the
    given qa_type. Returns ``None`` if no buckets are registered.
    """
    import torch  # local import keeps the module lean

    words = _allowed_first_words(qa_type)
    if not words:
        return None
    # Qwen2.5 reports tokenizer.vocab_size = 151643 (base BPE vocab) but
    # its eos_token_id is 151645 (added special token). Sizing the mask
    # by tokenizer.vocab_size alone produces an undersized tensor that
    # silently broadcasts incorrectly when LogitsProcessor.__call__
    # tries to mask the model's full-width score tensor. Take the max
    # over every vocab-defining attribute so the mask is at least as
    # wide as any token id we might want to mask.
    candidates = [
        int(getattr(tokenizer, 'vocab_size', 0) or 0),
        int(len(tokenizer)),
    ]
    for attr in ('eos_token_id', 'pad_token_id', 'bos_token_id', 'unk_token_id'):
        v = getattr(tokenizer, attr, None)
        if isinstance(v, int) and v >= 0:
            candidates.append(v + 1)
    vocab_size = max(candidates) if candidates else len(tokenizer)
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    n_set = 0
    for w in words:
        if w == '__LEADING_SPACE__':
            # Allow the bare leading-space token so index-branch answers
            # like ``" 1"`` (tokens [220, 16]) can still fire. The cue +
            # demo channel provides the actual digit guidance.
            try:
                space_ids = tokenizer.encode(' ', add_special_tokens=False)
            except Exception:
                space_ids = []
            if space_ids and 0 <= int(space_ids[0]) < vocab_size:
                mask[int(space_ids[0])] = True
                n_set += 1
            continue
        tid = _first_token_id(tokenizer, w)
        if tid is None or tid >= vocab_size:
            continue
        mask[tid] = True
        n_set += 1
    if n_set == 0:
        return None
    return mask


# Defer the LogitsProcessor base class import so this module imports
# without transformers when the user only wants build_prompt().
def _make_processor_class():
    from transformers import LogitsProcessor

    class HeldoutAnswerSpaceProcessor(LogitsProcessor):
        """Mask all but the allowed first tokens at the FIRST generation step.

        After the first new token is committed, the processor becomes a
        no-op so the model can complete the answer freely (e.g.
        ``"very"`` -> ``"very"`` -> ``"very fast"``). Only the first
        token is constrained because that's where the format failure
        manifests; once the model commits to ``"very"`` the rest of the
        bucket (``" fast"`` / ``" slow"`` / ``" low"`` / ``" high"``) is
        decided by the model's prior + the prefix signal.

        Stateful: maintains an internal counter of how many calls have
        elapsed since construction. Build a fresh instance for every
        ``adapter.forward(...)`` invocation.
        """

        def __init__(self, allowed_mask: 'torch.BoolTensor'):
            super().__init__()
            self._allowed_mask = allowed_mask
            self._n_calls = 0

        def __call__(self, input_ids, scores):
            # First call corresponds to the first new token. With
            # inputs_embeds the prompt isn't reflected in input_ids,
            # so input_ids.shape[1] starts at 0 and grows each call.
            if self._n_calls == 0:
                import torch  # local import so this module imports without torch
                # scores shape: [batch, model_vocab_size]. The mask was
                # sized against tokenizer attributes which may diverge
                # from model.config.vocab_size (Qwen2.5-1.5B has
                # vocab=151936 in the model config, while the tokenizer
                # only reports 151643 + a handful of special tokens).
                # Expand the mask to match scores.shape[-1] by padding
                # with False (disallowed) so any token id beyond our
                # mask is treated as not-an-answer.
                allowed = self._allowed_mask.to(scores.device)
                target_size = scores.shape[-1]
                if allowed.shape[0] < target_size:
                    pad = torch.zeros(
                        target_size - allowed.shape[0],
                        dtype=allowed.dtype,
                        device=allowed.device,
                    )
                    allowed = torch.cat([allowed, pad], dim=0)
                elif allowed.shape[0] > target_size:
                    allowed = allowed[:target_size]
                disallowed = ~allowed
                scores = scores.masked_fill(disallowed, float('-inf'))
            self._n_calls += 1
            return scores

    return HeldoutAnswerSpaceProcessor


_HeldoutAnswerSpaceProcessor_cls = None


def make_constrained_processor(qa_type: str, tokenizer):
    """Factory: build a fresh ``HeldoutAnswerSpaceProcessor`` for one call.

    Returns ``None`` if no buckets are registered for the qa_type or
    the tokenizer can't tokenize any of the allowed first words.
    Callers should check for ``None`` and skip the constrained-decoding
    path in that case.
    """
    global _HeldoutAnswerSpaceProcessor_cls
    mask = build_first_token_mask(qa_type, tokenizer)
    if mask is None:
        return None
    if _HeldoutAnswerSpaceProcessor_cls is None:
        _HeldoutAnswerSpaceProcessor_cls = _make_processor_class()
    return _HeldoutAnswerSpaceProcessor_cls(mask)


__all__ = [
    'ANSWER_SPACE_CUES',
    'IN_CONTEXT_DEMOS',
    'PromptConfig',
    'build_prompt',
    'build_first_token_mask',
    'make_constrained_processor',
]
