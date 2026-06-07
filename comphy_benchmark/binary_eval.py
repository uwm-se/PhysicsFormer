"""Canonical ComPhy per-option binary scoring protocol.

Reproduces the Chen et al. (ICLR 2022) evaluation: each MC choice is
presented as an independent statement and the adapter is asked
"Is this option correct? Answer only yes or no." for each. There is no
string matching against the ComPhy choice text, so the "would collide"
vs "collide" surface-form mismatch that bricks free-form generation
disappears.

Metrics emitted by the runner when this protocol is active:
    - per_option accuracy   (primary)   = #correct yes/no responses / total options
    - per_question accuracy (strict)    = all options for the question must be correct

Helpers in this module are deliberately small and self-contained -- no
dependency on the deleted ``physics_bench`` tree, no dependency on
``clevrer_benchmark`` beyond what the runner already imports.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# ───────────────────────────────────────────────────────────────────────────
# Prompt + parsing
# ───────────────────────────────────────────────────────────────────────────

def build_binary_option_prompt(question_text: str, option_text: str) -> str:
    """Construct the per-option yes/no prompt used by the canonical protocol."""
    return (
        f"{question_text}\n"
        f"Option: {option_text}\n"
        f"Is this option correct? Answer only yes or no."
    )


def parse_yes_no(text: str) -> Optional[bool]:
    """Parse a yes/no answer from free-form generation output.

    Returns True for yes, False for no, None if ambiguous. The parser
    is intentionally lenient -- a CLEVRER-trained adapter will sometimes
    emit phrases like "yes, the cube collides" or "no .", and we want
    those to score as the obvious yes/no rather than fall through to
    None (which the runner treats as a wrong answer).
    """
    if text is None:
        return None
    s = text.strip().lower()
    if not s:
        return None

    if s in ("yes", "yes.", "yes,", "true", "correct"):
        return True
    if s in ("no", "no.", "no,", "false", "incorrect", "wrong"):
        return False

    first = s.split()[0].rstrip(".,!?:") if s.split() else ""
    if first == "yes":
        return True
    if first == "no":
        return False

    has_yes = bool(re.search(r"\byes\b", s))
    has_no = bool(re.search(r"\bno\b", s))
    if has_yes and not has_no:
        return True
    if has_no and not has_yes:
        return False

    return None


# ───────────────────────────────────────────────────────────────────────────
# Per-option scoring against the 35-D adapter
# ───────────────────────────────────────────────────────────────────────────

def score_option(
    adapter,
    states: np.ndarray,
    masks: np.ndarray,
    question_text: str,
    option_text: str,
    *,
    device: str = "cuda",
    zero_physics: bool = False,
    max_length: int = 8,
) -> Tuple[Optional[bool], str]:
    """Greedy-decode the adapter's yes/no answer for one MC option.

    Returns ``(parsed_yes_no_or_None, raw_generated_text)``. ``max_length=8``
    is generous: typical outputs are 1-3 tokens ("yes" / "no" / "no .").
    Greedy decoding (no sampling) gives a deterministic answer per option
    so the per-option accuracy is reproducible without seed tuning.
    """
    states_tensor = torch.from_numpy(states).float().unsqueeze(0).to(device)
    if zero_physics:
        states_tensor = torch.zeros_like(states_tensor)

    masks_2d = masks[0] if masks.ndim == 2 else masks
    masks_tensor = torch.from_numpy(masks_2d).float().unsqueeze(0).to(device)

    prompt = build_binary_option_prompt(question_text, option_text)
    with torch.no_grad():
        answers = adapter.forward(
            physics_states=states_tensor,
            object_mask=masks_tensor,
            question_text=[prompt],
            max_length=max_length,
            do_sample=False,
            num_beams=1,
        )
    raw = answers[0] if isinstance(answers, list) and answers else str(answers)
    raw = str(raw).split("\n")[0].strip()
    return parse_yes_no(raw), raw


def score_mcq_binary(
    adapter,
    states: np.ndarray,
    masks: np.ndarray,
    question_text: str,
    choices: List[Dict[str, Any]],
    *,
    device: str = "cuda",
    zero_physics: bool = False,
) -> Dict[str, Any]:
    """Score every choice of a single MC question with the binary protocol.

    Returns a dict::

        {
            "options": [
                {"choice_id": int, "option_text": str,
                 "ground_truth": bool,        # True iff answer == "correct"
                 "predicted": bool|None,      # parsed yes/no; None = ambiguous
                 "raw": str,                  # raw model output for audit
                 "correct": bool},
                ...
            ],
            "all_correct": bool,             # strict per-question
            "n_correct_options": int,
            "n_options": int,
        }

    A choice with ``predicted=None`` (the parser couldn't decide) is treated
    as wrong, matching the upstream protocol.
    """
    options = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        option_text = (choice.get("choice") or "").strip()
        gt_label = (choice.get("answer") or "").lower() == "correct"
        parsed, raw = score_option(
            adapter, states, masks, question_text, option_text,
            device=device, zero_physics=zero_physics,
        )
        predicted = bool(parsed) if parsed is not None else False
        is_correct = (predicted == gt_label)
        options.append({
            "choice_id": choice.get("choice_id"),
            "option_text": option_text,
            "ground_truth": gt_label,
            "predicted": parsed,
            "raw": raw,
            "correct": is_correct,
        })

    n = len(options)
    nc = sum(1 for o in options if o["correct"])
    return {
        "options": options,
        "all_correct": (n > 0 and nc == n),
        "n_correct_options": nc,
        "n_options": n,
    }
