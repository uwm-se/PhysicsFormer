"""ComPhy question -> adapter MCQ input (Phase 3-compatible).

ComPhy QA chunk layout (one record per scene, like CLEVRER)::

    [
      {
        "scene_index": 0,
        "video_filename": "sim_00000.mp4",
        "questions": [
          # Factual (open-ended) -- NO 'question_type' key, has 'answer'
          {"question": "Are there any light moving gray objects ...",
           "answer": "yes",
           "question_family": "object_mass_exist", "question_id": 0,
           "program": [...]},
          # Predictive MC -- two-choice with one 'correct' and one 'wrong'
          {"question_type": "predictive_multiple_choice",
           "question": "Which event will happen next?",
           "choices": [{"choice": "...", "answer": "correct"|"wrong", "choice_id": 0}, ...],
           "question_id": 5,
           "program": [...]},
          # Counterfactual MC -- same shape as predictive MC
          {"question_type": "counterfactual_multiple_choice", ...},
        ]
      },
      ...
    ]

Distribution (chunk0 = 1000 scenes):
    factual (no question_type):              2926  (53%)
    counterfactual_multiple_choice:          2000  (37%)
    predictive_multiple_choice:               550  (10%)

We map all three into the same `q_data` shape the CLEVRER adapter runner
expects (``question_text``, ``ground_truth``, ``choices``), so the existing
``answer_with_adapter`` + ``evaluate_answer`` pipeline can score them
unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# Native ComPhy type strings (kept verbatim for stats grouping).
NATIVE_FACTUAL = "factual"
NATIVE_PREDICTIVE_MC = "predictive_multiple_choice"
NATIVE_COUNTERFACTUAL_MC = "counterfactual_multiple_choice"


def native_question_type(question: Dict[str, Any]) -> str:
    """Resolve a question's native ComPhy type.

    Factual questions in the released chunks have no ``question_type`` key
    at all; MC questions carry it explicitly. We surface a synthesized
    ``"factual"`` string for the no-type case so downstream stats grouping
    is uniform.
    """
    qt = question.get("question_type")
    if qt is None or qt == "":
        return NATIVE_FACTUAL
    return str(qt)


def coarsen_comphy_type(native: str) -> str:
    """Collapse native types into the 3 article-facing categories."""
    s = (native or "").strip().lower()
    if s.startswith("predictive"):
        return "predictive"
    if s.startswith("counterfactual"):
        return "counterfactual"
    if s.startswith("factual") or s == "":
        return "factual"
    return s


def is_multiple_choice(question: Dict[str, Any]) -> bool:
    """ComPhy: factual = open-ended (no choices); MC = predictive/counterfactual."""
    choices = question.get("choices")
    return isinstance(choices, list) and len(choices) > 0


def map_comphy_to_adapter_question(
    question: Dict[str, Any],
) -> Dict[str, Any]:
    """Wrap a single ComPhy question record in the shape ``answer_with_adapter`` consumes.

    For MC questions:
      - ``choices`` is the original list (with ``answer: correct|wrong``).
      - ``ground_truth`` is the text of the first ``correct`` choice.
    For factual (open-ended) questions:
      - ``choices`` is ``[]`` (signals open-ended path in the runner).
      - ``ground_truth`` is the literal ``answer`` field, lowercased.
    """
    q_text = question.get("question", "")
    native = native_question_type(question)
    coarse = coarsen_comphy_type(native)

    if is_multiple_choice(question):
        choices = question["choices"]
        correct_choices = [
            (c.get("choice") or "").strip().lower()
            for c in choices
            if isinstance(c, dict) and c.get("answer") == "correct"
        ]
        ground_truth = correct_choices[0] if correct_choices else ""
        return {
            "question_text": q_text,
            "question_type": native,
            "comphy_type": native,
            "coarse_type": coarse,
            "ground_truth": ground_truth,
            "choices": choices,
            "is_mcq": True,
        }

    # Factual open-ended path.
    answer = question.get("answer", "")
    return {
        "question_text": q_text,
        "question_type": native,
        "comphy_type": native,
        "coarse_type": coarse,
        "ground_truth": str(answer).strip().lower(),
        "choices": [],
        "is_mcq": False,
        "question_family": question.get("question_family", ""),
    }
