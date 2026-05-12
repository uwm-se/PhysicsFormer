"""Smoke test for the recipe-aware checkpoint validation that prevents the
Phase-4 silent-skip bug (mixed-format training was skipped because a stale
sentinel from a different recipe was loaded without checks).

The validation logic under test lives in the Colab notebook
``checkpoints/colab_train_adapter_v3.ipynb`` (Cell 9, ``_validate_recipe``).
This test re-implements the same logic locally and exercises it against
synthetic checkpoint dicts that cover every reject/accept path so the
notebook patch can be regression-tested without a Colab session.

Run:
    python -m physics_llm_adapter.tests.test_recipe_validation
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict


# --------------------------------------------------------------------------- #
# Local copy of the notebook's _validate_recipe (Cell 9). Kept byte-equivalent
# to the notebook copy so this test can serve as a regression suite if the
# notebook is ever rewritten.
# --------------------------------------------------------------------------- #


def _validate_recipe(ckpt: Dict[str, Any], source_path: Any, expected_recipe: Dict[str, Any]) -> None:
    problems = []
    for key, expected in expected_recipe.items():
        if key not in ckpt:
            problems.append(
                f"  - {key}: missing from checkpoint metadata (expected {expected!r}). "
                f"This sentinel predates the recipe-tagging guard."
            )
            continue
        actual = ckpt[key]
        if isinstance(expected, float) or isinstance(actual, float):
            if actual is None or not math.isclose(float(actual), float(expected), abs_tol=1e-6):
                problems.append(f"  - {key}: checkpoint has {actual!r}, expected {expected!r}")
        else:
            if actual != expected:
                problems.append(f"  - {key}: checkpoint has {actual!r}, expected {expected!r}")

    icp = expected_recipe.get("include_choices_prob")
    if icp is not None and 0.0 < icp < 1.0:
        cum_a = ckpt.get("cumulative_format_a")
        cum_b = ckpt.get("cumulative_format_b")
        if cum_a is not None and cum_b is not None and cum_a == 0 and cum_b == 0:
            problems.append(
                f"  - cumulative_format_a/b are both 0 but include_choices_prob={icp}. "
                f"Mixed-format training never actually ran; this sentinel is degenerate."
            )

    if problems:
        raise RuntimeError(
            f"\n[resume] RECIPE MISMATCH for {source_path}:\n"
            + "\n".join(problems)
            + "\n\nThis would silently load weights trained under a different recipe (or no\n"
            + "training at all). Either delete this file from the checkpoint directory\n"
            + "and re-run, or pass expected_recipe=None to disable this check (NOT recommended)."
        )


# --------------------------------------------------------------------------- #
# Test cases
# --------------------------------------------------------------------------- #

EXPECTED = {
    "adapter_class": "PhysicsLLMAdapterV3",
    "include_choices_prob": 0.5,
    "mixed_format_seed": 42,
}


def _expect_pass(case_name: str, ckpt: Dict[str, Any]) -> None:
    try:
        _validate_recipe(ckpt, Path("synthetic"), EXPECTED)
    except RuntimeError as exc:  # pragma: no cover -- defensive
        raise AssertionError(f"{case_name}: should have passed but raised:\n{exc}") from exc
    print(f"  [PASS] {case_name}")


def _expect_reject(case_name: str, ckpt: Dict[str, Any], expected_substring: str) -> None:
    try:
        _validate_recipe(ckpt, Path("synthetic"), EXPECTED)
    except RuntimeError as exc:
        msg = str(exc)
        if expected_substring not in msg:
            raise AssertionError(
                f"{case_name}: rejected but error message missing "
                f"{expected_substring!r}\nfull error:\n{msg}"
            )
        print(f"  [PASS] {case_name}  (rejected with {expected_substring!r})")
        return
    raise AssertionError(f"{case_name}: should have been rejected but passed")


def test_recipe_match_accepts() -> None:
    """Sentinel with matching recipe + non-degenerate counts is accepted."""
    _expect_pass(
        "recipe_match_accepts",
        {
            "adapter_class": "PhysicsLLMAdapterV3",
            "include_choices_prob": 0.5,
            "mixed_format_seed": 42,
            "cumulative_format_a": 50000,
            "cumulative_format_b": 49500,
        },
    )


def test_recipe_match_with_zero_format_b_is_ok_at_icp_0() -> None:
    """include_choices_prob=0 (V2 mode) does NOT require non-zero counts."""
    expected_v2 = {
        "adapter_class": "PhysicsLLMAdapterV2",
        "include_choices_prob": 0.0,
        "mixed_format_seed": 42,
    }
    try:
        _validate_recipe(
            {
                "adapter_class": "PhysicsLLMAdapterV2",
                "include_choices_prob": 0.0,
                "mixed_format_seed": 42,
                "cumulative_format_a": 0,
                "cumulative_format_b": 0,
            },
            Path("synthetic"),
            expected_v2,
        )
    except RuntimeError as exc:  # pragma: no cover -- defensive
        raise AssertionError(f"V2-mode sentinel was incorrectly rejected:\n{exc}") from exc
    print("  [PASS] icp=0.0 with zero counts (V2 mode) accepts")


def test_wrong_adapter_class_rejects() -> None:
    _expect_reject(
        "wrong_adapter_class_rejects",
        {
            "adapter_class": "PhysicsLLMAdapterV2",
            "include_choices_prob": 0.5,
            "mixed_format_seed": 42,
            "cumulative_format_a": 50000,
            "cumulative_format_b": 49500,
        },
        "adapter_class",
    )


def test_wrong_include_choices_prob_rejects() -> None:
    _expect_reject(
        "wrong_include_choices_prob_rejects",
        {
            "adapter_class": "PhysicsLLMAdapterV3",
            "include_choices_prob": 0.3,
            "mixed_format_seed": 42,
            "cumulative_format_a": 60000,
            "cumulative_format_b": 30000,
        },
        "include_choices_prob",
    )


def test_wrong_mixed_format_seed_rejects() -> None:
    _expect_reject(
        "wrong_mixed_format_seed_rejects",
        {
            "adapter_class": "PhysicsLLMAdapterV3",
            "include_choices_prob": 0.5,
            "mixed_format_seed": 123,
            "cumulative_format_a": 50000,
            "cumulative_format_b": 49500,
        },
        "mixed_format_seed",
    )


def test_missing_metadata_rejects() -> None:
    """The Phase-4 stale sentinel scenario: pre-fix checkpoint with no recipe tag."""
    _expect_reject(
        "missing_metadata_rejects",
        {
            # Plain V2-style save: no adapter_class, no include_choices_prob, etc.
            "phase": 2,
            "epoch": 10,
            "loss": 1.3021,
        },
        "predates the recipe-tagging guard",
    )


def test_degenerate_sentinel_rejects() -> None:
    """Recipe matches but cumulative_format_a==cumulative_format_b==0:
    this is the EXACT failure mode that produced the Phase-4 0% bug."""
    _expect_reject(
        "degenerate_sentinel_rejects",
        {
            "adapter_class": "PhysicsLLMAdapterV3",
            "include_choices_prob": 0.5,
            "mixed_format_seed": 42,
            "cumulative_format_a": 0,
            "cumulative_format_b": 0,
        },
        "cumulative_format_a/b are both 0",
    )


def test_partial_metadata_rejects() -> None:
    """Sentinel has some recipe fields but not all -> reject."""
    _expect_reject(
        "partial_metadata_rejects",
        {
            "adapter_class": "PhysicsLLMAdapterV3",
            # missing include_choices_prob and mixed_format_seed
            "cumulative_format_a": 50000,
            "cumulative_format_b": 49500,
        },
        "include_choices_prob",
    )


def main() -> None:
    print("Recipe validation smoke tests:")
    test_recipe_match_accepts()
    test_recipe_match_with_zero_format_b_is_ok_at_icp_0()
    test_wrong_adapter_class_rejects()
    test_wrong_include_choices_prob_rejects()
    test_wrong_mixed_format_seed_rejects()
    test_missing_metadata_rejects()
    test_degenerate_sentinel_rejects()
    test_partial_metadata_rejects()
    print("\nAll recipe validation smoke tests passed.")


if __name__ == "__main__":
    main()
