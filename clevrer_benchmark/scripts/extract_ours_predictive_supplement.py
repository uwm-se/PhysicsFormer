"""Extract Ours' predictive supplement from the existing FULL5000 measurement.

Grounded-Physics LM was already evaluated on every CLEVRER validation question
(``phase3_BASELINE_SHUFFLE_FULL5000.json``, n=21,378 across 5,000 scenes). The
predictive subset of that file is n=3,557 -- ~19x larger than the 1K-pool
predictive subset (n=185) that is used in the Predictive panel of
``fig:ci_forest`` by default.

This script copies the predictive subset of FULL5000 into a new file using
the supplement naming convention recognised by ``_compute_1k_cis.py``::

    phase3_BASELINE_SHUFFLE_FULL5000.json       (input, already exists)
        -> by_clevrer_type.predictive          (n=3557)

    adapter_phase3_1k_sampled_PREDICTIVE.json   (output written by this script)
        -> by_clevrer_type.predictive          (n=3557, copied verbatim)
        -> overall                              (mirrors predictive for callers
                                                 that read overall directly)

After running this script, ``plot_ci_forest.py`` automatically uses the
larger n=3557 measurement for the Ours row of the Predictive panel without
any other code changes (the loader auto-detects the supplement file).

Usage::

    python compsac_2026_code/clevrer_benchmark/scripts/extract_ours_predictive_supplement.py

The script is idempotent: re-running it overwrites the output deterministically.
No CLEVRER data, model weights, or API access are required.
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SOURCE = RESULTS_DIR / "phase3_BASELINE_SHUFFLE_FULL5000.json"
OUTPUT = RESULTS_DIR / "adapter_phase3_1k_sampled_PREDICTIVE.json"


def main():
    if not SOURCE.exists():
        raise SystemExit(
            f"FULL5000 source not found: {SOURCE}\n"
            "This file is produced by the canonical Phase-3 evaluation; see "
            "REPRODUCTION.md for how to regenerate it from the SOTA checkpoint."
        )

    with open(SOURCE) as f:
        full = json.load(f)

    bt = full.get("by_clevrer_type") or full.get("by_type") or {}
    pred = bt.get("predictive")
    if not pred or pred.get("total", 0) == 0:
        raise SystemExit(
            f"No predictive entries found in {SOURCE.name}; check the file "
            "format or regenerate the FULL5000 evaluation."
        )

    correct = int(pred["correct"])
    total = int(pred["total"])
    accuracy = correct / total * 100.0

    # Mirror the schema of the LLM _FULL.json files so _compute_1k_cis.py's
    # _read_predictive_supplement() can ingest it without special-casing.
    payload = {
        "source": str(SOURCE.name),
        "note": (
            "Predictive subset extracted from FULL5000 baseline-shuffled "
            "evaluation. Used as the Ours-side supplement for fig:ci_forest "
            "Predictive panel; pairs with run_predictive_supplement.py output "
            "for the LLM rows."
        ),
        "overall": {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
        },
        "by_clevrer_type": {
            "predictive": {
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
            },
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {OUTPUT}")
    print(f"  predictive: correct={correct}, total={total}, "
          f"accuracy={accuracy:.2f}%")
    print(
        "Re-run plot_ci_forest.py to regenerate fig:ci_forest with the "
        "tightened Ours predictive CI."
    )


if __name__ == "__main__":
    main()
