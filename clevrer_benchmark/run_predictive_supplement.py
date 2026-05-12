"""Predictive-only supplementary evaluation for the LLM baselines.

The 1K-pool protocol (run_all_baselines.py) yields only ~163 predictive
questions per LLM because predictive is the smallest question type in
CLEVRER's natural distribution (~16% of causal questions). The Wilson 95%
CI half-width at that n is ~6-7 pp, which is the widest CI in fig:ci_forest.

This script re-runs each LLM with --question_types predictive at a higher
max_questions budget so the predictive panel of fig:ci_forest can be
plotted with a tightened CI (target Wilson 95% CI width ~6 pp at n=1000,
i.e. half-width ~3 pp, the default below). Adjust ``--max_questions`` to
target a different CI width (n=1500 -> ~5 pp, n=500 -> ~9 pp).

Outputs are written to results/{key}_with_scene{suffix}_PREDICTIVE.json so
they are unambiguously identifiable as predictive-only supplements (the
existing _FULL.json files retain the natural-distribution 1K-pool primary
results that underwrite the explanatory and counterfactual panels).

Usage::

    # Smoke test (10 predictive questions per LLM):
    python run_predictive_supplement.py --smoke

    # Default: 500 predictive questions per LLM, all providers:
    python run_predictive_supplement.py --all

    # Single provider only:
    python run_predictive_supplement.py --provider together --max_questions 500

The runner is structured to mirror run_all_baselines.py exactly so an
operator already familiar with that script can drive this one without
re-learning the conventions.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ENV_FILE = SCRIPT_DIR.parent.parent / ".env"
RUN_SCRIPT = SCRIPT_DIR / "run_llm_with_scene.py"
RESULTS_DIR = SCRIPT_DIR / "results"

# Mirror run_all_baselines.py exactly so the supplementary run covers every
# LLM that contributes a row to fig:ci_forest.
RUNS = [
    # Together AI models (cheapest, run first)
    {"key": "qwen-7b",      "provider": "together"},
    {"key": "llama-3.3-70b","provider": "together"},
    {"key": "deepseek-v3",  "provider": "together"},
    {"key": "qwen-72b",     "provider": "together"},
    # OpenAI
    {"key": "gpt4",         "provider": "openai"},
    # Anthropic
    {"key": "claude",       "provider": "anthropic"},
    {"key": "claude-4.5",   "provider": "anthropic"},
    # Gemini
    {"key": "gemini",       "provider": "gemini"},
    # No-tools variants (predictive headline gap is largest under the
    # explicit no-tools constraint; we sample these to keep the
    # comparison symmetric with the 1K-pool no-tools rows of fig:ci_forest).
    {"key": "gpt4",         "provider": "openai",     "no_tools": True},
    {"key": "claude",       "provider": "anthropic",  "no_tools": True},
    {"key": "claude-4.5",   "provider": "anthropic",  "no_tools": True},
    {"key": "gemini",       "provider": "gemini",     "no_tools": True},
]


def load_env(env_file: Path) -> dict:
    """Parse a .env file into a dict (matches run_all_baselines.py)."""
    env = {}
    if not env_file.exists():
        return env
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if v:
            env[k] = v
    return env


def run_model(key: str, no_tools: bool, env: dict, max_q: int):
    """Run a single LLM on predictive questions only."""
    suffix = "_notools" if no_tools else ""
    output = RESULTS_DIR / f"{key}_with_scene{suffix}_PREDICTIVE.json"

    if output.exists():
        print(f"  SKIP {key}{suffix} -- {output.name} already exists")
        return True

    cmd = [
        sys.executable, str(RUN_SCRIPT),
        "--model", key,
        "--question_types", "predictive",
        "--max_questions", str(max_q),
        "--output", str(output),
    ]
    if no_tools:
        cmd.append("--no_tools")

    label = f"{key}{suffix} (predictive-only)"
    print(f"\n{'=' * 60}")
    print(f"  STARTING: {label}  (output: {output.name})")
    print(f"{'=' * 60}")

    merged_env = {**os.environ, **(env or {})}
    t0 = time.time()
    result = subprocess.run(cmd, env=merged_env, cwd=str(SCRIPT_DIR.parent))
    elapsed = time.time() - t0

    status = "OK" if result.returncode == 0 else f"FAIL (rc={result.returncode})"
    print(f"  DONE: {label}  [{elapsed/60:.1f} min]  {status}")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Predictive-only supplementary evaluation for LLM baselines."
    )
    parser.add_argument("--provider", type=str, default=None,
                        choices=["together", "openai", "anthropic", "gemini"],
                        help="Run only models for this provider")
    parser.add_argument("--all", action="store_true",
                        help="Run all models sequentially")
    parser.add_argument("--max_questions", type=int, default=1000,
                        help="Max predictive questions per model "
                             "(default: 1000; targets Wilson 95% CI width "
                             "~6 pp, i.e. half-width ~3 pp).")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 10 predictive questions per model")
    args = parser.parse_args()

    if args.smoke:
        args.max_questions = 10
        args.all = True

    if not args.provider and not args.all and not args.smoke:
        parser.error("Specify --provider <name> or --all or --smoke")

    env = load_env(ENV_FILE)
    key_status = {
        "TOGETHER_API_KEY": bool(env.get("TOGETHER_API_KEY")),
        "OPENAI_API_KEY": bool(env.get("OPENAI_API_KEY")),
        "ANTHROPIC_API_KEY": bool(env.get("ANTHROPIC_API_KEY")),
        "GOOGLE_API_KEY": bool(env.get("GOOGLE_API_KEY")),
    }
    print("API keys loaded from .env:")
    for k, v in key_status.items():
        print(f"  {k}: {'SET' if v else 'MISSING'}")

    runs = RUNS
    if args.provider:
        runs = [r for r in runs if r["provider"] == args.provider]

    print(f"\nPredictive-only runs to execute: {len(runs)}")
    print(f"Per-model max_questions: {args.max_questions}")
    print(f"Expected total predictive API calls: ~{len(runs) * args.max_questions}")
    for r in runs:
        nt = " (no-tools)" if r.get("no_tools") else ""
        print(f"  - {r['key']}{nt}")

    RESULTS_DIR.mkdir(exist_ok=True)

    successes, failures = 0, 0
    t_total = time.time()

    for r in runs:
        ok = run_model(r["key"], r.get("no_tools", False), env, args.max_questions)
        if ok:
            successes += 1
        else:
            failures += 1

    elapsed_total = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"ALL DONE: {successes} succeeded, {failures} failed  "
          f"[{elapsed_total/60:.1f} min total]")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
