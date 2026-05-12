"""
Launch all LLM baseline evaluations on the full CLEVRER validation set.

Reads API keys from ../../.env, runs each model sequentially per provider,
but can run multiple providers in parallel via separate invocations.

Usage:
    python run_all_baselines.py --provider together   # 5 models
    python run_all_baselines.py --provider openai      # 1 model
    python run_all_baselines.py --provider anthropic   # 2 models
    python run_all_baselines.py --provider gemini      # 1 model + no-tools
    python run_all_baselines.py --all                  # everything sequentially
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

# All models to run, grouped by provider
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
    # No-tools variants
    {"key": "gpt4",         "provider": "openai",    "no_tools": True},
    {"key": "claude",       "provider": "anthropic",  "no_tools": True},
    {"key": "claude-4.5",   "provider": "anthropic",  "no_tools": True},
    {"key": "gemini",       "provider": "gemini",     "no_tools": True},
]


def load_env(env_file: Path) -> dict:
    """Parse a .env file into a dict."""
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


def run_model(key: str, no_tools: bool = False, env: dict = None, max_q: int = 99999):
    """Run a single model benchmark."""
    suffix = "_notools" if no_tools else ""
    output = RESULTS_DIR / f"{key}_with_scene{suffix}_FULL.json"

    if output.exists():
        print(f"  SKIP {key}{suffix} — {output.name} already exists")
        return True

    cmd = [
        sys.executable, str(RUN_SCRIPT),
        "--model", key,
        "--max_questions", str(max_q),
        "--output", str(output),
    ]
    if no_tools:
        cmd.append("--no_tools")

    label = f"{key}{suffix}"
    print(f"\n{'='*60}")
    print(f"  STARTING: {label}  (output: {output.name})")
    print(f"{'='*60}")

    merged_env = {**os.environ, **(env or {})}
    t0 = time.time()
    result = subprocess.run(cmd, env=merged_env, cwd=str(SCRIPT_DIR.parent))
    elapsed = time.time() - t0

    status = "OK" if result.returncode == 0 else f"FAIL (rc={result.returncode})"
    print(f"  DONE: {label}  [{elapsed/60:.1f} min]  {status}")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=str, default=None,
                        choices=["together", "openai", "anthropic", "gemini"],
                        help="Run only models for this provider")
    parser.add_argument("--all", action="store_true", help="Run all models sequentially")
    parser.add_argument("--max_questions", type=int, default=99999,
                        help="Max questions per model (default: all)")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 10 questions per model")
    args = parser.parse_args()

    if args.smoke:
        args.max_questions = 10
        args.all = True

    if not args.provider and not args.all and not args.smoke:
        parser.error("Specify --provider <name> or --all or --smoke")

    # Load .env
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

    # Filter runs
    runs = RUNS
    if args.provider:
        runs = [r for r in runs if r["provider"] == args.provider]

    print(f"\nModels to run: {len(runs)}")
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
    print(f"\n{'='*60}")
    print(f"ALL DONE: {successes} succeeded, {failures} failed  [{elapsed_total/60:.1f} min total]")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
