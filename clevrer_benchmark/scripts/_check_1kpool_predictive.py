"""Quick diagnostic: dump 1K-pool natural-distribution predictive subset for each LLM
(without supplement override). Used to disambiguate prose numbers from supplement-tightened
CI plot values during the 4/28 supplement integration.
"""
import json
from pathlib import Path

RES = Path(__file__).parent.parent / "results"

FILES = [
    ("Qwen-7B", "qwen-7b_with_scene_FULL.json"),
    ("Qwen3-235B", "qwen-72b_with_scene_FULL.json"),
    ("Llama-70B", "llama-3.3-70b_with_scene_FULL.json"),
    ("DeepSeek-V3", "deepseek-v3_with_scene_FULL.json"),
    ("GPT-4o", "gpt4_with_scene_FULL.json"),
    ("Claude Sonnet 4", "claude_with_scene_FULL.json"),
    ("Claude 4.5 Sonnet", "claude-4.5_with_scene_FULL.json"),
    ("Gemini 2.0 Flash", "gemini_with_scene_FULL.json"),
    ("GPT-4o no-tools", "gpt4_with_scene_notools_FULL.json"),
    ("Claude Sonnet 4 no-tools", "claude_with_scene_notools_FULL.json"),
    ("Claude 4.5 Sonnet no-tools", "claude-4.5_with_scene_notools_FULL.json"),
    ("Gemini 2.0 Flash no-tools", "gemini_with_scene_notools_FULL.json"),
]

print(f"{'Model':<32} 1K-pool pred (n=163)    | Supplement (PREDICTIVE.json)")
print("-" * 90)
for name, fname in FILES:
    path = RES / fname
    d = json.loads(path.read_text())
    pred = d.get("by_type", {}).get("predictive", {})
    n1k = pred.get("total", 0)
    c1k = pred.get("correct", 0)
    acc1k = (c1k / n1k * 100) if n1k else 0.0

    sup_path = RES / fname.replace("_FULL.json", "_PREDICTIVE.json")
    if sup_path.exists():
        ds = json.loads(sup_path.read_text())
        bts = ds.get("by_type", ds.get("by_clevrer_type", {}))
        ps = bts.get("predictive", {})
        ns = ps.get("total", 0)
        cs = ps.get("correct", 0)
        accs = (cs / ns * 100) if ns else 0.0
        sup_str = f"n={ns}, acc={accs:.1f}%"
    else:
        sup_str = "(no supplement)"

    print(f"{name:<32} n={n1k}, acc={acc1k:.1f}%       | {sup_str}")
