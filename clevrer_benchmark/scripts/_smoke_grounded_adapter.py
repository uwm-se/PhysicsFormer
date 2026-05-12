"""Smoke test: load grounded_physics_adapter and generate on 5 heldout CLEVRER questions.

Goal: confirm the model loads, accepts our state tensor format, and
produces non-degenerate output (i.e., not 100% "unknown") before
investing in a full n=1998 paper-quality eval.

This is a temporary diagnostic, not a final eval script. It lives in
clevrer_benchmark/scripts/ alongside free_form_transfer_test.py for
discoverability; remove or fold into a proper eval script once we know
whether the grounded adapter is worth scaling up.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

# Project paths -- mirror the layout used by zero_shot_test.py.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # repo root
_GROUNDED_DIR = _PROJECT_ROOT / 'grounded_physics_adapter'
_PHYSICS_TOKENIZER_DIR = _PROJECT_ROOT / 'physics_tokenizer'
_CLEVRER_BENCHMARK_DIR = _PROJECT_ROOT / 'clevrer_benchmark'

# load_grounded_adapter does its own sys.path manipulation; just need the
# entry point on sys.path.
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_GROUNDED_DIR))
sys.path.insert(0, str(_PHYSICS_TOKENIZER_DIR))
sys.path.insert(0, str(_CLEVRER_BENCHMARK_DIR))

from grounded_physics_adapter.adapter import load_grounded_adapter  # noqa: E402
from scene_converter import clevrer_scene_to_state_tensor  # noqa: E402


CAUSAL_ENCODER_PATH = _PROJECT_ROOT / 'physics_llm_adapter' / 'causal_encoder_best.pt'
TOKENIZER_PATH = _PROJECT_ROOT / 'physics_tokenizer' / 'physics_tokenizer_finetuned.pt'
ADAPTER_PATH = _GROUNDED_DIR / 'grounded_adapter_best.pt'

CLEVRER_QUESTIONS = Path(r'$CLEVRER_DIR\questions\clevrer_validation.json')
CLEVRER_SCENES_DIR = Path(r'$CLEVRER_DIR\scenes\clevrer_scenes')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _load_scene_state(scene_index: int):
    """Load CLEVRER scene JSON -> (states, masks) ready for the adapter."""
    scene_path = CLEVRER_SCENES_DIR / f'annotation_{scene_index:05d}.json'
    if not scene_path.exists():
        return None, None
    with open(scene_path, 'r') as f:
        scene = json.load(f)
    states, masks, _ = clevrer_scene_to_state_tensor(scene)
    # states: [T, N, 35] numpy; masks: [T, N] numpy
    states_t = torch.from_numpy(states).float().unsqueeze(0).to(DEVICE)  # [1, T, N, 35]
    masks_t = torch.from_numpy(masks).float().unsqueeze(0).to(DEVICE)    # [1, T, N]
    return states_t, masks_t


def main():
    print(f'[smoke] device: {DEVICE}')

    # 1. Load adapter ------------------------------------------------
    print(f'[smoke] loading grounded adapter ...')
    adapter = load_grounded_adapter(
        causal_encoder_path=str(CAUSAL_ENCODER_PATH),
        tokenizer_path=str(TOKENIZER_PATH),
        adapter_checkpoint=str(ADAPTER_PATH),
        device=DEVICE,
    )
    adapter.eval()
    n_params = sum(p.numel() for p in adapter.parameters())
    n_train = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f'[smoke] params: {n_params:,} total, {n_train:,} trainable')

    # 2. Sample 5 heldout-style questions ----------------------------
    print(f'[smoke] loading CLEVRER validation questions ...')
    with open(CLEVRER_QUESTIONS, 'r') as f:
        clevrer = json.load(f)
    print(f'[smoke] {len(clevrer)} scenes in validation pool')

    # Pick 5 non-descriptive questions across the first few scenes.
    samples = []
    for scene_entry in clevrer:
        if len(samples) >= 5:
            break
        scene_idx = scene_entry['scene_index']
        states, masks = _load_scene_state(scene_idx)
        if states is None:
            continue
        for q in scene_entry.get('questions', []):
            if len(samples) >= 5:
                break
            qt = q.get('question_type', '')
            if qt == 'descriptive':
                continue
            choices = q.get('choices', [])
            correct = [c['choice'] for c in choices if c.get('answer') == 'correct']
            if not correct:
                continue
            samples.append({
                'scene_idx': scene_idx,
                'question_type': qt,
                'question': q['question'],
                'correct_choices': correct,
                'states': states,
                'masks': masks,
            })

    print(f'[smoke] sampled {len(samples)} questions\n')

    # 3. Generate -- BOTH free-form (no Options) AND MCQ (with Options) -------
    for i, s in enumerate(samples, 1):
        ff_prompt = f"Question: {s['question']}\nAnswer:"
        with torch.no_grad():
            ff_out = adapter.generate(
                physics_states=s['states'],
                object_mask=s['masks'],
                prompt=ff_prompt,
                max_new_tokens=40,
                temperature=0.7,
            )
        print(f'[{i}] scene={s["scene_idx"]} type={s["question_type"]}')
        print(f'    Q: {s["question"]}')
        print(f'    correct(s): {s["correct_choices"][:2]}')
        print(f'    FREE-FORM out: {ff_out!r}')
        print()


if __name__ == '__main__':
    main()
