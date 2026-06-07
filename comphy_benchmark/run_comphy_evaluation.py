"""ComPhy zero-shot OOD evaluation entry point (Phase 3 adapter, no retraining).

The CLEVRER-trained Phase 3 adapter is loaded as-is and evaluated on ComPhy
validation questions. Same MCQ protocol that produced the 79.6% CLEVRER SOTA
(``--eval_method generate --gen_mode sample --single_frame 64``), so the
ComPhy number is directly comparable to the CLEVRER headline.

Data layout expected on disk
----------------------------
::

    benchmark_data/comphy/
    ├── target_annotation/
    │   ├── annotation_00000_01000/<id>.json   # one per scene, id zero-padded to 5 digits
    │   ├── annotation_01000_02000/...
    │   └── ...
    ├── qa_chunk0.json     # validation chunks (1000 scenes each, 8 chunks)
    ├── qa_chunk1.json
    ├── ...
    ├── qa_chunk7.json
    └── train_qa*.json     # NOT consumed by default (training-set chunks)

Each ``qa_chunk*.json`` is a per-scene list:
``[{scene_index, video_filename, questions: [...]}, ...]``. Question types
mix open-ended factual and 2-choice predictive/counterfactual MC; see
``question_mapper.py`` for the full breakdown.

Usage
-----
::

    python comphy_benchmark/run_comphy_evaluation.py \\
        --adapter_checkpoint checkpoints/adapter_phase3.pt \\
        --output comphy_benchmark/results/phase3_comphy_zeroshot.json \\
        --save_details

``--comphy_dir`` defaults to ``benchmark_data/comphy`` so a fresh checkout
with the data populated runs zero-config.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clevrer_benchmark.run_adapter_evaluation import (  # noqa: E402
    answer_with_adapter,
    evaluate_answer,
    load_adapter_model,
    _contrastive_score_choices,
)
from comphy_benchmark.binary_eval import score_mcq_binary  # noqa: E402
from comphy_benchmark.question_mapper import (  # noqa: E402
    coarsen_comphy_type,
    map_comphy_to_adapter_question,
    native_question_type,
)
from comphy_benchmark.scene_converter import (  # noqa: E402
    comphy_scene_to_state_tensor,
    find_annotation_for_scene,
    load_comphy_annotation,
)


# ───────────────────────────────────────────────────────────────────────────
# Defaults
# ───────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
# Default data path. The ComPhy release (10k annotation JSONs + 8 QA chunks,
# ~2.6 GB) lives **outside** the workspace tree so it doesn't pollute VS Code,
# file searches, or git enumerations. Override with --comphy_dir or set the
# ``COMPHY_DIR`` env var.
DEFAULT_COMPHY_DIR = Path(
    __import__("os").environ.get("COMPHY_DIR", "D:/comphy")
)


# ───────────────────────────────────────────────────────────────────────────
# Result objects
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class ComPhyResult:
    """One result record per question.

    For ``eval_method='generate'`` (default): ``predicted`` is the generated
    string, ``correct`` is a string-match boolean, ``option_correct_count``
    and ``option_total_count`` stay zero (no per-option scoring happened).

    For ``eval_method='binary'`` on MC questions: ``predicted`` is a
    pipe-joined summary of per-option predictions (e.g. ``"yes|no"``),
    ``correct`` is the strict per-question score (all options right),
    and the option-level counts track per-option accuracy for the primary
    ComPhy metric.
    """
    scene_index: int
    question_id: str
    question_text: str
    comphy_type: str
    coarse_type: str
    is_mcq: bool
    ground_truth: Any
    predicted: str
    correct: bool
    choices: Optional[List[Dict[str, Any]]] = None
    question_family: str = ""
    option_correct_count: int = 0
    option_total_count: int = 0
    option_records: Optional[List[Dict[str, Any]]] = None
    # Per-choice physics-on / physics-off log-likelihoods (eval_method
    # 'alpha_sweep' only). Aligned to ``choices`` order. Logged so the alpha
    # axis (and the zero-only / delta-only endpoints) can be reconstructed
    # offline from a single eval pass.
    real_scores: Optional[List[float]] = None
    zero_scores: Optional[List[float]] = None

    def to_detail_dict(self) -> Dict[str, Any]:
        return {
            "scene_index": self.scene_index,
            "question_id": self.question_id,
            "question_text": self.question_text,
            "comphy_type": self.comphy_type,
            "coarse_type": self.coarse_type,
            "is_mcq": self.is_mcq,
            "ground_truth": self.ground_truth,
            "predicted": self.predicted,
            "correct": self.correct,
            "choices": self.choices,
            "question_family": self.question_family,
            "option_correct_count": self.option_correct_count,
            "option_total_count": self.option_total_count,
            "option_records": self.option_records,
            "real_scores": self.real_scores,
            "zero_scores": self.zero_scores,
        }


@dataclass
class ComPhyResults:
    """Aggregator. Tracks per-question accuracy *and* per-option accuracy.

    Per-question (strict) is the primary number for free-form generation
    eval and the strict variant of the binary protocol.
    Per-option is the primary ComPhy paper metric for the binary protocol;
    it's computed from option-level records and is zero for the generate
    method (no option-level decisions there).
    """
    total: int = 0
    correct: int = 0
    by_coarse: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: {"correct": 0, "total": 0})
    )
    by_native: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: {"correct": 0, "total": 0})
    )
    # Per-option totals (only populated under eval_method='binary')
    option_total: int = 0
    option_correct: int = 0
    by_coarse_options: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: {"correct": 0, "total": 0})
    )
    by_native_options: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: {"correct": 0, "total": 0})
    )
    individual: List[ComPhyResult] = field(default_factory=list)

    def add(self, r: ComPhyResult) -> None:
        self.total += 1
        self.correct += int(r.correct)
        self.by_coarse[r.coarse_type]["total"] += 1
        self.by_coarse[r.coarse_type]["correct"] += int(r.correct)
        self.by_native[r.comphy_type]["total"] += 1
        self.by_native[r.comphy_type]["correct"] += int(r.correct)
        if r.option_total_count > 0:
            self.option_total += r.option_total_count
            self.option_correct += r.option_correct_count
            self.by_coarse_options[r.coarse_type]["total"] += r.option_total_count
            self.by_coarse_options[r.coarse_type]["correct"] += r.option_correct_count
            self.by_native_options[r.comphy_type]["total"] += r.option_total_count
            self.by_native_options[r.comphy_type]["correct"] += r.option_correct_count
        self.individual.append(r)

    def to_dict(self) -> Dict[str, Any]:
        def _bucket(d):
            return {
                k: {
                    "total": v["total"],
                    "correct": v["correct"],
                    "accuracy": v["correct"] / max(1, v["total"]),
                }
                for k, v in d.items()
            }
        out = {
            "overall": {
                "total": self.total,
                "correct": self.correct,
                "accuracy": self.correct / max(1, self.total),
            },
            "by_coarse_type": _bucket(self.by_coarse),
            "by_native_type": _bucket(self.by_native),
        }
        if self.option_total > 0:
            out["overall_per_option"] = {
                "total": self.option_total,
                "correct": self.option_correct,
                "accuracy": self.option_correct / max(1, self.option_total),
            }
            out["by_coarse_type_per_option"] = _bucket(self.by_coarse_options)
            out["by_native_type_per_option"] = _bucket(self.by_native_options)
        return out


# ───────────────────────────────────────────────────────────────────────────
# Path resolution
# ───────────────────────────────────────────────────────────────────────────

def _resolve_annotations_dir(comphy_dir: Path,
                             override: Optional[Path]) -> Path:
    if override is not None:
        if not override.exists():
            raise FileNotFoundError(f"--annotations_dir not found: {override}")
        return override
    for candidate in [comphy_dir / "target_annotation",
                      comphy_dir / "annotations"]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find ComPhy annotations under {comphy_dir} "
        f"(expected target_annotation/ or annotations/). "
        f"Pass --annotations_dir to override."
    )


def _resolve_qa_chunks(comphy_dir: Path,
                       override_files: Optional[List[Path]],
                       split: str) -> List[Path]:
    """Return the list of QA-chunk JSON files to iterate.

    Defaults:
      - ``split='validation'`` -> all ``qa_chunk*.json`` (NOT ``train_qa*``)
      - ``split='train'``     -> all ``train_qa_chunk*.json``

    Either override list bypasses auto-detection.
    """
    if override_files:
        for f in override_files:
            if not f.exists():
                raise FileNotFoundError(f"--qa_files entry not found: {f}")
        return list(override_files)

    if split == "train":
        pattern = "train_qa_chunk*.json"
    else:
        # Validation: exclude train_* chunks even though they share the qa_chunk* prefix.
        pattern = "qa_chunk*.json"

    matches = sorted(comphy_dir.glob(pattern))
    if split == "validation":
        matches = [p for p in matches if not p.name.startswith("train_")]
    if not matches:
        raise FileNotFoundError(
            f"No QA chunk files matching {pattern!r} under {comphy_dir}. "
            f"Pass --qa_files explicitly or fix --comphy_dir."
        )
    return matches


# ───────────────────────────────────────────────────────────────────────────
# Question iteration
# ───────────────────────────────────────────────────────────────────────────

def _iter_scenes(qa_files: Iterable[Path]) -> Iterable[Dict[str, Any]]:
    """Yield scene dicts (with .questions) from a sequence of QA chunk files."""
    for qa_file in qa_files:
        with open(qa_file, "r", encoding="utf-8") as f:
            chunk = json.load(f)
        if not isinstance(chunk, list):
            raise ValueError(f"{qa_file}: expected per-scene list, got {type(chunk).__name__}")
        for scene_entry in chunk:
            yield scene_entry


# ───────────────────────────────────────────────────────────────────────────
# Resume: replay the streaming details.jsonl back into ``results`` so a
# killed run can pick up where it left off without re-doing completed
# questions. The details file is the source of truth; the summary JSON is
# always recomputed from it.
# ───────────────────────────────────────────────────────────────────────────

def _replay_details_jsonl(details_path: Path):
    """Read existing per-question records and return ``(results, done_keys)``.

    ``done_keys`` is a set of ``(scene_index, question_id)`` tuples covering
    every record we successfully parsed. The caller uses this set to skip
    already-completed questions in the iteration loop. ``results`` is a
    freshly-populated :class:`ComPhyResults` containing every replayed
    record so the running accumulators (totals, per-coarse, per-native,
    per-option metrics) match what was on disk before the kill.

    Malformed JSONL lines (e.g. a partial write at the tail caused by a
    hard kill) are skipped with a warning and excluded from both
    ``done_keys`` and the rebuilt accumulator.
    """
    results = ComPhyResults()
    done_keys: set = set()
    if not details_path.exists():
        return results, done_keys

    n_lines = 0
    n_skipped = 0
    with open(details_path, "r", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_skipped += 1
                continue
            try:
                r = ComPhyResult(
                    scene_index=int(rec.get("scene_index", -1)),
                    question_id=str(rec.get("question_id", "")),
                    question_text=str(rec.get("question_text", "")),
                    comphy_type=str(rec.get("comphy_type", "")),
                    coarse_type=str(rec.get("coarse_type", "")),
                    is_mcq=bool(rec.get("is_mcq", False)),
                    ground_truth=rec.get("ground_truth"),
                    predicted=str(rec.get("predicted", "")),
                    correct=bool(rec.get("correct", False)),
                    choices=rec.get("choices"),
                    question_family=str(rec.get("question_family", "")),
                    option_correct_count=int(rec.get("option_correct_count", 0) or 0),
                    option_total_count=int(rec.get("option_total_count", 0) or 0),
                    option_records=rec.get("option_records"),
                )
            except (TypeError, ValueError):
                n_skipped += 1
                continue
            results.add(r)
            done_keys.add((r.scene_index, r.question_id))

    print(f"[resume] replayed {n_lines} lines from {details_path.name}, "
          f"{len(done_keys)} unique completed questions"
          + (f" ({n_skipped} skipped malformed)" if n_skipped else ""))
    return results, done_keys


# ───────────────────────────────────────────────────────────────────────────
# Atomic progress writer (mirrors CLEVRER runner)
# ───────────────────────────────────────────────────────────────────────────

def _save_progress(output_path: Path, results: ComPhyResults) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results.to_dict(), f, indent=2)
    tmp.replace(output_path)


# ───────────────────────────────────────────────────────────────────────────
# Main eval loop
# ───────────────────────────────────────────────────────────────────────────

def run_evaluation(
    comphy_dir: Path,
    adapter_checkpoint: str,
    *,
    physics_checkpoint: str = "",
    split: str = "validation",
    qa_files: Optional[List[Path]] = None,
    annotations_dir: Optional[Path] = None,
    max_scenes: Optional[int] = None,
    max_questions: Optional[int] = None,
    device: str = "cuda",
    single_frame: int = 64,
    clevrer_align: bool = False,
    zero_physics: bool = False,
    eval_method: str = "generate",
    gen_mode: str = "sample",
    gen_seed: Optional[int] = 42,
    contrastive_alpha: float = 1.0,
    only_coarse_types: Optional[List[str]] = None,
    skip_factual: bool = False,
    filter_malformed: bool = True,
    shuffle_choices: bool = False,
    output_path: Optional[Path] = None,
    save_details: bool = False,
    save_every: int = 25,
    resume: bool = True,
) -> ComPhyResults:
    annotations_dir = _resolve_annotations_dir(comphy_dir, annotations_dir)
    qa_files = _resolve_qa_chunks(comphy_dir, qa_files, split)
    print(f"[data] annotations: {annotations_dir}")
    print(f"[data] qa_files:    {[str(p) for p in qa_files]}")

    # Resume support: replay any existing details file BEFORE loading the
    # adapter so we can short-circuit out if absolutely nothing is left to do
    # (saves the 5-10s adapter load on a fully-finished run).
    results = ComPhyResults()
    done_keys: set = set()
    details_path = (Path(output_path).with_suffix(".details.jsonl")
                    if (save_details and output_path is not None) else None)
    resuming = (resume and details_path is not None and details_path.exists())
    if resuming:
        results, done_keys = _replay_details_jsonl(details_path)

    print("Loading adapter (CLEVRER-trained Phase 3, zero-shot transfer)...")
    adapter = load_adapter_model(adapter_checkpoint, physics_checkpoint, device)

    if gen_seed is not None:
        torch.manual_seed(gen_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(gen_seed)
        random.seed(gen_seed)
        print(f"[seed] torch.manual_seed({gen_seed})")
    print(f"[gen] mode={gen_mode}, method={eval_method}")

    # Append when resuming so we preserve the existing details; otherwise
    # truncate-and-create (or no-op when --save_details was omitted).
    details_handle = None
    if details_path is not None:
        details_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if resuming else "w"
        details_handle = open(details_path, mode, encoding="utf-8")
        print(f"[stream] streaming per-question details to {details_path} "
              f"(mode={mode!r}, resumed={resuming})")
    if output_path is not None:
        print(f"[stream] periodic summary save every {save_every} scenes -> {output_path}")

    scenes_seen = 0
    scenes_skipped_done = 0
    questions_skipped_done = 0
    annotation_missing = 0

    for scene_entry in _iter_scenes(qa_files):
        scene_index = scene_entry.get("scene_index")
        if scene_index is None:
            continue
        if max_scenes is not None and scenes_seen >= max_scenes:
            break

        # Resume fast-path: if every question for this scene is already in
        # done_keys, skip the annotation load entirely. Important for a
        # >10k-record run that's been killed and restarted: re-loading 7k
        # scene JSONs just to discover their questions are all done would
        # waste minutes on disk I/O.
        if done_keys:
            scene_qs = scene_entry.get("questions", []) or []
            scene_q_keys = {
                (int(scene_index), str(q.get("question_id", f"s{scene_index}_q{i}")))
                for i, q in enumerate(scene_qs)
            }
            if scene_q_keys and scene_q_keys.issubset(done_keys):
                scenes_seen += 1
                scenes_skipped_done += 1
                continue

        ann_path = find_annotation_for_scene(annotations_dir, scene_index)
        if ann_path is None:
            annotation_missing += 1
            scenes_seen += 1
            continue

        try:
            ann = load_comphy_annotation(ann_path)
            states, masks, _meta = comphy_scene_to_state_tensor(
                ann, clevrer_align=clevrer_align)
            if single_frame is not None and single_frame >= 0:
                f_idx = min(single_frame, states.shape[0] - 1)
                states = states[f_idx:f_idx + 1]
                masks = masks[f_idx:f_idx + 1] if masks.ndim == 2 else masks
        except Exception as e:
            print(f"\n[err] scene {scene_index} ({ann_path.name}): {e}")
            scenes_seen += 1
            continue

        for q_idx, question in enumerate(scene_entry.get("questions", [])):
            qid = str(question.get("question_id", f"s{scene_index}_q{q_idx}"))

            # Resume: skip this question if it's already in the replayed set.
            # We check BEFORE mapping/filtering so the skip count is honest
            # (we don't want to "skip" a question that wouldn't have been
            # evaluated this run anyway, e.g. a factual when --skip_factual).
            if done_keys and (int(scene_index), qid) in done_keys:
                questions_skipped_done += 1
                continue

            q_data = map_comphy_to_adapter_question(question)
            coarse = q_data["coarse_type"]

            if skip_factual and coarse == "factual":
                continue
            if only_coarse_types and coarse not in only_coarse_types:
                continue

            choices = q_data.get("choices") or []
            if filter_malformed and choices:
                has_correct = any(
                    isinstance(c, dict) and c.get("answer") == "correct"
                    for c in choices
                )
                if not has_correct:
                    continue

            try:

                # Binary protocol: independent yes/no per choice. MC only;
                # factual still goes through the generate path because it
                # has no choices to score.
                use_binary = (eval_method == "binary"
                              and bool(q_data.get("is_mcq"))
                              and (q_data.get("choices") or []))
                use_alpha_sweep = (eval_method == "alpha_sweep"
                                   and bool(q_data.get("is_mcq"))
                                   and (q_data.get("choices") or []))

                if use_alpha_sweep:
                    # Single-pass instrument: score every choice with physics
                    # ON and OFF, log both, and decide live at the reference
                    # alpha (contrastive_alpha). The full alpha axis -- plus
                    # the zero-only (physics-blind) and delta-only (pure
                    # physics-uplift) endpoints -- is reconstructed offline by
                    # scripts/analyze_alpha_sweep.py from the logged scores.
                    raw_choices = q_data.get("choices") or []
                    choice_texts = [
                        c.get("choice", c) if isinstance(c, dict) else str(c)
                        for c in raw_choices
                    ]
                    states_tensor = torch.from_numpy(states).float().unsqueeze(0).to(device)
                    masks_2d = masks[0] if masks.ndim == 2 else masks
                    masks_tensor = torch.from_numpy(masks_2d).float().unsqueeze(0).to(device)
                    combined, real_s, zero_s = _contrastive_score_choices(
                        adapter, states_tensor, masks_tensor,
                        q_data["question_text"], choice_texts,
                        alpha=contrastive_alpha, device=device,
                        return_components=True,
                    )
                    real_list = [float(x) for x in real_s[0].tolist()]
                    zero_list = [float(x) for x in zero_s[0].tolist()]
                    pred_idx = int(combined[0].argmax().item())
                    predicted = str(choice_texts[pred_idx]).strip().lower()
                    correct = evaluate_answer(
                        predicted, q_data["ground_truth"], raw_choices)
                    r = ComPhyResult(
                        scene_index=int(scene_index),
                        question_id=qid,
                        question_text=q_data["question_text"],
                        comphy_type=q_data["comphy_type"],
                        coarse_type=coarse,
                        is_mcq=True,
                        ground_truth=q_data["ground_truth"],
                        predicted=predicted,
                        correct=correct,
                        choices=raw_choices,
                        question_family=q_data.get("question_family", ""),
                        real_scores=real_list,
                        zero_scores=zero_list,
                    )
                elif use_binary:
                    binary_out = score_mcq_binary(
                        adapter, states, masks,
                        q_data["question_text"], q_data["choices"],
                        device=device, zero_physics=zero_physics,
                    )
                    # Per-question correctness = all options correct (strict).
                    correct = bool(binary_out["all_correct"])
                    # Summarize per-option decisions for the predicted field
                    # (audit; not used for accuracy).
                    predicted_summary = "|".join(
                        ("yes" if o["predicted"] is True
                         else "no" if o["predicted"] is False
                         else "?")
                        for o in binary_out["options"]
                    )
                    r = ComPhyResult(
                        scene_index=int(scene_index),
                        question_id=qid,
                        question_text=q_data["question_text"],
                        comphy_type=q_data["comphy_type"],
                        coarse_type=coarse,
                        is_mcq=True,
                        ground_truth=q_data["ground_truth"],
                        predicted=predicted_summary,
                        correct=correct,
                        choices=q_data.get("choices") or None,
                        question_family=q_data.get("question_family", ""),
                        option_correct_count=int(binary_out["n_correct_options"]),
                        option_total_count=int(binary_out["n_options"]),
                        option_records=binary_out["options"],
                    )
                else:
                    predicted = answer_with_adapter(
                        adapter, states, masks, q_data, device,
                        zero_physics=zero_physics,
                        eval_method=("generate" if eval_method == "binary" else eval_method),
                        gen_mode=gen_mode,
                        contrastive_alpha=contrastive_alpha,
                        shuffle_choices=shuffle_choices,
                    )
                    correct = evaluate_answer(
                        predicted, q_data["ground_truth"], q_data.get("choices"))
                    r = ComPhyResult(
                        scene_index=int(scene_index),
                        question_id=qid,
                        question_text=q_data["question_text"],
                        comphy_type=q_data["comphy_type"],
                        coarse_type=coarse,
                        is_mcq=bool(q_data.get("is_mcq", False)),
                        ground_truth=q_data["ground_truth"],
                        predicted=predicted,
                        correct=correct,
                        choices=q_data.get("choices") or None,
                        question_family=q_data.get("question_family", ""),
                    )

                results.add(r)
                if details_handle is not None:
                    details_handle.write(
                        json.dumps(r.to_detail_dict(), ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"\n[err] scene {scene_index} Q{q_idx}: {e}")
                continue

        scenes_seen += 1

        if max_questions and results.total >= max_questions:
            print(f"\n[cap] {results.total} questions answered, stopping.")
            break

        if scenes_seen % 10 == 0:
            acc = (results.correct / max(1, results.total)) * 100
            print(f"\rProgress: {scenes_seen} scenes, "
                  f"{results.total} questions, "
                  f"Acc: {acc:.1f}%", end="", flush=True)

        if output_path is not None and scenes_seen % save_every == 0:
            _save_progress(output_path, results)
            if details_handle is not None:
                details_handle.flush()

    print()
    if scenes_skipped_done or questions_skipped_done:
        print(f"[resume] skipped {scenes_skipped_done} fully-done scenes "
              f"and {questions_skipped_done} already-done questions on this run")
    if annotation_missing:
        print(f"[warn] {annotation_missing} scenes had no annotation file "
              f"(check --annotations_dir)")
    if details_handle is not None:
        details_handle.flush()
        details_handle.close()
    if output_path is not None:
        _save_progress(output_path, results)
    return results


def print_results(results: ComPhyResults) -> None:
    print("\n" + "=" * 64)
    print("COMPHY ZERO-SHOT OOD - Physics-LLM Adapter (Phase 3)")
    print("=" * 64)
    d = results.to_dict()
    o = d["overall"]
    print(f"\nPer-question (strict, all options correct):")
    print(f"  Overall: {o['accuracy']:.1%} ({o['correct']}/{o['total']})")
    print("  By coarse type:")
    for k, v in sorted(d["by_coarse_type"].items()):
        print(f"    {k:18s}: {v['accuracy']:.1%} ({v['correct']}/{v['total']})")
    print("  By native ComPhy type:")
    for k, v in sorted(d["by_native_type"].items()):
        print(f"    {k:32s}: {v['accuracy']:.1%} ({v['correct']}/{v['total']})")

    if "overall_per_option" in d:
        po = d["overall_per_option"]
        print(f"\nPer-option (binary protocol primary metric):")
        print(f"  Overall: {po['accuracy']:.1%} ({po['correct']}/{po['total']})")
        print("  By coarse type:")
        for k, v in sorted(d["by_coarse_type_per_option"].items()):
            print(f"    {k:18s}: {v['accuracy']:.1%} ({v['correct']}/{v['total']})")
        print("  By native ComPhy type:")
        for k, v in sorted(d["by_native_type_per_option"].items()):
            print(f"    {k:32s}: {v['accuracy']:.1%} ({v['correct']}/{v['total']})")
    print("=" * 64)


def main() -> None:
    p = argparse.ArgumentParser(description="ComPhy zero-shot OOD eval for Physics-LLM adapter")
    p.add_argument("--comphy_dir", type=str, default=str(DEFAULT_COMPHY_DIR),
                   help=f"Root of ComPhy data. Default: {DEFAULT_COMPHY_DIR}")
    p.add_argument("--split", type=str, default="validation",
                   choices=["validation", "train"],
                   help="Which QA chunk family to evaluate (validation = qa_chunk*.json).")
    p.add_argument("--qa_files", nargs="+", type=str, default=None,
                   help="Override: explicit list of QA chunk JSONs to iterate.")
    p.add_argument("--annotations_dir", type=str, default=None,
                   help="Override: annotation directory (default: <comphy_dir>/target_annotation).")
    p.add_argument("--adapter_checkpoint", type=str,
                   default="checkpoints/adapter_phase3.pt")
    p.add_argument("--physics_checkpoint", type=str, default="",
                   help="Ignored: the adapter checkpoint is self-contained.")
    p.add_argument("--max_scenes", type=int, default=None,
                   help="Cap scenes (validation set is 8000 across 8 chunks).")
    p.add_argument("--max_questions", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--single_frame", type=int, default=64,
                   help="Match the CLEVRER SOTA protocol. Pass -1 for all frames.")
    p.add_argument("--clevrer_align", action="store_true",
                   help="Map ComPhy states onto the CLEVRER training distribution: "
                        "mass->{1,2}; friction/restitution->CLEVRER material defaults; "
                        "radius->0.3; quaternion->identity; angular velocity->0. "
                        "Position and velocity are kept (verified in-range vs the "
                        "training h5). Fixes the OOD inputs surfaced by "
                        "scripts/check_state_distribution.py.")
    p.add_argument("--zero_physics", action="store_true")
    p.add_argument("--eval_method", type=str, default="generate",
                   choices=["generate", "contrastive", "binary", "alpha_sweep"],
                   help="MC evaluation. 'generate' (default): free-form generation + "
                        "substring-match against choice text (matches CLEVRER SOTA "
                        "protocol; brittle on counterfactual due to 'would' modal). "
                        "'binary' (canonical ComPhy protocol, Chen et al. 2022): each "
                        "choice asked independently as yes/no, scored per-option AND "
                        "per-question (strict). Factual open-ended questions always "
                        "use 'generate' (no choices to score). 'contrastive' is the "
                        "CLEVRER Jan-26 Contrastive Decoding recipe. 'alpha_sweep' "
                        "runs the contrastive scorer but logs the physics-on / "
                        "physics-off log-likelihoods per choice so the full alpha "
                        "axis (incl. zero-only and pure-delta endpoints) can be "
                        "reconstructed offline via scripts/analyze_alpha_sweep.py.")
    p.add_argument("--contrastive_alpha", type=float, default=1.0)
    p.add_argument("--gen_mode", type=str, default="sample",
                   choices=["sample", "greedy", "beam4", "beam8"])
    p.add_argument("--gen_seed", type=int, default=42)
    p.add_argument("--shuffle_choices", action="store_true")
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated coarse types to keep "
                        "(factual,predictive,counterfactual). Default: all.")
    p.add_argument("--skip_factual", action="store_true",
                   help="Skip the open-ended factual questions (focus on MC types, "
                        "matches the CLEVRER --skip_descriptive convention).")
    filt = p.add_mutually_exclusive_group()
    filt.add_argument("--filter_malformed", dest="filter_malformed", action="store_true")
    filt.add_argument("--no_filter_malformed", dest="filter_malformed", action="store_false")
    p.set_defaults(filter_malformed=True)
    p.add_argument("--save_details", action="store_true")
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--no_resume", action="store_true",
                   help="Disable resume. Default behavior: when --output is set and "
                        "the .details.jsonl sidecar exists, replay it and skip "
                        "already-completed (scene_index, question_id) pairs. "
                        "Pass --no_resume to force a fresh run (truncates the sidecar).")
    args = p.parse_args()

    only_coarse = None
    if args.only:
        only_coarse = [s.strip().lower() for s in args.only.split(",") if s.strip()]

    qa_files = ([Path(x) for x in args.qa_files] if args.qa_files else None)
    annotations_dir = Path(args.annotations_dir) if args.annotations_dir else None
    output_path = Path(args.output) if args.output else None

    results = run_evaluation(
        comphy_dir=Path(args.comphy_dir),
        adapter_checkpoint=args.adapter_checkpoint,
        physics_checkpoint=args.physics_checkpoint,
        split=args.split,
        qa_files=qa_files,
        annotations_dir=annotations_dir,
        max_scenes=args.max_scenes,
        max_questions=args.max_questions,
        device=args.device,
        single_frame=args.single_frame,
        clevrer_align=args.clevrer_align,
        zero_physics=args.zero_physics,
        eval_method=args.eval_method,
        gen_mode=args.gen_mode,
        gen_seed=args.gen_seed,
        contrastive_alpha=args.contrastive_alpha,
        only_coarse_types=only_coarse,
        skip_factual=args.skip_factual,
        filter_malformed=args.filter_malformed,
        shuffle_choices=args.shuffle_choices,
        output_path=output_path,
        save_details=args.save_details,
        save_every=args.save_every,
        resume=(not args.no_resume),
    )
    print_results(results)
    if output_path is not None:
        print(f"\nResults saved to: {output_path}")
        if args.save_details:
            print(f"Per-question details: {output_path.with_suffix('.details.jsonl')} "
                  f"({len(results.individual)} records)")


if __name__ == "__main__":
    main()
