"""
15-Object CLEVRER Benchmark for Physics-LLM

Uses ACTUAL CLEVRER questions with scenes augmented to 15 objects,
matching the same benchmark format used for LLM comparison.

The adapter receives state tensors [B, T, N, 35] rather than text descriptions.

Usage:
    python benchmark_15_objects_physics_llm.py --adapter_checkpoint path/to/adapter.pt --num_questions 90
"""

import os
import argparse
import json
import random
import sys
import zipfile
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import scene converter utilities
from clevrer_benchmark.scene_converter import (
    SHAPE_MAP, COLOR_MAP, MATERIAL_MAP, DEFAULT_SIZE, CLEVRER_FPS,
    construct_state_vector, clevrer_scene_to_state_tensor
)

# Extended color map for 15 objects
EXTENDED_COLOR_MAP = {
    'gray': [0.5, 0.5, 0.5],
    'red': [1.0, 0.0, 0.0],
    'blue': [0.0, 0.0, 1.0],
    'green': [0.0, 1.0, 0.0],
    'brown': [0.6, 0.3, 0.1],
    'purple': [0.5, 0.0, 0.5],
    'cyan': [0.0, 1.0, 1.0],
    'yellow': [1.0, 1.0, 0.0],
    'orange': [1.0, 0.5, 0.0],
    'pink': [1.0, 0.75, 0.8],
    'white': [1.0, 1.0, 1.0],
    'teal': [0.0, 0.5, 0.5],
    'gold': [1.0, 0.84, 0.0],
    'silver': [0.75, 0.75, 0.75],
    'black': [0.1, 0.1, 0.1],
}

COLORS = list(EXTENDED_COLOR_MAP.keys())
MATERIALS = list(MATERIAL_MAP.keys())
SHAPES = list(SHAPE_MAP.keys())


@dataclass
class ModelResult:
    model_name: str
    total: int = 0
    correct: int = 0
    by_type: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: {'correct': 0, 'total': 0})
    )

    def accuracy(self) -> float:
        return self.correct / max(1, self.total) * 100

    def add(self, q_type: str, is_correct: bool):
        self.total += 1
        self.correct += int(is_correct)
        self.by_type[q_type]['total'] += 1
        self.by_type[q_type]['correct'] += int(is_correct)

    def type_accuracy(self, q_type: str) -> float:
        d = self.by_type.get(q_type, {'correct': 0, 'total': 0})
        return d['correct'] / max(1, d['total']) * 100


def load_clevrer_questions(clevrer_dir: str) -> List[Dict]:
    """Load CLEVRER validation questions. Tries common filename variants."""
    candidates = [
        Path(clevrer_dir) / 'questions' / 'validation.json',
        Path(clevrer_dir) / 'questions' / 'clevrer_validation.json',
        Path(clevrer_dir) / 'questions' / 'validation_v0.2.json',
    ]
    for questions_file in candidates:
        if questions_file.exists():
            with open(questions_file, 'r') as f:
                return json.load(f)
    raise FileNotFoundError(
        f"Questions file not found. Tried: {[str(p) for p in candidates]}"
    )


def load_annotation(scene_index: int, clevrer_dir: str) -> Optional[Dict]:
    """Load CLEVRER annotation for a scene."""
    # Try zip file first (CLEVRER uses 'annotation_validation.zip' naming)
    zip_path = Path(clevrer_dir) / 'annotations' / 'annotation_validation.zip'
    if zip_path.exists():
        chunk_start = (scene_index // 1000) * 1000
        chunk_end = chunk_start + 1000
        folder = f"annotation_{chunk_start}-{chunk_end}"
        filename = f"{folder}/annotation_{scene_index}.json"
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                with zf.open(filename) as f:
                    return json.load(f)
        except KeyError:
            pass

    # Try alternative zip name
    zip_path2 = Path(clevrer_dir) / 'annotations' / 'validation.zip'
    if zip_path2.exists():
        chunk_start = (scene_index // 1000) * 1000
        chunk_end = chunk_start + 1000
        folder = f"annotation_{chunk_start}-{chunk_end}"
        filename = f"{folder}/annotation_{scene_index}.json"
        try:
            with zipfile.ZipFile(zip_path2, 'r') as zf:
                with zf.open(filename) as f:
                    return json.load(f)
        except KeyError:
            pass

    # Try direct file
    ann_file = Path(clevrer_dir) / 'annotations' / 'validation' / f'annotation_{scene_index}.json'
    if ann_file.exists():
        with open(ann_file, 'r') as f:
            return json.load(f)

    return None


def generate_additional_objects(existing_objects: List[Dict], target_count: int = 15, seed: int = 0) -> List[Dict]:
    """Generate additional objects to reach target_count, avoiding duplicates."""
    random.seed(seed)
    np.random.seed(seed)

    # Get existing color-material-shape combinations
    existing_combos = set()
    for obj in existing_objects:
        combo = (obj['color'], obj['material'], obj['shape'])
        existing_combos.add(combo)

    # Generate new objects
    new_objects = []
    next_id = max(obj['object_id'] for obj in existing_objects) + 1

    # Available combinations
    all_combos = []
    for color in COLORS:
        for material in MATERIALS:
            for shape in SHAPES:
                combo = (color, material, shape)
                if combo not in existing_combos:
                    all_combos.append(combo)

    random.shuffle(all_combos)

    needed = target_count - len(existing_objects)
    for i in range(min(needed, len(all_combos))):
        color, material, shape = all_combos[i]
        new_objects.append({
            'object_id': next_id + i,
            'color': color,
            'material': material,
            'shape': shape
        })

    return new_objects


def generate_object_trajectory_states(obj_id: int, color: str, material: str, shape: str,
                                        num_frames: int = 128, seed: int = 0) -> np.ndarray:
    """Generate state trajectory for an additional object. Returns [T, 35] array."""
    random.seed(seed + obj_id * 100)
    np.random.seed(seed + obj_id * 100)

    # Random starting position (spread out in scene)
    start_x = random.uniform(-4, 4)
    start_y = random.uniform(-4, 4)
    start_z = 0.2

    # Random velocity (some stationary, some moving)
    if random.random() < 0.5:
        vx, vy, vz = 0, 0, 0
    else:
        vx = random.uniform(-2, 2)
        vy = random.uniform(-2, 2)
        vz = 0

    states = np.zeros((num_frames, 35), dtype=np.float32)
    x, y, z = start_x, start_y, start_z

    for t in range(num_frames):
        # Construct state vector
        position = [x, y, z]
        velocity = np.array([vx, vy, vz])
        states[t] = construct_state_vector(position, velocity, shape, color, material)

        # Simple linear motion
        x += vx * (1/24)
        y += vy * (1/24)

    return states


def annotation_to_state_tensor(annotation: Dict, target_objects: int = 15, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert CLEVRER annotation to state tensors, augmenting to target_objects.

    Returns:
        states: [T, N, 35] state tensor
        masks: [T, N] validity masks
    """
    # First convert the original annotation
    states, masks, metadata = clevrer_scene_to_state_tensor(annotation)

    num_frames = states.shape[0]
    num_objects = states.shape[1]

    if num_objects >= target_objects:
        # Already have enough objects, just return first target_objects
        return states[:, :target_objects, :], masks[:, :target_objects]

    # Need to add more objects
    existing_objects = annotation.get('object_property', [])
    new_object_props = generate_additional_objects(existing_objects, target_objects, seed)

    # Create expanded tensors
    expanded_states = np.zeros((num_frames, target_objects, 35), dtype=np.float32)
    expanded_masks = np.ones((num_frames, target_objects), dtype=np.float32)

    # Copy existing objects
    expanded_states[:, :num_objects, :] = states
    expanded_masks[:, :num_objects] = masks

    # Add new objects
    for i, obj in enumerate(new_object_props):
        obj_idx = num_objects + i
        if obj_idx >= target_objects:
            break

        obj_states = generate_object_trajectory_states(
            obj['object_id'],
            obj['color'],
            obj['material'],
            obj['shape'],
            num_frames,
            seed
        )
        expanded_states[:, obj_idx, :] = obj_states

    return expanded_states, expanded_masks


def get_question_type(question: Dict) -> str:
    """Determine CLEVRER question type."""
    q_type = question.get('question_type', 'descriptive')
    if q_type == 'explanatory':
        return 'explanatory'
    elif q_type == 'predictive':
        return 'predictive'
    elif q_type == 'counterfactual':
        return 'counterfactual'
    return 'descriptive'


def check_answer(predicted: str, question: Dict) -> bool:
    """Check if answer is correct."""
    pred = predicted.strip().upper()
    if pred:
        pred = pred[0]

    choices = question.get('choices', [])
    if choices:
        # MCQ - find correct choice
        for i, c in enumerate(choices):
            if isinstance(c, dict) and c.get('answer') == 'correct':
                correct_letter = chr(65 + i)
                return pred == correct_letter

    return False


def load_adapter_model(adapter_checkpoint: str, device: str = 'cuda'):
    """Load the Physics-LLM adapter model."""
    from physics_llm_adapter.adapter_v2 import PhysicsLLMAdapterV2
    from physics_former.training.models.physics_former_full import FullPhysicsFormer

    adapter_ckpt = torch.load(adapter_checkpoint, map_location=device, weights_only=False)
    adapter_sd = adapter_ckpt['model_state_dict']

    # Infer physics model config from adapter checkpoint
    hidden_dim = adapter_sd['physics_model.encoder.object_encoder.4.weight'].shape[0]
    ff_dim = adapter_sd['physics_model.transformer_layers.0.ff.0.weight'].shape[0]
    num_layers = len([k for k in adapter_sd.keys()
                      if 'physics_model.transformer_layers' in k and '.attention.q_proj.weight' in k])
    num_heads = adapter_sd['physics_model.transformer_layers.0.attention.attention_bias_net.2.weight'].shape[0]
    max_count = adapter_sd['physics_model.counting_head_classification.6.weight'].shape[0]
    max_objects = max_count - 1

    schema_key = 'physics_model.schema_classifier.3.weight'
    num_schema_classes = adapter_sd[schema_key].shape[0] if schema_key in adapter_sd else 37

    print(f"  Physics model config: hidden_dim={hidden_dim}, num_layers={num_layers}, "
          f"num_heads={num_heads}, max_objects={max_objects}")

    # Check if max_objects is sufficient for 15 objects
    if max_objects < 15:
        print(f"  WARNING: Model trained with max_objects={max_objects}, but testing with 15 objects")
        print(f"  Will pad/truncate as needed")

    # Detect optional adapter heads + LoRA tensors so adapter_phase3.pt (with LoRA)
    # loads completely instead of partial-loading. Mirrors run_adapter_evaluation.py.
    llm_name = adapter_ckpt.get('llm_name', 'distilgpt2')
    num_prefix_tokens = adapter_ckpt.get('num_prefix_tokens', 64)
    has_mcq_head = any(k.startswith('mcq_head.') for k in adapter_sd)
    has_descriptive_head = any(k.startswith('descriptive_head.') for k in adapter_sd)
    adapter_mlp_style = "deep" if 'adapter.8.weight' in adapter_sd else "shallow"
    lora_a_keys = [k for k in adapter_sd if k.endswith('.lora_A')]
    has_lora = bool(lora_a_keys)
    lora_rank = adapter_sd[lora_a_keys[0]].shape[1] if has_lora else None
    lora_alpha = float(adapter_ckpt.get('lora_alpha', 2 * lora_rank)) if has_lora else None

    print(f"  Adapter config: llm={llm_name}, num_prefix_tokens={num_prefix_tokens}, "
          f"mlp_style={adapter_mlp_style}, mcq_head={has_mcq_head}, "
          f"descriptive_head={has_descriptive_head}, lora={has_lora}"
          + (f" (rank={lora_rank}, alpha={lora_alpha})" if has_lora else ""))

    physics_model = FullPhysicsFormer(
        state_dim=35,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        max_objects=max(max_objects, 15),  # Ensure we can handle 15 objects
        dropout=0.1,
        num_schema_classes=num_schema_classes
    ).to(device)

    adapter = PhysicsLLMAdapterV2(
        physics_model=physics_model,
        physics_dim=hidden_dim,
        llm_name=llm_name,
        num_prefix_tokens=num_prefix_tokens,
        freeze_physics=True,
        freeze_llm=False,
        build_mcq_head=has_mcq_head,
        build_descriptive_head=has_descriptive_head,
        adapter_mlp_style=adapter_mlp_style,
    ).to(device)

    if has_lora:
        # Insert LoRA wrappers BEFORE load_state_dict so the suffixed keys exist.
        adapter.apply_lora(rank=lora_rank, alpha=lora_alpha)

    # Load weights (may need to handle mismatched dimensions on legacy checkpoints)
    try:
        adapter.load_state_dict(adapter_sd)
    except RuntimeError as e:
        print(f"  Warning: Partial weight loading due to dimension mismatch: {e}")
        # Load compatible weights
        model_dict = adapter.state_dict()
        pretrained_dict = {k: v for k, v in adapter_sd.items()
                          if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        adapter.load_state_dict(model_dict)
        print(f"  Loaded {len(pretrained_dict)}/{len(adapter_sd)} weights")

    adapter.eval()

    print(f"Loaded adapter from {adapter_checkpoint}")
    return adapter


def answer_question_with_adapter(adapter, states: np.ndarray, masks: np.ndarray,
                                  question: str, choices: List[Dict], device: str = 'cuda') -> str:
    """
    Get Physics-LLM's answer to a question given state tensors.

    Args:
        adapter: PhysicsLLMAdapterV2 model
        states: [T, N, 35] state tensor
        masks: [T, N] validity mask
        question: Question text
        choices: Answer choices (list of dicts with 'choice' key)
        device: Compute device

    Returns:
        Predicted answer (A, B, C, or D)
    """
    # Extract choice texts
    choice_texts = []
    for i, c in enumerate(choices):
        choice_text = c.get('choice', c) if isinstance(c, dict) else str(c)
        choice_texts.append(f"{chr(65+i)}) {choice_text}")

    full_question = f"{question} Options: {', '.join(choice_texts)}"

    # FAIR COMPARISON: Use only frame 64 (middle frame) to match LLM input
    # LLMs receive scene description from frame 64, so Physics-LLM should too
    frame_idx = min(64, states.shape[0] - 1)  # Use frame 64 or last frame if shorter
    single_frame_states = states[frame_idx:frame_idx+1]  # [1, N, 35]

    # Convert to tensors [1, 1, N, 35] - single frame
    states_tensor = torch.tensor(single_frame_states, dtype=torch.float32).unsqueeze(0).to(device)

    # Use 2D mask [1, N] - single frame
    if masks.ndim == 2:
        masks_2d = masks[frame_idx]  # [T, N] -> [N] take frame 64
    else:
        masks_2d = masks
    masks_tensor = torch.tensor(masks_2d, dtype=torch.float32).unsqueeze(0).to(device)

    # Handle dimension mismatches - pad or truncate to model's max_objects
    model_max_objects = getattr(adapter.physics_model, 'max_objects', 20)
    num_objects = states_tensor.shape[2]

    if num_objects > model_max_objects:
        # Truncate to max_objects
        states_tensor = states_tensor[:, :, :model_max_objects, :]
        masks_tensor = masks_tensor[:, :model_max_objects]
    elif num_objects < model_max_objects:
        # Pad with zeros
        pad_size = model_max_objects - num_objects
        states_tensor = torch.nn.functional.pad(states_tensor, (0, 0, 0, pad_size))
        masks_tensor = torch.nn.functional.pad(masks_tensor, (0, pad_size))

    with torch.no_grad():
        try:
            # Call adapter's forward method with question as list
            answers = adapter.forward(
                physics_states=states_tensor,
                object_mask=masks_tensor,
                question_text=[full_question],
                max_length=50
            )

            if isinstance(answers, list) and len(answers) > 0:
                answer = str(answers[0]).split('\n')[0].strip()
            else:
                answer = str(answers).strip()

        except Exception as e:
            print(f"    Error in adapter inference: {e}")
            answer = ""

    # Extract letter answer from response
    answer_upper = answer.upper()

    # Check for explicit letter answer (A, B, C, D)
    for letter in 'ABCD':
        if letter + ')' in answer_upper or f'ANSWER: {letter}' in answer_upper or f'ANSWER IS {letter}' in answer_upper:
            return letter

    # Check if answer matches any choice text
    answer_lower = answer.lower()
    for i, c in enumerate(choices):
        choice_text = c.get('choice', c) if isinstance(c, dict) else str(c)
        if choice_text.lower() in answer_lower or answer_lower in choice_text.lower():
            return chr(65 + i)

    # Try to find any letter at start
    if answer_upper and answer_upper[0] in 'ABCD':
        return answer_upper[0]

    return ''


def run_benchmark(adapter_checkpoint: str, clevrer_dir: str, num_questions: int = 90,
                  output_dir: str = "results_15obj", device: str = 'cuda',
                  target_objects: int = 15,
                  uniform: bool = False,
                  filter_malformed: bool = False,
                  shuffle_choices: bool = False,
                  save_details: bool = False,
                  output_name: Optional[str] = None) -> ModelResult:
    """Run the N-object CLEVRER benchmark on Physics-LLM adapter.

    Args:
        uniform: If True, sample ``num_questions`` causal questions uniformly
            (matches LLM 1K-pool protocol; CLEVRER's natural per-type distribution
            is preserved). If False (legacy), balance ``num_questions // 3`` per
            type. Both branches use ``random.seed(42)``.
        filter_malformed: If True, skip MCQ questions where every choice is
            labeled wrong (matches the LLM ``validate_question`` filter and the
            1K 3--6 obj evaluation default).
        shuffle_choices: If True, randomize MCQ choice order per question (matches
            the 1K 3--6 obj primary protocol's shuffled MCQ).
        save_details: If True, write a sidecar ``.details.jsonl`` alongside the
            summary JSON with per-question records (scene_id, q_type, predicted,
            correct, choices). Useful for downstream CI / failure analysis.
        output_name: Override the default output filename stem
            ``physics_llm_{target_objects}obj``.
    """

    # Load adapter
    print("\nLoading Physics-LLM adapter...")
    adapter = load_adapter_model(adapter_checkpoint, device)

    # Load CLEVRER questions
    print("Loading CLEVRER questions...")
    all_data = load_clevrer_questions(clevrer_dir)

    # Filter to causal questions only (explanatory, predictive, counterfactual).
    # Optionally also drop malformed MCQs where every choice is wrong (matches
    # the LLM baselines' validate_question() filter for apples-to-apples comparison).
    causal_questions = []
    dropped_malformed = 0
    for scene_data in all_data:
        scene_index = scene_data.get('scene_index', 0)
        for q in scene_data.get('questions', []):
            q_type = get_question_type(q)
            if q_type not in ['explanatory', 'predictive', 'counterfactual']:
                continue
            if filter_malformed:
                choices = q.get('choices', [])
                if choices:
                    has_correct = any(
                        isinstance(c, dict) and c.get('answer') == 'correct'
                        for c in choices
                    )
                    if not has_correct:
                        dropped_malformed += 1
                        continue
            causal_questions.append({
                'scene_index': scene_index,
                'question': q,
                'q_type': q_type,
            })
    if filter_malformed:
        print(f"  validate_question filter dropped {dropped_malformed} malformed MCQs")

    # Sample questions. Two protocols:
    #   uniform=True  -> uniform random sample matching CLEVRER's natural per-type
    #                    distribution (LLM 1K-pool protocol).
    #   uniform=False -> balanced num_questions//3 per type (legacy n=90 stratified).
    random.seed(42)
    if uniform:
        random.shuffle(causal_questions)
        selected = causal_questions[:num_questions]
    else:
        by_type = defaultdict(list)
        for q in causal_questions:
            by_type[q['q_type']].append(q)
        num_per_type = num_questions // 3
        selected = []
        for q_type in ['explanatory', 'predictive', 'counterfactual']:
            type_qs = by_type[q_type]
            random.shuffle(type_qs)
            selected.extend(type_qs[:num_per_type])
        random.shuffle(selected)

    result = ModelResult(model_name="Physics-LLM")
    wrong_answers = []

    print(f"\n{'='*60}")
    print(f"{target_objects}-Object CLEVRER Benchmark: Physics-LLM")
    if uniform:
        type_counts = defaultdict(int)
        for q in selected:
            type_counts[q['q_type']] += 1
        breakdown = ', '.join(f"{t}={type_counts[t]}" for t in ['explanatory', 'predictive', 'counterfactual'])
        print(f"Questions: {len(selected)} (uniform sample; {breakdown})")
    else:
        print(f"Questions: {len(selected)} ({num_per_type} per type, balanced)")
    print(f"{'='*60}")

    # When shuffle_choices is on, derive a per-question deterministic permutation
    # from the question_id so reruns are reproducible (matches run_adapter_evaluation
    # behavior under --shuffle_choices).
    details_records = []

    for i, item in enumerate(selected):
        scene_index = item['scene_index']
        question = item['question']
        q_type = item['q_type']

        # Load and augment annotation
        annotation = load_annotation(scene_index, clevrer_dir)
        if not annotation:
            print(f"  [{i+1:3d}/{len(selected)}] {q_type:15s} SKIP (no annotation)")
            continue

        # Convert to state tensors and augment to target_objects
        try:
            states, masks = annotation_to_state_tensor(annotation, target_objects=target_objects, seed=scene_index)
        except Exception as e:
            print(f"  [{i+1:3d}/{len(selected)}] {q_type:15s} SKIP (conversion error: {e})")
            continue

        # Optional MCQ choice shuffle (deterministic per question_id).
        ask_question = question
        if shuffle_choices and question.get('choices'):
            qid = question.get('question_id', f'{scene_index}_{i}')
            rng = random.Random(f'shuffle::{qid}')
            shuffled = list(question['choices'])
            rng.shuffle(shuffled)
            ask_question = dict(question)
            ask_question['choices'] = shuffled

        # Get prediction
        pred = answer_question_with_adapter(
            adapter, states, masks,
            ask_question['question'], ask_question.get('choices', []), device
        )

        is_correct = check_answer(pred, ask_question)
        result.add(q_type, is_correct)

        status = "OK" if is_correct else "X"
        print(f"  [{i+1:3d}/{len(selected)}] {q_type:15s} {status}")

        if save_details:
            details_records.append({
                'scene_index': scene_index,
                'question_id': question.get('question_id'),
                'clevrer_type': q_type,
                'question': ask_question['question'],
                'predicted': pred,
                'correct': bool(is_correct),
                'choices': ask_question.get('choices', []),
                'num_objects': target_objects,
            })

        if not is_correct:
            wrong_answers.append({
                "scene_index": scene_index,
                "type": q_type,
                "question": ask_question['question'],
                "predicted": pred,
                "num_objects": target_objects
            })

    # Save results
    Path(output_dir).mkdir(exist_ok=True)

    results_data = {
        "model": "Physics-LLM",
        "num_objects": target_objects,
        "total": result.total,
        "correct": result.correct,
        "accuracy": result.accuracy(),
        "by_type": {
            k: {
                "correct": v["correct"],
                "total": v["total"],
                "accuracy": result.type_accuracy(k)
            }
            for k, v in result.by_type.items()
        }
    }

    stem = output_name or f"physics_llm_{target_objects}obj"
    with open(f"{output_dir}/{stem}.json", "w") as f:
        json.dump(results_data, f, indent=2)

    with open(f"{output_dir}/{stem}_wrong.json", "w") as f:
        json.dump(wrong_answers, f, indent=2)

    if save_details and details_records:
        details_path = f"{output_dir}/{stem}.details.jsonl"
        with open(details_path, 'w', encoding='utf-8') as f:
            for r in details_records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"Per-question details: {details_path} ({len(details_records)} records)")

    return result


def main():
    parser = argparse.ArgumentParser(description="N-Object CLEVRER Benchmark for Physics-LLM")
    parser.add_argument("--adapter_checkpoint", type=str, required=True,
                       help="Path to Physics-LLM adapter checkpoint")
    parser.add_argument("--clevrer_dir", type=str, default=os.environ.get('CLEVRER_DIR', 'clevrer'),
                       help="Path to CLEVRER dataset")
    parser.add_argument("--num_questions", type=int, default=90,
                       help="Number of questions. With --uniform, total sample size; "
                            "without, split evenly across the three causal types.")
    parser.add_argument("--output_dir", type=str, default="results_15obj",
                       help="Output directory for results")
    parser.add_argument("--output_name", type=str, default=None,
                       help="Output filename stem (default: physics_llm_{target_objects}obj)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device: cuda or cpu")
    parser.add_argument("--target_objects", type=int, default=15,
                       help="Target number of objects in augmented scenes")
    parser.add_argument("--uniform", action="store_true",
                       help="Uniform random sample over causal questions (matches LLM 1K-pool "
                            "protocol). Default is the legacy balanced sampling (num_questions/3 per type).")
    parser.add_argument("--filter_malformed", action="store_true",
                       help="Skip MCQs where every choice is labeled wrong (matches LLM "
                            "baselines' validate_question filter and the 1K 3-6 obj eval default).")
    parser.add_argument("--shuffle_choices", action="store_true",
                       help="Shuffle MCQ choice order per question (matches the 1K 3-6 obj "
                            "primary shuffled-MCQ protocol).")
    parser.add_argument("--save_details", action="store_true",
                       help="Write per-question .details.jsonl sidecar for downstream analysis.")
    args = parser.parse_args()

    result = run_benchmark(
        args.adapter_checkpoint,
        args.clevrer_dir,
        args.num_questions,
        args.output_dir,
        args.device,
        args.target_objects,
        uniform=args.uniform,
        filter_malformed=args.filter_malformed,
        shuffle_choices=args.shuffle_choices,
        save_details=args.save_details,
        output_name=args.output_name,
    )

    print(f"\n{'='*60}")
    print(f"Results for Physics-LLM ({args.target_objects} objects):")
    print(f"  Overall:       {result.accuracy():.1f}%")
    print(f"  Explanatory:   {result.type_accuracy('explanatory'):.1f}%")
    print(f"  Predictive:    {result.type_accuracy('predictive'):.1f}%")
    print(f"  Counterfactual: {result.type_accuracy('counterfactual'):.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
