"""
True 20-Object Physics Benchmark for Physics-LLM

Uses the same synthetic 20-object scenarios as the LLM benchmark,
but provides state tensors instead of text descriptions.

Usage:
    python benchmark_20_objects_physics_llm.py --adapter_checkpoint path/to/adapter.pt --num_questions 90
"""

import argparse
import json
import random
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from clevrer_benchmark.scene_converter import (
    SHAPE_MAP, MATERIAL_MAP, DEFAULT_SIZE, construct_state_vector
)

# All 20 unique colors for objects
COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow',
          'orange', 'pink', 'white', 'teal', 'gold', 'silver', 'black',
          'maroon', 'navy', 'olive', 'coral', 'magenta']

EXTENDED_COLOR_MAP = {
    'gray': [0.5, 0.5, 0.5], 'red': [1.0, 0.0, 0.0], 'blue': [0.0, 0.0, 1.0],
    'green': [0.0, 1.0, 0.0], 'brown': [0.6, 0.3, 0.1], 'purple': [0.5, 0.0, 0.5],
    'cyan': [0.0, 1.0, 1.0], 'yellow': [1.0, 1.0, 0.0], 'orange': [1.0, 0.5, 0.0],
    'pink': [1.0, 0.75, 0.8], 'white': [1.0, 1.0, 1.0], 'teal': [0.0, 0.5, 0.5],
    'gold': [1.0, 0.84, 0.0], 'silver': [0.75, 0.75, 0.75], 'black': [0.1, 0.1, 0.1],
    'maroon': [0.5, 0.0, 0.0], 'navy': [0.0, 0.0, 0.5], 'olive': [0.5, 0.5, 0.0],
    'coral': [1.0, 0.5, 0.31], 'magenta': [1.0, 0.0, 1.0]
}

MATERIALS = list(MATERIAL_MAP.keys())
SHAPES = list(SHAPE_MAP.keys())
NUM_OBJECTS = 20
NUM_FRAMES = 128


@dataclass
class PhysicsObject:
    id: int
    color: str
    material: str
    shape: str
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]

    @property
    def description(self) -> str:
        return f"{self.color} {self.material} {self.shape}"

    @property
    def is_moving(self) -> bool:
        return abs(self.velocity[0]) > 0.1 or abs(self.velocity[1]) > 0.1


@dataclass
class Collision:
    obj_a: int
    obj_b: int
    frame: int
    causes_movement: bool


@dataclass
class Scene:
    objects: List[PhysicsObject]
    collisions: List[Collision]
    collision_chain: List[int]
    states: np.ndarray  # [T, N, 35]
    masks: np.ndarray   # [T, N]


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


def construct_state_vector_extended(position, velocity, shape, color, material):
    """Construct 35-dim state vector with extended color map."""
    state = np.zeros(35, dtype=np.float32)

    state[0:3] = position
    state[3:6] = velocity
    state[6:10] = [0.0, 0.0, 0.0, 1.0]  # Identity quaternion
    state[10:13] = [0.0, 0.0, 0.0]  # No angular velocity

    mat_props = MATERIAL_MAP.get(material, MATERIAL_MAP['rubber'])
    state[13] = mat_props['mass']
    state[14] = DEFAULT_SIZE
    state[15:18] = EXTENDED_COLOR_MAP.get(color, [0.5, 0.5, 0.5])
    state[18] = SHAPE_MAP.get(shape, 0)
    state[19] = 0.0  # Is static
    state[20] = mat_props['friction']
    state[21] = 1.0  # Is active

    if shape == 'sphere':
        state[25:28] = [DEFAULT_SIZE * 2, DEFAULT_SIZE * 2, DEFAULT_SIZE * 2]
    elif shape == 'cylinder':
        state[25:28] = [DEFAULT_SIZE, DEFAULT_SIZE, DEFAULT_SIZE * 1.5]
    else:
        state[25:28] = [DEFAULT_SIZE, DEFAULT_SIZE, DEFAULT_SIZE]

    state[34] = mat_props['restitution']
    return state


def generate_scene(seed: int, num_objects: int = NUM_OBJECTS) -> Scene:
    """Generate a scene with state tensors."""
    random.seed(seed)
    np.random.seed(seed)

    colors = COLORS.copy()
    random.shuffle(colors)

    objects = []
    collisions = []

    chain_length = random.randint(7, 14)
    chain_objects = list(range(chain_length))

    for i in range(num_objects):
        if i < chain_length:
            x = i * 1.5
            y = 0.0 + random.uniform(-0.2, 0.2)
            z = 0.2
            if i == 0:
                vx, vy, vz = 2.0, 0.0, 0.0
            else:
                vx, vy, vz = 0.0, 0.0, 0.0
        else:
            x = random.uniform(-5, 5)
            y = random.uniform(2, 6)
            z = 0.2
            if random.random() < 0.3:
                vx = random.uniform(-2, 2)
                vy = random.uniform(-2, 2)
                vz = 0.0
            else:
                vx, vy, vz = 0.0, 0.0, 0.0

        obj = PhysicsObject(
            id=i,
            color=colors[i],
            material=random.choice(MATERIALS),
            shape=random.choice(SHAPES),
            position=(x, y, z),
            velocity=(vx, vy, vz)
        )
        objects.append(obj)

    frame = 20
    for i in range(chain_length - 1):
        collisions.append(Collision(i, i + 1, frame, True))
        frame += random.randint(15, 25)

    non_chain = list(range(chain_length, num_objects))
    if len(non_chain) >= 2:
        for _ in range(random.randint(1, 3)):
            a, b = random.sample(non_chain, 2)
            if objects[a].is_moving:
                collisions.append(Collision(a, b, random.randint(30, 100), random.random() < 0.5))

    # Create state tensors
    states = np.zeros((NUM_FRAMES, num_objects, 35), dtype=np.float32)
    masks = np.ones((NUM_FRAMES, num_objects), dtype=np.float32)

    for t in range(NUM_FRAMES):
        for i, obj in enumerate(objects):
            # Simple linear motion
            pos = list(obj.position)
            pos[0] += obj.velocity[0] * t * (1/24)
            pos[1] += obj.velocity[1] * t * (1/24)

            states[t, i] = construct_state_vector_extended(
                pos, obj.velocity, obj.shape, obj.color, obj.material
            )

    return Scene(
        objects=objects,
        collisions=collisions,
        collision_chain=chain_objects,
        states=states,
        masks=masks
    )


def generate_explanatory_question(scene: Scene, seed: int) -> Tuple[str, List[str], int]:
    random.seed(seed)
    chain = scene.collision_chain
    if len(chain) < 2:
        target_idx = random.randint(1, min(5, len(scene.objects)-1))
    else:
        target_idx = random.choice(chain[1:])

    target = scene.objects[target_idx]
    cause_idx = None
    for c in scene.collisions:
        if c.obj_b == target_idx and c.causes_movement:
            cause_idx = c.obj_a
            break
    if cause_idx is None:
        cause_idx = target_idx - 1 if target_idx > 0 else 0

    cause = scene.objects[cause_idx]
    other_objects = [o for o in scene.objects if o.id not in [cause_idx, target_idx]]
    random.shuffle(other_objects)

    choices = [
        f"The {cause.description}",
        f"The {other_objects[0].description}" if len(other_objects) > 0 else "The gray metal cube",
        f"The {other_objects[1].description}" if len(other_objects) > 1 else "The blue rubber sphere",
        "Nothing - it was already moving"
    ]
    question = f"What caused the {target.description} to move?"
    return question, choices, 0


def generate_predictive_question(scene: Scene, seed: int) -> Tuple[str, List[str], int]:
    random.seed(seed)
    if scene.collisions:
        c = random.choice(scene.collisions)
        obj_a = scene.objects[c.obj_a]
        obj_b = scene.objects[c.obj_b]
        question = f"Will the {obj_a.description} collide with the {obj_b.description}?"
        choices = ["Yes, they will collide", "No, they will miss each other",
                   "Cannot determine from the scene", "They are moving in opposite directions"]
        return question, choices, 0

    objs = random.sample(scene.objects, 2)
    question = f"Will the {objs[0].description} collide with the {objs[1].description}?"
    choices = ["Yes, they will collide", "No, they will miss each other",
               "Cannot determine from the scene", "They are moving in opposite directions"]
    return question, choices, 1


def generate_counterfactual_question(scene: Scene, seed: int) -> Tuple[str, List[str], int]:
    random.seed(seed)
    chain = scene.collision_chain
    if len(chain) < 3:
        removed_idx = 0
        affected_idx = 1
    else:
        chain_pos = random.randint(0, len(chain) - 2)
        removed_idx = chain[chain_pos]
        affected_idx = chain[chain_pos + 1]

    removed = scene.objects[removed_idx]
    affected = scene.objects[affected_idx]

    question = f"What would happen if the {removed.description} were removed from the scene?"
    choices = [
        f"The {affected.description} would remain stationary",
        f"The {affected.description} would still move due to other collisions",
        "All objects would stop moving",
        "The scene would be unchanged"
    ]
    return question, choices, 0


def load_adapter_model(adapter_checkpoint: str, device: str = 'cuda'):
    from physics_llm_adapter.adapter_v2 import PhysicsLLMAdapterV2
    from physics_former.training.models.physics_former_full import FullPhysicsFormer

    adapter_ckpt = torch.load(adapter_checkpoint, map_location=device, weights_only=False)
    adapter_sd = adapter_ckpt['model_state_dict']

    hidden_dim = adapter_sd['physics_model.encoder.object_encoder.4.weight'].shape[0]
    ff_dim = adapter_sd['physics_model.transformer_layers.0.ff.0.weight'].shape[0]
    num_layers = len([k for k in adapter_sd.keys()
                      if 'physics_model.transformer_layers' in k and '.attention.q_proj.weight' in k])
    num_heads = adapter_sd['physics_model.transformer_layers.0.attention.attention_bias_net.2.weight'].shape[0]
    max_count = adapter_sd['physics_model.counting_head_classification.6.weight'].shape[0]
    max_objects = max_count - 1

    schema_key = 'physics_model.schema_classifier.3.weight'
    num_schema_classes = adapter_sd[schema_key].shape[0] if schema_key in adapter_sd else 37

    print(f"  Physics model: hidden={hidden_dim}, layers={num_layers}, heads={num_heads}, max_obj={max_objects}")

    physics_model = FullPhysicsFormer(
        state_dim=35,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        max_objects=max(max_objects, NUM_OBJECTS),
        dropout=0.1,
        num_schema_classes=num_schema_classes
    ).to(device)

    adapter = PhysicsLLMAdapterV2(
        physics_model=physics_model,
        physics_dim=hidden_dim,
        freeze_physics=True,
        freeze_llm=False
    ).to(device)

    try:
        adapter.load_state_dict(adapter_sd)
    except RuntimeError as e:
        print(f"  Partial weight loading: {str(e)[:100]}")
        model_dict = adapter.state_dict()
        pretrained_dict = {k: v for k, v in adapter_sd.items()
                          if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        adapter.load_state_dict(model_dict)

    adapter.eval()
    print(f"Loaded adapter from {adapter_checkpoint}")
    return adapter


def answer_question(adapter, states: np.ndarray, masks: np.ndarray,
                    question: str, choices: List[str], device: str = 'cuda') -> str:
    choice_texts = [f"{chr(65+i)}) {c}" for i, c in enumerate(choices)]
    full_question = f"{question} Options: {', '.join(choice_texts)}"

    states_tensor = torch.tensor(states, dtype=torch.float32).unsqueeze(0).to(device)
    masks_2d = masks[0]
    masks_tensor = torch.tensor(masks_2d, dtype=torch.float32).unsqueeze(0).to(device)

    model_max_objects = getattr(adapter.physics_model, 'max_objects', 20)
    num_objects = states_tensor.shape[2]

    if num_objects > model_max_objects:
        states_tensor = states_tensor[:, :, :model_max_objects, :]
        masks_tensor = masks_tensor[:, :model_max_objects]
    elif num_objects < model_max_objects:
        pad_size = model_max_objects - num_objects
        states_tensor = torch.nn.functional.pad(states_tensor, (0, 0, 0, pad_size))
        masks_tensor = torch.nn.functional.pad(masks_tensor, (0, pad_size))

    with torch.no_grad():
        try:
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
            print(f"    Error: {e}")
            answer = ""

    answer_upper = answer.upper()
    for letter in 'ABCD':
        if letter + ')' in answer_upper or f'ANSWER: {letter}' in answer_upper:
            return letter

    answer_lower = answer.lower()
    for i, c in enumerate(choices):
        if c.lower() in answer_lower or answer_lower in c.lower():
            return chr(65 + i)

    if answer_upper and answer_upper[0] in 'ABCD':
        return answer_upper[0]

    return ''


def run_benchmark(adapter_checkpoint: str, num_questions: int = 90,
                  output_dir: str = "results_20obj_synthetic", device: str = 'cuda') -> ModelResult:
    print("\nLoading Physics-LLM adapter...")
    adapter = load_adapter_model(adapter_checkpoint, device)

    result = ModelResult(model_name="Physics-LLM")
    wrong_answers = []

    num_per_type = num_questions // 3
    questions = []

    for i in range(num_per_type):
        scene = generate_scene(seed=i * 3)
        q, choices, correct = generate_explanatory_question(scene, i * 3)
        questions.append(('explanatory', scene, q, choices, correct))

        scene = generate_scene(seed=i * 3 + 1)
        q, choices, correct = generate_predictive_question(scene, i * 3 + 1)
        questions.append(('predictive', scene, q, choices, correct))

        scene = generate_scene(seed=i * 3 + 2)
        q, choices, correct = generate_counterfactual_question(scene, i * 3 + 2)
        questions.append(('counterfactual', scene, q, choices, correct))

    random.seed(42)
    random.shuffle(questions)

    print(f"\n{'='*60}")
    print(f"True {NUM_OBJECTS}-Object Physics Benchmark: Physics-LLM")
    print(f"Questions: {len(questions)} ({num_per_type} per type)")
    print(f"{'='*60}")

    for i, (q_type, scene, question, choices, correct_idx) in enumerate(questions):
        pred = answer_question(adapter, scene.states, scene.masks, question, choices, device)
        correct_letter = chr(65 + correct_idx)
        is_correct = pred == correct_letter
        result.add(q_type, is_correct)

        status = "OK" if is_correct else "X"
        print(f"  [{i+1:3d}/{len(questions)}] {q_type:15s} {status}")

        if not is_correct:
            wrong_answers.append({
                "type": q_type,
                "question": question,
                "expected": correct_letter,
                "got": pred
            })

    Path(output_dir).mkdir(exist_ok=True)

    results_data = {
        "model": "Physics-LLM",
        "benchmark": f"true_{NUM_OBJECTS}_object_synthetic",
        "num_objects": NUM_OBJECTS,
        "total": result.total,
        "correct": result.correct,
        "accuracy": result.accuracy(),
        "by_type": {k: {"correct": v["correct"], "total": v["total"], "accuracy": result.type_accuracy(k)}
                    for k, v in result.by_type.items()}
    }

    with open(f"{output_dir}/physics_llm_{NUM_OBJECTS}obj_synthetic.json", "w") as f:
        json.dump(results_data, f, indent=2)

    with open(f"{output_dir}/physics_llm_{NUM_OBJECTS}obj_synthetic_wrong.json", "w") as f:
        json.dump(wrong_answers, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="20-Object Physics Benchmark for Physics-LLM")
    parser.add_argument("--adapter_checkpoint", type=str, required=True)
    parser.add_argument("--num_questions", type=int, default=90)
    parser.add_argument("--output_dir", type=str, default="results_20obj_synthetic")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    result = run_benchmark(args.adapter_checkpoint, args.num_questions, args.output_dir, args.device)

    print(f"\n{'='*60}")
    print(f"Results for Physics-LLM (True {NUM_OBJECTS}-Object Synthetic):")
    print(f"  Overall:       {result.accuracy():.1f}%")
    print(f"  Explanatory:   {result.type_accuracy('explanatory'):.1f}%")
    print(f"  Predictive:    {result.type_accuracy('predictive'):.1f}%")
    print(f"  Counterfactual: {result.type_accuracy('counterfactual'):.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
