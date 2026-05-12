"""
CLEVRER-Style QA Generator

Generates QA pairs that conform to CLEVRER benchmark format:
- Multiple choice questions with 3-4 options
- Answers must match one of the choice texts exactly
- Question types: causal_chain, future_prediction, counterfactual_reasoning
"""

import torch
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "physics_former"))


class CLEVRERQuestionType(Enum):
    CAUSAL_CHAIN = "causal_chain"
    FUTURE_PREDICTION = "future_prediction"
    COUNTERFACTUAL_REASONING = "counterfactual_reasoning"


# Object descriptors for generating natural language
COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "brown", "gray", "cyan", "pink"]
SHAPES = ["sphere", "cube", "cylinder", "cone", "object"]
MATERIALS = ["metal", "rubber", "plastic", "wooden"]


@dataclass
class CLEVRERQAPair:
    """A CLEVRER-style QA pair."""
    states: torch.Tensor
    mask: torch.Tensor
    question: str
    answer: str
    question_type: str
    choices: List[str]
    metadata: Dict


class CLEVRERQAGenerator:
    """Generates CLEVRER-conforming QA pairs from physics states."""

    # State indices (matching physics_former format)
    POSITION_IDX = slice(0, 3)
    VELOCITY_IDX = slice(3, 6)
    MASS_IDX = 13
    RADIUS_IDX = 14

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.question_types = list(CLEVRERQuestionType)

    def _get_object_descriptor(self, obj_idx: int, states: torch.Tensor) -> str:
        """Generate a natural language descriptor for an object."""
        color = COLORS[obj_idx % len(COLORS)]
        shape = SHAPES[obj_idx % len(SHAPES)]
        return f"the {color} {shape}"

    def _get_active_objects(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple:
        """Extract active objects from states."""
        if states.dim() == 4:  # [batch, seq, objects, features]
            states = states[0]
        if states.dim() == 3:  # [seq, objects, features]
            end_states = states[-1]
            start_states = states[0]
            if states.shape[0] > 1:
                computed_vel = states[-1, :, self.VELOCITY_IDX] - states[-2, :, self.VELOCITY_IDX]
            else:
                computed_vel = states[-1, :, self.VELOCITY_IDX]
        else:  # [objects, features]
            end_states = states
            start_states = states
            computed_vel = states[:, self.VELOCITY_IDX]

        if mask.dim() == 3:  # [batch, seq, objects]
            mask = mask[0, -1]
        elif mask.dim() == 2:  # [seq, objects] or [batch, objects]
            mask = mask[-1] if mask.shape[0] > 1 else mask[0]

        active_mask = mask > 0.5
        count = int(active_mask.sum().item())

        return end_states, start_states, computed_vel, active_mask, count

    def _compute_collision_pairs(
        self,
        end_states: torch.Tensor,
        computed_vel: torch.Tensor,
        active_mask: torch.Tensor,
        count: int
    ) -> List[Tuple[int, int, float]]:
        """Find pairs of objects that will/might collide."""
        collision_pairs = []

        for i in range(count):
            for j in range(i + 1, count):
                if not (active_mask[i] and active_mask[j]):
                    continue

                pos_i = end_states[i, self.POSITION_IDX]
                pos_j = end_states[j, self.POSITION_IDX]
                vel_i = computed_vel[i]
                vel_j = computed_vel[j]

                relative_pos = pos_j - pos_i
                relative_vel = vel_j - vel_i
                distance = torch.norm(relative_pos).item()

                if distance > 0.01:
                    closing_speed = -torch.dot(relative_vel, relative_pos / distance).item()
                else:
                    closing_speed = 0

                radius_i = end_states[i, self.RADIUS_IDX].item() if end_states.shape[-1] > self.RADIUS_IDX else 0.5
                radius_j = end_states[j, self.RADIUS_IDX].item() if end_states.shape[-1] > self.RADIUS_IDX else 0.5
                contact_dist = radius_i + radius_j

                # Score collision likelihood
                if closing_speed > 0 and distance < contact_dist * 5:
                    time_to_collision = distance / (closing_speed + 0.01)
                    score = closing_speed / (time_to_collision + 1)
                    collision_pairs.append((i, j, score))

        return sorted(collision_pairs, key=lambda x: -x[2])

    def generate_causal_chain_qa(
        self,
        states: torch.Tensor,
        mask: torch.Tensor
    ) -> CLEVRERQAPair:
        """Generate a causal chain (explanatory) question."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)

        if count < 2:
            # Fallback for insufficient objects
            return self._generate_fallback_qa(states, mask, CLEVRERQuestionType.CAUSAL_CHAIN)

        # Find collision pairs
        collision_pairs = self._compute_collision_pairs(end_states, computed_vel, active_mask, count)

        # Generate question about what caused a collision/event
        if len(collision_pairs) >= 2:
            # Question: What is NOT responsible for collision between X and Y?
            main_pair = collision_pairs[0]
            obj_a = self._get_object_descriptor(main_pair[0], end_states)
            obj_b = self._get_object_descriptor(main_pair[1], end_states)

            # Correct answer: something unrelated
            unrelated_objs = [i for i in range(count) if i not in main_pair[:2] and active_mask[i]]

            if unrelated_objs:
                correct_idx = self.rng.choice(unrelated_objs)
                correct_answer = f"{self._get_object_descriptor(correct_idx, end_states)}'s presence"

                # Wrong choices: things that ARE responsible
                wrong_choices = []
                wrong_choices.append(f"{obj_a}'s motion toward {obj_b}")
                wrong_choices.append(f"{obj_b}'s trajectory")
                if len(collision_pairs) > 1:
                    other_pair = collision_pairs[1]
                    wrong_choices.append(f"the collision between {self._get_object_descriptor(other_pair[0], end_states)} and {self._get_object_descriptor(other_pair[1], end_states)}")

                choices = [correct_answer] + wrong_choices[:3]
                self.rng.shuffle(choices)

                question = f"Which of the following is NOT responsible for the collision between {obj_a} and {obj_b}? Options: {', '.join(choices)}"

                return CLEVRERQAPair(
                    states=states,
                    mask=mask,
                    question=question,
                    answer=correct_answer,
                    question_type=CLEVRERQuestionType.CAUSAL_CHAIN.value,
                    choices=choices,
                    metadata={
                        "main_collision": (main_pair[0], main_pair[1]),
                        "unrelated_object": correct_idx
                    }
                )

        # Simpler causal question
        moving_objects = [(i, torch.norm(computed_vel[i]).item()) for i in range(count) if active_mask[i]]
        moving_objects.sort(key=lambda x: -x[1])

        if moving_objects and moving_objects[0][1] > 0.1:
            fastest_idx = moving_objects[0][0]
            fastest_obj = self._get_object_descriptor(fastest_idx, end_states)

            correct_answer = f"{fastest_obj} received an initial impulse"
            wrong_choices = [
                f"{self._get_object_descriptor((fastest_idx + 1) % count, end_states)} pushed it",
                "gravity acceleration",
                "magnetic attraction"
            ]

            choices = [correct_answer] + wrong_choices
            self.rng.shuffle(choices)

            question = f"What caused {fastest_obj} to move? Options: {', '.join(choices)}"

            return CLEVRERQAPair(
                states=states,
                mask=mask,
                question=question,
                answer=correct_answer,
                question_type=CLEVRERQuestionType.CAUSAL_CHAIN.value,
                choices=choices,
                metadata={"moving_object": fastest_idx}
            )

        return self._generate_fallback_qa(states, mask, CLEVRERQuestionType.CAUSAL_CHAIN)

    def generate_future_prediction_qa(
        self,
        states: torch.Tensor,
        mask: torch.Tensor
    ) -> CLEVRERQAPair:
        """Generate a future prediction (predictive) question."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)

        if count < 2:
            return self._generate_fallback_qa(states, mask, CLEVRERQuestionType.FUTURE_PREDICTION)

        collision_pairs = self._compute_collision_pairs(end_states, computed_vel, active_mask, count)

        if collision_pairs:
            # Predict which collision will happen
            best_pair = collision_pairs[0]
            obj_a = self._get_object_descriptor(best_pair[0], end_states)
            obj_b = self._get_object_descriptor(best_pair[1], end_states)

            correct_answer = f"{obj_a} collides with {obj_b}"

            # Generate wrong answers (collisions that won't happen or will happen later)
            wrong_choices = []

            # Add unlikely collisions
            for i in range(count):
                for j in range(i + 1, count):
                    if (i, j) != (best_pair[0], best_pair[1]) and active_mask[i] and active_mask[j]:
                        obj_i = self._get_object_descriptor(i, end_states)
                        obj_j = self._get_object_descriptor(j, end_states)
                        wrong_choices.append(f"{obj_i} collides with {obj_j}")
                        if len(wrong_choices) >= 3:
                            break
                if len(wrong_choices) >= 3:
                    break

            # Add non-collision events as distractors
            if len(wrong_choices) < 3:
                wrong_choices.append(f"{obj_a} exits the scene")
            if len(wrong_choices) < 3:
                wrong_choices.append("no collision occurs")

            choices = [correct_answer] + wrong_choices[:3]
            self.rng.shuffle(choices)

            question = f"What will happen next? Options: {', '.join(choices)}"

            return CLEVRERQAPair(
                states=states,
                mask=mask,
                question=question,
                answer=correct_answer,
                question_type=CLEVRERQuestionType.FUTURE_PREDICTION.value,
                choices=choices,
                metadata={"predicted_collision": (best_pair[0], best_pair[1])}
            )

        # No collision predicted - objects will continue moving
        moving_objects = [(i, torch.norm(computed_vel[i]).item()) for i in range(count) if active_mask[i]]

        if moving_objects:
            fastest_idx, fastest_speed = max(moving_objects, key=lambda x: x[1])
            fastest_obj = self._get_object_descriptor(fastest_idx, end_states)

            if fastest_speed > 0.05:
                correct_answer = f"{fastest_obj} continues moving"
            else:
                correct_answer = "objects remain stationary"

            wrong_choices = [
                f"{self._get_object_descriptor((fastest_idx + 1) % count, end_states)} collides with {fastest_obj}",
                f"{fastest_obj} reverses direction",
                "all objects stop"
            ]

            choices = [correct_answer] + wrong_choices
            self.rng.shuffle(choices)

            question = f"What will happen next? Options: {', '.join(choices)}"

            return CLEVRERQAPair(
                states=states,
                mask=mask,
                question=question,
                answer=correct_answer,
                question_type=CLEVRERQuestionType.FUTURE_PREDICTION.value,
                choices=choices,
                metadata={"prediction": "continued_motion" if fastest_speed > 0.05 else "stationary"}
            )

        return self._generate_fallback_qa(states, mask, CLEVRERQuestionType.FUTURE_PREDICTION)

    def generate_counterfactual_qa(
        self,
        states: torch.Tensor,
        mask: torch.Tensor
    ) -> CLEVRERQAPair:
        """Generate a counterfactual reasoning question."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)

        if count < 2:
            return self._generate_fallback_qa(states, mask, CLEVRERQuestionType.COUNTERFACTUAL_REASONING)

        # Pick an object to hypothetically remove
        active_indices = [i for i in range(count) if active_mask[i]]
        removed_idx = self.rng.choice(active_indices)
        removed_obj = self._get_object_descriptor(removed_idx, end_states)

        # Find collisions that would/wouldn't happen without this object
        collision_pairs = self._compute_collision_pairs(end_states, computed_vel, active_mask, count)

        # Collisions involving the removed object
        removed_collisions = [(i, j, s) for i, j, s in collision_pairs if removed_idx in (i, j)]
        # Collisions NOT involving the removed object
        other_collisions = [(i, j, s) for i, j, s in collision_pairs if removed_idx not in (i, j)]

        if removed_collisions:
            # Question: What will NOT happen if we remove this object?
            collision = removed_collisions[0]
            other_obj_idx = collision[0] if collision[1] == removed_idx else collision[1]
            other_obj = self._get_object_descriptor(other_obj_idx, end_states)

            correct_answer = f"{removed_obj} collides with {other_obj}"

            # Wrong choices: things that WILL still happen
            wrong_choices = []
            for i, j, _ in other_collisions[:2]:
                obj_i = self._get_object_descriptor(i, end_states)
                obj_j = self._get_object_descriptor(j, end_states)
                wrong_choices.append(f"{obj_i} collides with {obj_j}")

            if len(wrong_choices) < 3:
                remaining = [i for i in active_indices if i != removed_idx]
                if remaining:
                    obj = self._get_object_descriptor(self.rng.choice(remaining), end_states)
                    wrong_choices.append(f"{obj} continues moving")
            if len(wrong_choices) < 3:
                wrong_choices.append("other objects remain unaffected")

            choices = [correct_answer] + wrong_choices[:3]
            self.rng.shuffle(choices)

            question = f"Without {removed_obj}, which of the following will NOT happen? Options: {', '.join(choices)}"

            return CLEVRERQAPair(
                states=states,
                mask=mask,
                question=question,
                answer=correct_answer,
                question_type=CLEVRERQuestionType.COUNTERFACTUAL_REASONING.value,
                choices=choices,
                metadata={"removed_object": removed_idx, "prevented_collision": (collision[0], collision[1])}
            )

        # Removing object has minimal effect
        correct_answer = f"the scene is largely unchanged"

        wrong_choices = []
        remaining = [i for i in active_indices if i != removed_idx]
        if len(remaining) >= 2:
            obj1 = self._get_object_descriptor(remaining[0], end_states)
            obj2 = self._get_object_descriptor(remaining[1], end_states)
            wrong_choices.append(f"{obj1} collides with {obj2}")
        wrong_choices.append(f"all motion stops")
        wrong_choices.append(f"a chain reaction occurs")

        choices = [correct_answer] + wrong_choices[:3]
        self.rng.shuffle(choices)

        question = f"If {removed_obj} were removed, what would happen? Options: {', '.join(choices)}"

        return CLEVRERQAPair(
            states=states,
            mask=mask,
            question=question,
            answer=correct_answer,
            question_type=CLEVRERQuestionType.COUNTERFACTUAL_REASONING.value,
            choices=choices,
            metadata={"removed_object": removed_idx, "effect": "minimal"}
        )

    def _generate_fallback_qa(
        self,
        states: torch.Tensor,
        mask: torch.Tensor,
        q_type: CLEVRERQuestionType
    ) -> CLEVRERQAPair:
        """Generate a fallback QA when there aren't enough objects."""
        correct_answer = "insufficient objects to analyze"
        choices = [
            correct_answer,
            "collision will occur",
            "objects are moving",
            "scene is static"
        ]
        self.rng.shuffle(choices)

        question = f"What can be determined about this scene? Options: {', '.join(choices)}"

        return CLEVRERQAPair(
            states=states,
            mask=mask,
            question=question,
            answer=correct_answer,
            question_type=q_type.value,
            choices=choices,
            metadata={"fallback": True}
        )

    def generate_qa_pair(
        self,
        states: torch.Tensor,
        mask: torch.Tensor,
        question_type: Optional[CLEVRERQuestionType] = None
    ) -> CLEVRERQAPair:
        """Generate a CLEVRER-style QA pair."""
        if question_type is None:
            question_type = self.rng.choice(self.question_types)

        if question_type == CLEVRERQuestionType.CAUSAL_CHAIN:
            return self.generate_causal_chain_qa(states, mask)
        elif question_type == CLEVRERQuestionType.FUTURE_PREDICTION:
            return self.generate_future_prediction_qa(states, mask)
        elif question_type == CLEVRERQuestionType.COUNTERFACTUAL_REASONING:
            return self.generate_counterfactual_qa(states, mask)
        else:
            raise ValueError(f"Unknown question type: {question_type}")


def normalize_states(states: torch.Tensor, max_objects: int = 20) -> torch.Tensor:
    """Normalize states to shape [max_objects, state_dim]."""
    # Handle different input shapes
    if states.dim() == 4:  # [batch, seq, objects, features]
        states = states[0, -1]  # Take last timestep of first batch
    elif states.dim() == 3:  # [seq, objects, features] or [batch, objects, features]
        states = states[-1]  # Take last timestep
    # Now states is [objects, features]

    num_objects, state_dim = states.shape

    # Pad or truncate to max_objects
    if num_objects < max_objects:
        pad = torch.zeros(max_objects - num_objects, state_dim, dtype=states.dtype)
        states = torch.cat([states, pad], dim=0)
    elif num_objects > max_objects:
        states = states[:max_objects]

    return states


def normalize_mask(mask: torch.Tensor, max_objects: int = 20) -> torch.Tensor:
    """Normalize mask to shape [max_objects]."""
    # Handle different input shapes
    if mask.dim() == 3:  # [batch, seq, objects]
        mask = mask[0, -1]
    elif mask.dim() == 2:  # [seq, objects] or [batch, objects]
        mask = mask[-1]
    # Now mask is [objects]

    num_objects = mask.shape[0]

    # Pad or truncate to max_objects
    if num_objects < max_objects:
        pad = torch.zeros(max_objects - num_objects, dtype=mask.dtype)
        mask = torch.cat([mask, pad], dim=0)
    elif num_objects > max_objects:
        mask = mask[:max_objects]

    return mask


def generate_clevrer_training_data(
    physics_dataset,
    num_samples: int = 10000,
    seed: int = 42,
    max_objects: int = 20
) -> List[Dict]:
    """Generate CLEVRER-conforming training data from physics dataset."""
    generator = CLEVRERQAGenerator(seed=seed)
    qa_pairs = []

    from tqdm import tqdm

    dataset_size = len(physics_dataset)

    for i in tqdm(range(num_samples), desc="Generating CLEVRER QA"):
        idx = random.Random(seed + i).randint(0, dataset_size - 1)
        sample = physics_dataset[idx]

        states = sample['object_states']
        mask = sample['object_mask']

        qa = generator.generate_qa_pair(states, mask)

        # Normalize states and mask to consistent shape for training
        normalized_states = normalize_states(qa.states, max_objects)
        normalized_mask = normalize_mask(qa.mask, max_objects)

        qa_pairs.append({
            'states': normalized_states,
            'mask': normalized_mask,
            'question': qa.question,
            'answer': qa.answer,
            'question_type': qa.question_type,
            'choices': qa.choices,
            'metadata': qa.metadata,
            'numerical_targets': {
                'distance': 0.0,
                'speed': 0.0,
                'time_to_collision': 0.0,
                'kinetic_energy': 0.0,
                'momentum': 0.0,
                'object_count': 0.0
            }
        })

    return qa_pairs


if __name__ == '__main__':
    print("CLEVRER-Style QA Generator")
    print("=" * 50)
    print("Usage:")
    print("  from clevrer_qa_generator import CLEVRERQAGenerator, generate_clevrer_training_data")
    print("  generator = CLEVRERQAGenerator()")
    print("  qa = generator.generate_qa_pair(states, mask)")
    print()
    print("Or generate full training data:")
    print("  qa_pairs = generate_clevrer_training_data(physics_dataset, num_samples=10000)")
