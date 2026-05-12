"""
CLEVRER Question to Physics-LLM Question Mapper

Maps CLEVRER question types to Physics-LLM question format and handles
answer extraction and comparison.
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum


class CLEVRERQuestionType(Enum):
    DESCRIPTIVE = "descriptive"
    EXPLANATORY = "explanatory"
    PREDICTIVE = "predictive"
    COUNTERFACTUAL = "counterfactual"


class PhysicsLLMQuestionType(Enum):
    WILL_COLLIDE = "will_collide_soon"
    TRAJECTORY = "trajectory_prediction"
    COLLISION_OUTCOME = "collision_outcome"
    ENERGY_TRANSFER = "energy_transfer"
    MOMENTUM = "momentum_conservation"
    CAUSAL = "causal_attribution"
    COUNTERFACTUAL = "counterfactual"


CLEVRER_TO_PHYSICS_LLM_MAP = {
    CLEVRERQuestionType.PREDICTIVE: [
        PhysicsLLMQuestionType.WILL_COLLIDE,
        PhysicsLLMQuestionType.TRAJECTORY,
        PhysicsLLMQuestionType.COLLISION_OUTCOME
    ],
    CLEVRERQuestionType.EXPLANATORY: [
        PhysicsLLMQuestionType.CAUSAL,
        PhysicsLLMQuestionType.ENERGY_TRANSFER,
        PhysicsLLMQuestionType.MOMENTUM
    ],
    CLEVRERQuestionType.COUNTERFACTUAL: [
        PhysicsLLMQuestionType.COUNTERFACTUAL
    ]
}

COLLISION_PATTERNS = [
    r"will .+ collide with",
    r"will .+ hit",
    r"will there be .+ collision",
    r"are .+ going to collide"
]

TRAJECTORY_PATTERNS = [
    r"where will .+ be",
    r"what direction will .+ move",
    r"will .+ enter",
    r"will .+ exit",
    r"will .+ leave"
]

CAUSAL_PATTERNS = [
    r"what caused",
    r"why did",
    r"what is responsible for",
    r"what made .+ happen"
]

COUNTERFACTUAL_PATTERNS = [
    r"what if .+ were removed",
    r"without the",
    r"if .+ had not",
    r"what would happen if"
]


def classify_clevrer_question(question: str) -> CLEVRERQuestionType:
    """Classify a CLEVRER question into its type."""
    q_lower = question.lower()
    
    for pattern in COUNTERFACTUAL_PATTERNS:
        if re.search(pattern, q_lower):
            return CLEVRERQuestionType.COUNTERFACTUAL
    
    for pattern in CAUSAL_PATTERNS:
        if re.search(pattern, q_lower):
            return CLEVRERQuestionType.EXPLANATORY
    
    for pattern in COLLISION_PATTERNS + TRAJECTORY_PATTERNS:
        if re.search(pattern, q_lower):
            return CLEVRERQuestionType.PREDICTIVE
    
    return CLEVRERQuestionType.DESCRIPTIVE


def map_to_physics_llm_type(question: str, clevrer_type: CLEVRERQuestionType) -> PhysicsLLMQuestionType:
    """Map a CLEVRER question to the most appropriate Physics-LLM question type."""
    q_lower = question.lower()
    
    if clevrer_type == CLEVRERQuestionType.COUNTERFACTUAL:
        return PhysicsLLMQuestionType.COUNTERFACTUAL
    
    if clevrer_type == CLEVRERQuestionType.EXPLANATORY:
        if "collision" in q_lower or "hit" in q_lower:
            return PhysicsLLMQuestionType.CAUSAL
        if "energy" in q_lower or "speed" in q_lower or "fast" in q_lower:
            return PhysicsLLMQuestionType.ENERGY_TRANSFER
        return PhysicsLLMQuestionType.CAUSAL
    
    if clevrer_type == CLEVRERQuestionType.PREDICTIVE:
        for pattern in COLLISION_PATTERNS:
            if re.search(pattern, q_lower):
                return PhysicsLLMQuestionType.WILL_COLLIDE
        return PhysicsLLMQuestionType.TRAJECTORY
    
    return PhysicsLLMQuestionType.TRAJECTORY


def extract_objects_from_question(question: str, object_list: List[Dict[str, str]]) -> List[int]:
    """Extract object indices mentioned in a question."""
    q_lower = question.lower()
    mentioned = []
    
    for idx, obj in enumerate(object_list):
        color = obj.get('color', '').lower()
        shape = obj.get('shape', '').lower()
        material = obj.get('material', '').lower()
        
        if color in q_lower and shape in q_lower:
            mentioned.append(idx)
        elif f"{color} {shape}" in q_lower:
            mentioned.append(idx)
        elif f"{material} {shape}" in q_lower:
            mentioned.append(idx)
    
    return mentioned


def format_physics_llm_question(
    original_question: str,
    physics_type: PhysicsLLMQuestionType,
    object_indices: List[int],
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Format a CLEVRER question for Physics-LLM input.
    
    Returns a dictionary with:
    - question_text: The formatted question
    - question_type: Physics-LLM question type
    - target_objects: Indices of objects involved
    - original_question: The original CLEVRER question
    """
    objects = metadata.get('objects', [])
    
    object_descriptions = []
    for idx in object_indices:
        if idx < len(objects):
            obj = objects[idx]
            desc = f"{obj.get('color', 'unknown')} {obj.get('material', '')} {obj.get('shape', 'object')}"
            object_descriptions.append(desc.strip())
    
    formatted = {
        'question_text': original_question,
        'question_type': physics_type.value,
        'target_objects': object_indices,
        'object_descriptions': object_descriptions,
        'original_question': original_question
    }
    
    return formatted


def parse_clevrer_answer(answer: Any) -> Tuple[str, Optional[List[str]]]:
    """
    Parse CLEVRER answer format.
    
    CLEVRER answers can be:
    - Boolean: "yes" / "no"
    - Multiple choice: list of options
    - Open-ended: text description
    """
    if isinstance(answer, bool):
        return "yes" if answer else "no", None
    
    if isinstance(answer, str):
        return answer.lower().strip(), None
    
    if isinstance(answer, list):
        return None, [str(a).lower().strip() for a in answer]
    
    return str(answer).lower().strip(), None


def compare_answers(predicted: str, ground_truth: Any, question_type: CLEVRERQuestionType) -> Tuple[bool, float]:
    """
    Compare predicted answer to ground truth.
    
    Returns:
    - correct: Boolean indicating exact match
    - score: Float score (1.0 for correct, partial credit for close answers)
    """
    gt_single, gt_list = parse_clevrer_answer(ground_truth)
    pred_lower = predicted.lower().strip()
    
    if gt_single:
        if pred_lower == gt_single:
            return True, 1.0
        
        yes_variants = {'yes', 'true', 'correct', 'will', 'does', 'did'}
        no_variants = {'no', 'false', 'incorrect', 'will not', 'won\'t', 'does not', 'didn\'t'}
        
        if gt_single in yes_variants and pred_lower in yes_variants:
            return True, 1.0
        if gt_single in no_variants and pred_lower in no_variants:
            return True, 1.0
        
        return False, 0.0
    
    if gt_list:
        if pred_lower in gt_list:
            return True, 1.0
        
        for gt_item in gt_list:
            if gt_item in pred_lower or pred_lower in gt_item:
                return True, 0.8
        
        return False, 0.0
    
    return False, 0.0


class CLEVRERQuestionMapper:
    """Maps and processes CLEVRER questions for Physics-LLM evaluation."""
    
    def __init__(self, target_types: Optional[List[CLEVRERQuestionType]] = None):
        """
        Initialize mapper with target question types.
        
        Args:
            target_types: List of CLEVRER question types to include.
                         Default: Explanatory, Predictive, Counterfactual
        """
        self.target_types = target_types or [
            CLEVRERQuestionType.EXPLANATORY,
            CLEVRERQuestionType.PREDICTIVE,
            CLEVRERQuestionType.COUNTERFACTUAL
        ]
    
    def filter_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter questions to only include target types."""
        filtered = []
        for q in questions:
            q_type = classify_clevrer_question(q.get('question', ''))
            if q_type in self.target_types:
                q['clevrer_type'] = q_type
                q['physics_llm_type'] = map_to_physics_llm_type(q.get('question', ''), q_type)
                filtered.append(q)
        return filtered
    
    def process_question(
        self,
        question_data: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a single CLEVRER question for Physics-LLM."""
        question_text = question_data.get('question', '')
        
        clevrer_type = question_data.get('clevrer_type') or classify_clevrer_question(question_text)
        physics_type = question_data.get('physics_llm_type') or map_to_physics_llm_type(question_text, clevrer_type)
        
        object_indices = extract_objects_from_question(question_text, metadata.get('objects', []))
        
        formatted = format_physics_llm_question(question_text, physics_type, object_indices, metadata)
        formatted['ground_truth'] = question_data.get('answer')
        formatted['clevrer_type'] = clevrer_type.value
        formatted['choices'] = question_data.get('choices')
        
        return formatted
    
    def evaluate_answer(self, predicted: str, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a predicted answer against ground truth."""
        ground_truth = question_data.get('ground_truth')
        clevrer_type = CLEVRERQuestionType(question_data.get('clevrer_type', 'descriptive'))
        
        correct, score = compare_answers(predicted, ground_truth, clevrer_type)
        
        return {
            'correct': correct,
            'score': score,
            'predicted': predicted,
            'ground_truth': ground_truth,
            'question_type': question_data.get('clevrer_type')
        }
