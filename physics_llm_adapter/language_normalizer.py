"""
Language Normalizer for Physics Reasoning

Uses the LLM to translate rich human language into canonical physics concepts
that the physics-grounded adapter can reason about.

CLEVRER-Humans uses 157 unique words and 30+ verbs, while our adapter trains
on canonical physics concepts. This module bridges that gap.

Example:
    "The blue sphere rolled into the red cube" 
    → CausalEvent(cause="blue_sphere", action="COLLISION", effect="red_cube")
    
    "The metal sphere was pushed by the gray cube"
    → CausalEvent(cause="gray_cube", action="COLLISION", effect="metal_sphere")
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Tuple
import torch


class PhysicsAction(Enum):
    """Canonical physics actions that the adapter understands."""
    COLLISION = "collision"          # Objects make contact
    ENTER = "enter"                  # Object enters scene
    EXIT = "exit"                    # Object exits scene  
    MOVE = "move"                    # Object changes position
    STOP = "stop"                    # Object stops moving
    ACCELERATE = "accelerate"        # Object speeds up
    DECELERATE = "decelerate"        # Object slows down
    PUSH = "push"                    # Force transfer (implies collision)
    BOUNCE = "bounce"                # Elastic collision
    FALL = "fall"                    # Gravity-driven motion
    SLIDE = "slide"                  # Surface contact motion
    ROLL = "roll"                    # Rotational motion
    KNOCK = "knock"                  # Impact causing displacement
    BLOCK = "block"                  # Obstruction
    CHAIN_REACTION = "chain_reaction"  # Sequential causation


# Map rich human verbs to canonical physics actions
VERB_TO_ACTION = {
    # Collision family
    "collide": PhysicsAction.COLLISION,
    "collided": PhysicsAction.COLLISION,
    "collides": PhysicsAction.COLLISION,
    "hit": PhysicsAction.COLLISION,
    "hits": PhysicsAction.COLLISION,
    "struck": PhysicsAction.COLLISION,
    "strike": PhysicsAction.COLLISION,
    "strikes": PhysicsAction.COLLISION,
    "bump": PhysicsAction.COLLISION,
    "bumped": PhysicsAction.COLLISION,
    "bumps": PhysicsAction.COLLISION,
    "crash": PhysicsAction.COLLISION,
    "crashed": PhysicsAction.COLLISION,
    "crashes": PhysicsAction.COLLISION,
    "impact": PhysicsAction.COLLISION,
    "impacted": PhysicsAction.COLLISION,
    "impacts": PhysicsAction.COLLISION,
    
    # Push/force family (implies collision + force transfer)
    "push": PhysicsAction.PUSH,
    "pushed": PhysicsAction.PUSH,
    "pushes": PhysicsAction.PUSH,
    "shove": PhysicsAction.PUSH,
    "shoved": PhysicsAction.PUSH,
    "shoves": PhysicsAction.PUSH,
    "nudge": PhysicsAction.PUSH,
    "nudged": PhysicsAction.PUSH,
    "nudges": PhysicsAction.PUSH,
    
    # Knock family (collision causing displacement)
    "knock": PhysicsAction.KNOCK,
    "knocked": PhysicsAction.KNOCK,
    "knocks": PhysicsAction.KNOCK,
    "topple": PhysicsAction.KNOCK,
    "toppled": PhysicsAction.KNOCK,
    "topples": PhysicsAction.KNOCK,
    "tip": PhysicsAction.KNOCK,
    "tipped": PhysicsAction.KNOCK,
    "tips": PhysicsAction.KNOCK,
    
    # Rolling (rotational motion, often leads to collision)
    "roll": PhysicsAction.ROLL,
    "rolled": PhysicsAction.ROLL,
    "rolls": PhysicsAction.ROLL,
    "rolling": PhysicsAction.ROLL,
    
    # Sliding (surface motion)
    "slide": PhysicsAction.SLIDE,
    "slid": PhysicsAction.SLIDE,
    "slides": PhysicsAction.SLIDE,
    "sliding": PhysicsAction.SLIDE,
    
    # Bounce (elastic collision)
    "bounce": PhysicsAction.BOUNCE,
    "bounced": PhysicsAction.BOUNCE,
    "bounces": PhysicsAction.BOUNCE,
    "rebound": PhysicsAction.BOUNCE,
    "rebounded": PhysicsAction.BOUNCE,
    "rebounds": PhysicsAction.BOUNCE,
    
    # Enter/exit
    "enter": PhysicsAction.ENTER,
    "entered": PhysicsAction.ENTER,
    "enters": PhysicsAction.ENTER,
    "appear": PhysicsAction.ENTER,
    "appeared": PhysicsAction.ENTER,
    "appears": PhysicsAction.ENTER,
    "exit": PhysicsAction.EXIT,
    "exited": PhysicsAction.EXIT,
    "exits": PhysicsAction.EXIT,
    "leave": PhysicsAction.EXIT,
    "left": PhysicsAction.EXIT,
    "leaves": PhysicsAction.EXIT,
    "disappear": PhysicsAction.EXIT,
    "disappeared": PhysicsAction.EXIT,
    "disappears": PhysicsAction.EXIT,
    
    # Motion
    "move": PhysicsAction.MOVE,
    "moved": PhysicsAction.MOVE,
    "moves": PhysicsAction.MOVE,
    "moving": PhysicsAction.MOVE,
    "travel": PhysicsAction.MOVE,
    "traveled": PhysicsAction.MOVE,
    "travels": PhysicsAction.MOVE,
    
    # Stop
    "stop": PhysicsAction.STOP,
    "stopped": PhysicsAction.STOP,
    "stops": PhysicsAction.STOP,
    "halt": PhysicsAction.STOP,
    "halted": PhysicsAction.STOP,
    "halts": PhysicsAction.STOP,
    "stationary": PhysicsAction.STOP,
    "remain": PhysicsAction.STOP,
    "remained": PhysicsAction.STOP,
    "remains": PhysicsAction.STOP,
    
    # Fall
    "fall": PhysicsAction.FALL,
    "fell": PhysicsAction.FALL,
    "falls": PhysicsAction.FALL,
    "falling": PhysicsAction.FALL,
    "drop": PhysicsAction.FALL,
    "dropped": PhysicsAction.FALL,
    "drops": PhysicsAction.FALL,
    
    # Acceleration
    "accelerate": PhysicsAction.ACCELERATE,
    "accelerated": PhysicsAction.ACCELERATE,
    "accelerates": PhysicsAction.ACCELERATE,
    "speed": PhysicsAction.ACCELERATE,
    "sped": PhysicsAction.ACCELERATE,
    "speeds": PhysicsAction.ACCELERATE,
    
    # Block
    "block": PhysicsAction.BLOCK,
    "blocked": PhysicsAction.BLOCK,
    "blocks": PhysicsAction.BLOCK,
    "obstruct": PhysicsAction.BLOCK,
    "obstructed": PhysicsAction.BLOCK,
    "obstructs": PhysicsAction.BLOCK,
}

# Prepositions that indicate causal direction
CAUSAL_PREPOSITIONS = {
    "into": "cause_to_effect",      # X rolled INTO Y → X causes effect on Y
    "onto": "cause_to_effect",
    "toward": "cause_to_effect",
    "towards": "cause_to_effect",
    "against": "cause_to_effect",
    "by": "effect_from_cause",      # X was pushed BY Y → Y is cause
    "from": "effect_from_cause",
    "off": "cause_to_effect",       # X knocked Y OFF → X causes Y to fall
    "over": "cause_to_effect",      # X knocked Y OVER → X causes Y to fall
}


@dataclass
class CausalEvent:
    """A normalized causal event extracted from natural language."""
    cause_object: Optional[str]     # Object that initiated the event
    action: PhysicsAction           # Canonical physics action
    effect_object: Optional[str]    # Object affected by the event
    is_causal: bool = True          # Whether this describes causation
    confidence: float = 1.0         # Confidence in the extraction
    original_text: str = ""         # Original text for debugging


@dataclass  
class NormalizedQuestion:
    """A question normalized to canonical physics concepts."""
    target_event: CausalEvent       # The event being asked about
    candidate_causes: List[CausalEvent]  # Potential causes to evaluate
    original_question: str
    original_options: List[str]


class LanguageNormalizer:
    """
    Normalizes rich human language to canonical physics concepts.
    
    Two modes:
    1. Rule-based: Fast, deterministic extraction using patterns
    2. LLM-enhanced: Uses the adapter's LLM for complex cases
    """
    
    # Patterns for object extraction
    OBJECT_PATTERN = r"(?:the\s+)?(\w+(?:\s+\w+)?)\s+(sphere|cube|cylinder|object|ball)"
    COLOR_PATTERN = r"(red|blue|green|yellow|purple|cyan|gray|brown|orange|pink|metal|rubber|small|large|big|tiny)"
    SIZE_WORDS = {"small", "large", "big", "tiny", "medium"}
    
    def __init__(self, use_llm: bool = False, llm_model=None, tokenizer=None):
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        
    def extract_object_reference(self, text: str) -> Optional[str]:
        """Extract object reference from text (e.g., 'the blue sphere' → 'blue_sphere')."""
        text_lower = text.lower()
        
        # Try full pattern first
        match = re.search(self.OBJECT_PATTERN, text_lower)
        if match:
            descriptor = match.group(1).strip()
            shape = match.group(2)
            return f"{descriptor}_{shape}".replace(" ", "_")
        
        # Try color + shape
        colors = re.findall(self.COLOR_PATTERN, text_lower)
        shapes = re.findall(r"(sphere|cube|cylinder|object|ball)", text_lower)
        
        if colors and shapes:
            return f"{colors[0]}_{shapes[0]}"
        elif shapes:
            return shapes[0]
        elif colors:
            return f"{colors[0]}_object"
            
        return None
    
    def extract_action(self, text: str) -> Tuple[Optional[PhysicsAction], str]:
        """Extract the physics action from text."""
        text_lower = text.lower()
        
        # Find matching verb
        for verb, action in VERB_TO_ACTION.items():
            if verb in text_lower:
                return action, verb
        
        return None, ""
    
    def extract_causal_direction(self, text: str) -> str:
        """Determine causal direction from prepositions."""
        text_lower = text.lower()
        
        for prep, direction in CAUSAL_PREPOSITIONS.items():
            if f" {prep} " in text_lower:
                return direction
        
        return "cause_to_effect"  # Default
    
    def normalize_event_description(self, text: str) -> CausalEvent:
        """
        Convert a natural language event description to a CausalEvent.
        
        Examples:
            "The blue sphere rolled into the red cube"
            → CausalEvent(cause="blue_sphere", action=ROLL/COLLISION, effect="red_cube")
            
            "The metal sphere was pushed by the gray cube"  
            → CausalEvent(cause="gray_cube", action=PUSH, effect="metal_sphere")
        """
        action, verb = self.extract_action(text)
        direction = self.extract_causal_direction(text)
        
        # Extract all object references
        objects = []
        for match in re.finditer(self.OBJECT_PATTERN, text.lower()):
            obj_ref = f"{match.group(1).strip()}_{match.group(2)}".replace(" ", "_")
            objects.append((match.start(), obj_ref))
        
        # Also try simpler color+shape patterns
        if len(objects) < 2:
            for color_match in re.finditer(self.COLOR_PATTERN, text.lower()):
                # Look for shape after color
                remaining = text.lower()[color_match.end():]
                shape_match = re.search(r"^\s*(\w*\s*)?(sphere|cube|cylinder|object|ball)", remaining)
                if shape_match:
                    obj_ref = f"{color_match.group(1)}_{shape_match.group(2)}"
                    objects.append((color_match.start(), obj_ref))
        
        # Sort by position in text
        objects.sort(key=lambda x: x[0])
        object_refs = [obj[1] for obj in objects]
        
        # Assign cause and effect based on direction
        cause_obj = None
        effect_obj = None
        
        if len(object_refs) >= 2:
            if direction == "effect_from_cause":
                # "X was pushed BY Y" → Y is cause, X is effect
                effect_obj = object_refs[0]
                cause_obj = object_refs[1]
            else:
                # "X rolled INTO Y" → X is cause, Y is effect
                cause_obj = object_refs[0]
                effect_obj = object_refs[1]
        elif len(object_refs) == 1:
            # Single object - could be either
            if direction == "effect_from_cause":
                effect_obj = object_refs[0]
            else:
                cause_obj = object_refs[0]
        
        # Determine if this is a causal statement
        is_causal = action in [
            PhysicsAction.COLLISION, PhysicsAction.PUSH, PhysicsAction.KNOCK,
            PhysicsAction.ROLL, PhysicsAction.BOUNCE, PhysicsAction.SLIDE
        ]
        
        # Non-causal actions
        if action in [PhysicsAction.STOP, PhysicsAction.ENTER, PhysicsAction.EXIT]:
            is_causal = "by" in text.lower() or "from" in text.lower()
        
        # Bouncing implies collision
        if "bouncing" in text.lower() or "bounce" in text.lower():
            action = PhysicsAction.BOUNCE
            is_causal = True
        
        # Collision keyword
        if "collision" in text.lower():
            action = PhysicsAction.COLLISION
            is_causal = True
        
        return CausalEvent(
            cause_object=cause_obj,
            action=action or PhysicsAction.MOVE,
            effect_object=effect_obj,
            is_causal=is_causal,
            confidence=0.9 if action else 0.5,
            original_text=text
        )
    
    def normalize_clevrer_humans_question(
        self,
        question: str,
        event: str,
        options: List[Dict]
    ) -> NormalizedQuestion:
        """
        Normalize a CLEVRER-Humans question to canonical physics concepts.
        
        Args:
            question: The question text
            event: The target event description
            options: List of {text: str, is_cause: bool} dicts
            
        Returns:
            NormalizedQuestion with canonical representations
        """
        # Normalize the target event
        target_event = self.normalize_event_description(event)
        
        # Normalize each candidate cause
        candidate_causes = []
        for opt in options:
            cause_event = self.normalize_event_description(opt['text'])
            candidate_causes.append(cause_event)
        
        return NormalizedQuestion(
            target_event=target_event,
            candidate_causes=candidate_causes,
            original_question=question,
            original_options=[opt['text'] for opt in options]
        )
    
    def check_causal_compatibility(
        self,
        cause_event: CausalEvent,
        effect_event: CausalEvent,
        collision_pairs: List[Tuple[str, str]] = None
    ) -> Tuple[bool, float, str]:
        """
        Check if cause_event could have caused effect_event.
        
        Uses physics reasoning:
        1. Object overlap: Does the cause involve objects in the effect?
        2. Action compatibility: Can this action type cause this effect?
        3. Collision evidence: Is there collision data supporting this?
        
        Returns:
            (is_compatible, confidence, reason)
        """
        # Check object overlap
        cause_objects = {cause_event.cause_object, cause_event.effect_object} - {None}
        effect_objects = {effect_event.cause_object, effect_event.effect_object} - {None}
        
        object_overlap = cause_objects & effect_objects
        
        # No object overlap = unlikely to be causal
        if not object_overlap and cause_objects and effect_objects:
            return False, 0.8, "no_object_overlap"
        
        # Check action compatibility
        causal_actions = {
            PhysicsAction.COLLISION, PhysicsAction.PUSH, PhysicsAction.KNOCK,
            PhysicsAction.ROLL, PhysicsAction.BOUNCE, PhysicsAction.SLIDE
        }
        
        if cause_event.action not in causal_actions:
            return False, 0.7, "non_causal_action"
        
        # Check collision evidence if available
        if collision_pairs:
            for obj1, obj2 in collision_pairs:
                if cause_event.cause_object in [obj1, obj2]:
                    if effect_event.effect_object in [obj1, obj2]:
                        return True, 0.95, "collision_evidence"
        
        # Default: compatible if objects overlap and action is causal
        if object_overlap:
            return True, 0.7, "object_overlap"
        
        return False, 0.5, "uncertain"


def create_llm_normalization_prompt(text: str) -> str:
    """Create a prompt for LLM-based normalization."""
    return f"""Extract the physics event from this description.

Text: "{text}"

Identify:
1. CAUSE_OBJECT: The object that initiated the action (color_shape format, e.g., "blue_sphere")
2. ACTION: One of [COLLISION, PUSH, KNOCK, ROLL, SLIDE, BOUNCE, MOVE, STOP, ENTER, EXIT]
3. EFFECT_OBJECT: The object affected by the action
4. IS_CAUSAL: true if this describes one object causing an effect on another

Output format (JSON):
{{"cause": "...", "action": "...", "effect": "...", "is_causal": true/false}}

Answer:"""


def create_causal_judgment_prompt(event: str, candidate: str) -> str:
    """Create a prompt for LLM to judge causal relationship."""
    return f"""Physics Causal Reasoning Task

Target Event: "{event}"
Candidate Cause: "{candidate}"

Question: Could the candidate cause have physically caused the target event?

Consider:
- Does the candidate involve physical contact or force transfer?
- Are the objects mentioned in both descriptions related?
- Is this a plausible causal chain in physics?

Answer with ONLY "yes" or "no":"""


class LLMNormalizer:
    """
    Uses an LLM to normalize language and judge causal relationships.
    
    This leverages the LLM's ability to:
    1. Understand paraphrases ("rolled into" = "collided with")
    2. Infer implicit objects ("it" refers to previously mentioned object)
    3. Reason about causal chains (A pushed B, B hit C → A caused C to move)
    """
    
    def __init__(self, model_name: str = "distilgpt2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._loaded = False
    
    def _load_model(self):
        """Lazy load the model."""
        if self._loaded:
            return
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)
            self.model.eval()
            self._loaded = True
        except Exception as e:
            print(f"Warning: Could not load LLM for normalization: {e}")
            self._loaded = False
    
    def judge_causality(self, event: str, candidate: str) -> Tuple[bool, float]:
        """
        Use LLM to judge if candidate could cause event.
        
        Returns:
            (is_causal, confidence)
        """
        self._load_model()
        if not self._loaded:
            return False, 0.0
        
        import torch
        
        prompt = create_causal_judgment_prompt(event, candidate)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response[len(prompt):].strip().lower()
        
        # Parse yes/no
        if "yes" in answer:
            return True, 0.8
        elif "no" in answer:
            return False, 0.8
        else:
            # Ambiguous - default to rule-based
            return False, 0.3
    
    def normalize_with_llm(self, text: str) -> Optional[Dict]:
        """
        Use LLM to extract structured event from text.
        
        Returns dict with cause, action, effect, is_causal or None if failed.
        """
        self._load_model()
        if not self._loaded:
            return None
        
        import torch
        import json
        
        prompt = create_llm_normalization_prompt(text)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response[len(prompt):].strip()
        
        # Try to parse JSON
        try:
            # Find JSON in response
            start = answer.find("{")
            end = answer.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(answer[start:end])
        except json.JSONDecodeError:
            pass
        
        return None


# Convenience function for quick normalization
def normalize_text(text: str) -> CausalEvent:
    """Quick normalization using rule-based approach."""
    normalizer = LanguageNormalizer()
    return normalizer.normalize_event_description(text)


if __name__ == "__main__":
    # Test examples
    normalizer = LanguageNormalizer()
    
    test_cases = [
        "The blue sphere rolled into the red cube",
        "The metal sphere was pushed by the gray cube",
        "The rubber cylinder accelerated on its own",
        "The yellow cylinder entered the scene",
        "The green cube remained stationary",
        "The off-center collision with the red sphere",
        "Sequential collisions pushed them together",
        "The small cube bouncing off the large sphere",
    ]
    
    print("Language Normalization Examples")
    print("=" * 60)
    
    for text in test_cases:
        event = normalizer.normalize_event_description(text)
        print(f"\nInput: \"{text}\"")
        print(f"  Cause:  {event.cause_object}")
        print(f"  Action: {event.action.value if event.action else 'None'}")
        print(f"  Effect: {event.effect_object}")
        print(f"  Causal: {event.is_causal}")
