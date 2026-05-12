"""
Adapter head modules and classification utilities for Physics-LLM Adapter V2.

Contains:
- OutputType, DescriptiveSubtype enums
- CLEVRER vocabulary constants and question classification
- DescriptiveSubHead, DescriptiveHead, NumericalHead nn.Modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from typing import Dict, List, Tuple, Optional


class OutputType(Enum):
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    DESCRIPTIVE = "descriptive"


# CLEVRER descriptive answer vocabulary (21 classes)
CLEVRER_DESCRIPTIVE_VOCAB = [
    # Numbers (count questions)
    "0", "1", "2", "3", "4", "5",
    # Yes/No (exist questions)
    "yes", "no",
    # Colors (query_color)
    "brown", "red", "purple", "cyan", "gray", "green", "blue", "yellow",
    # Shapes (query_shape)
    "cylinder", "sphere", "cube",
    # Materials (query_material)
    "metal", "rubber"
]
CLEVRER_DESCRIPTIVE_VOCAB_SIZE = len(CLEVRER_DESCRIPTIVE_VOCAB)
CLEVRER_ANSWER_TO_IDX = {ans: idx for idx, ans in enumerate(CLEVRER_DESCRIPTIVE_VOCAB)}
CLEVRER_IDX_TO_ANSWER = {idx: ans for idx, ans in enumerate(CLEVRER_DESCRIPTIVE_VOCAB)}


# Separate vocabularies for each descriptive subtype
class DescriptiveSubtype(Enum):
    COUNT = "count"           # 0-5
    EXIST = "exist"           # yes/no
    QUERY_COLOR = "color"     # 8 colors
    QUERY_SHAPE = "shape"     # cylinder, sphere, cube
    QUERY_MATERIAL = "material"  # metal, rubber


SUBTYPE_VOCABS = {
    DescriptiveSubtype.COUNT: ["0", "1", "2", "3", "4", "5"],
    DescriptiveSubtype.EXIST: ["yes", "no"],
    DescriptiveSubtype.QUERY_COLOR: ["brown", "red", "purple", "cyan", "gray", "green", "blue", "yellow"],
    DescriptiveSubtype.QUERY_SHAPE: ["cylinder", "sphere", "cube"],
    DescriptiveSubtype.QUERY_MATERIAL: ["metal", "rubber"],
}


def classify_descriptive_subtype(question_text: str) -> DescriptiveSubtype:
    """Classify a descriptive question into its subtype."""
    q_lower = question_text.lower().strip()
    
    if "how many" in q_lower:
        return DescriptiveSubtype.COUNT
    elif "what color" in q_lower or "what is the color" in q_lower:
        return DescriptiveSubtype.QUERY_COLOR
    elif "what shape" in q_lower or "what is the shape" in q_lower:
        return DescriptiveSubtype.QUERY_SHAPE
    elif "what material" in q_lower or "what is the material" in q_lower:
        return DescriptiveSubtype.QUERY_MATERIAL
    elif any(p in q_lower for p in ["is there", "are there", "any"]):
        return DescriptiveSubtype.EXIST
    
    # Default to exist for yes/no style questions
    return DescriptiveSubtype.EXIST


AGENT_CENTRIC_QUESTIONS = {
    "collision": OutputType.CATEGORICAL,
    "time_to_collision": OutputType.CATEGORICAL,
    "relative_motion": OutputType.CATEGORICAL,
    "distance": OutputType.CATEGORICAL,
    "speed": OutputType.CATEGORICAL,
    "direction": OutputType.CATEGORICAL,
}


# CLEVRER question type classification
class CLEVRERQuestionCategory(Enum):
    DESCRIPTIVE = "descriptive"      # count, exist, query_color, query_shape, query_material
    EXPLANATORY = "explanatory"      # what caused X, responsible for
    PREDICTIVE = "predictive"        # what will happen next
    COUNTERFACTUAL = "counterfactual"  # what if X were removed


# Patterns to detect CLEVRER question types
DESCRIPTIVE_PATTERNS = [
    "how many",
    "what color",
    "what shape",
    "what material",
    "what is the color",
    "what is the shape", 
    "what is the material",
    "are there any",
    "is there a",
    "are there",
    "is there",
]

EXPLANATORY_PATTERNS = [
    "what caused",
    "responsible for",
    "why did",
    "what made",
    "which of the following is responsible",
]

PREDICTIVE_PATTERNS = [
    "what will happen",
    "which event will happen",
    "will the",
    "what happens next",
]

COUNTERFACTUAL_PATTERNS = [
    "what if",
    "without the",
    "if the .* is removed",
    "if the .* were removed",
    "if we remove",
]


def classify_clevrer_question(question_text: str) -> CLEVRERQuestionCategory:
    """Classify a CLEVRER question into its category."""
    import re
    q_lower = question_text.lower().strip()
    
    # Check counterfactual first (most specific)
    for pattern in COUNTERFACTUAL_PATTERNS:
        if re.search(pattern, q_lower):
            return CLEVRERQuestionCategory.COUNTERFACTUAL
    
    # Check explanatory
    for pattern in EXPLANATORY_PATTERNS:
        if pattern in q_lower:
            return CLEVRERQuestionCategory.EXPLANATORY
    
    # Check predictive
    for pattern in PREDICTIVE_PATTERNS:
        if pattern in q_lower:
            return CLEVRERQuestionCategory.PREDICTIVE
    
    # Check descriptive
    for pattern in DESCRIPTIVE_PATTERNS:
        if pattern in q_lower:
            return CLEVRERQuestionCategory.DESCRIPTIVE
    
    # Default to descriptive for simple questions
    return CLEVRERQuestionCategory.DESCRIPTIVE


class DescriptiveSubHead(nn.Module):
    """Single classification sub-head for a specific descriptive subtype."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DescriptiveHead(nn.Module):
    """
    Multi-head classification for CLEVRER descriptive questions.
    
    Contains 5 specialized sub-heads:
    - count_head: 6 classes (0-5)
    - exist_head: 2 classes (yes/no)
    - color_head: 8 classes (brown, red, purple, cyan, gray, green, blue, yellow)
    - shape_head: 3 classes (cylinder, sphere, cube)
    - material_head: 2 classes (metal, rubber)
    
    Routes to appropriate sub-head based on question subtype.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Shared feature encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Specialized sub-heads
        self.count_head = DescriptiveSubHead(hidden_dim, hidden_dim // 2, num_classes=6)
        self.exist_head = DescriptiveSubHead(hidden_dim, hidden_dim // 2, num_classes=2)
        self.color_head = DescriptiveSubHead(hidden_dim, hidden_dim // 2, num_classes=8)
        self.shape_head = DescriptiveSubHead(hidden_dim, hidden_dim // 2, num_classes=3)
        self.material_head = DescriptiveSubHead(hidden_dim, hidden_dim // 2, num_classes=2)
        
        # Vocabulary mappings for each subtype
        self.subtype_vocabs = {
            DescriptiveSubtype.COUNT: ["0", "1", "2", "3", "4", "5"],
            DescriptiveSubtype.EXIST: ["yes", "no"],
            DescriptiveSubtype.QUERY_COLOR: ["brown", "red", "purple", "cyan", "gray", "green", "blue", "yellow"],
            DescriptiveSubtype.QUERY_SHAPE: ["cylinder", "sphere", "cube"],
            DescriptiveSubtype.QUERY_MATERIAL: ["metal", "rubber"],
        }
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.shared_encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _get_head_for_subtype(self, subtype: DescriptiveSubtype) -> DescriptiveSubHead:
        """Get the appropriate sub-head for a question subtype."""
        if subtype == DescriptiveSubtype.COUNT:
            return self.count_head
        elif subtype == DescriptiveSubtype.EXIST:
            return self.exist_head
        elif subtype == DescriptiveSubtype.QUERY_COLOR:
            return self.color_head
        elif subtype == DescriptiveSubtype.QUERY_SHAPE:
            return self.shape_head
        elif subtype == DescriptiveSubtype.QUERY_MATERIAL:
            return self.material_head
        else:
            return self.exist_head  # Default fallback
    
    def forward(
        self, 
        physics_features: torch.Tensor,
        subtype: Optional[DescriptiveSubtype] = None
    ) -> torch.Tensor:
        """
        Forward pass through appropriate sub-head.
        
        Args:
            physics_features: [batch, physics_dim]
            subtype: Which sub-head to use (if None, returns shared features)
        Returns:
            logits: [batch, num_classes] for the specified subtype
        """
        shared = self.shared_encoder(physics_features)
        
        if subtype is None:
            return shared
        
        head = self._get_head_for_subtype(subtype)
        return head(shared)
    
    def predict(
        self,
        physics_features: torch.Tensor,
        question_text: str
    ) -> str:
        """
        Predict answer for a descriptive question.
        
        Args:
            physics_features: [1, physics_dim] single sample
            question_text: The question to classify and answer
        Returns:
            Answer string from appropriate vocabulary
        """
        subtype = classify_descriptive_subtype(question_text)
        logits = self.forward(physics_features, subtype)
        pred_idx = torch.argmax(logits, dim=-1).item()
        vocab = self.subtype_vocabs[subtype]
        return vocab[pred_idx]
    
    def predict_batch(
        self,
        physics_features: torch.Tensor,
        question_texts: List[str]
    ) -> List[str]:
        """
        Predict answers for a batch of descriptive questions.
        
        Note: All questions in batch should ideally be same subtype for efficiency.
        For mixed subtypes, processes each individually.
        """
        # Group by subtype for efficiency
        subtypes = [classify_descriptive_subtype(q) for q in question_texts]
        
        # If all same subtype, batch process
        if len(set(subtypes)) == 1:
            subtype = subtypes[0]
            logits = self.forward(physics_features, subtype)
            pred_indices = torch.argmax(logits, dim=-1).tolist()
            vocab = self.subtype_vocabs[subtype]
            return [vocab[idx] for idx in pred_indices]
        
        # Mixed subtypes - process individually
        answers = []
        for i, (subtype, q) in enumerate(zip(subtypes, question_texts)):
            feat = physics_features[i:i+1]
            logits = self.forward(feat, subtype)
            pred_idx = torch.argmax(logits, dim=-1).item()
            vocab = self.subtype_vocabs[subtype]
            answers.append(vocab[pred_idx])
        return answers
    
    def compute_loss(
        self,
        physics_features: torch.Tensor,
        question_texts: List[str],
        answer_texts: List[str],
        label_smoothing: float = 0.1
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for descriptive questions.
        
        Routes each sample to appropriate sub-head based on question.
        """
        total_loss = torch.tensor(0.0, device=physics_features.device)
        count = 0
        
        for i, (q, a) in enumerate(zip(question_texts, answer_texts)):
            subtype = classify_descriptive_subtype(q)
            feat = physics_features[i:i+1]
            logits = self.forward(feat, subtype)
            
            vocab = self.subtype_vocabs[subtype]
            a_lower = a.lower().strip()
            if a_lower in vocab:
                target_idx = vocab.index(a_lower)
                target = torch.tensor([target_idx], device=logits.device)
                loss = F.cross_entropy(logits, target, label_smoothing=label_smoothing)
                total_loss = total_loss + loss
                count += 1
        
        return total_loss / max(count, 1)


class ObjectMaskingHead(nn.Module):
    """ALOE-style self-supervised object masking head.

    Masks random object embeddings during training and reconstructs them
    from the unmasked context.  This teaches inter-object relationships
    (who affects whom) without needing extra labels.

    Six masking schemes (randomly chosen per sample):
      0. mask single random object
      1. mask all-but-one object
      2. mask random 50% of objects
      3. mask temporally (zero one random timestep for one object)
      4. mask spatially (zero position dims for one object)
      5. mask by importance (mask highest-norm object — hardest)
    """

    def __init__(self, embed_dim: int, num_mask_schemes: int = 6,
                 mask_ratio: float = 0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_mask_schemes = num_mask_schemes
        self.mask_ratio = mask_ratio

        # Reconstruction: predict masked embeddings from context
        self.context_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.reconstruction_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _create_mask(self, batch_size: int, num_objects: int,
                     object_mask: Optional[torch.Tensor],
                     device: torch.device) -> torch.Tensor:
        """Create binary masking tensor [B, N] where 1 = masked (to reconstruct).

        Uses a randomly selected masking scheme per sample.
        """
        import random as _random

        mask = torch.zeros(batch_size, num_objects, device=device)

        for b in range(batch_size):
            if object_mask is not None:
                active = object_mask[b].bool()
                active_idx = torch.where(active)[0]
                n_active = active_idx.numel()
            else:
                active_idx = torch.arange(num_objects, device=device)
                n_active = num_objects

            if n_active < 2:
                continue  # nothing to mask if < 2 objects

            scheme = _random.randint(0, min(self.num_mask_schemes - 1, 5))

            if scheme == 0:
                # Mask single random object
                idx = active_idx[_random.randint(0, n_active - 1)]
                mask[b, idx] = 1.0
            elif scheme == 1:
                # Mask all-but-one
                keep = _random.randint(0, n_active - 1)
                for j, idx in enumerate(active_idx):
                    if j != keep:
                        mask[b, idx] = 1.0
            elif scheme == 2:
                # Mask random ~50%
                n_mask = max(1, n_active // 2)
                perm = torch.randperm(n_active, device=device)[:n_mask]
                mask[b, active_idx[perm]] = 1.0
            elif scheme == 3:
                # Mask random subset by mask_ratio
                n_mask = max(1, int(n_active * self.mask_ratio))
                perm = torch.randperm(n_active, device=device)[:n_mask]
                mask[b, active_idx[perm]] = 1.0
            elif scheme == 4:
                # Mask two random objects
                n_mask = min(2, n_active)
                perm = torch.randperm(n_active, device=device)[:n_mask]
                mask[b, active_idx[perm]] = 1.0
            else:
                # Mask single random (fallback)
                idx = active_idx[_random.randint(0, n_active - 1)]
                mask[b, idx] = 1.0

        return mask

    def forward(self, object_embeddings: torch.Tensor,
                object_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute self-supervised masking loss.

        Args:
            object_embeddings: [B, N, D] per-object embeddings (detached from
                encoder — gradient only flows through this head).
            object_mask: [B, N] binary mask of active objects.

        Returns:
            (loss, reconstruction, mask_indices) tuple.
        """
        B, N, D = object_embeddings.shape
        device = object_embeddings.device

        # Collapse object_mask to 2D if needed
        if object_mask is not None and object_mask.dim() == 3:
            object_mask = object_mask.max(dim=1)[0]

        # Create masking pattern
        masking = self._create_mask(B, N, object_mask, device)  # [B, N]

        # Save targets before masking
        targets = object_embeddings.detach().clone()

        # Zero out masked objects
        mask_expanded = (1.0 - masking).unsqueeze(-1)  # [B, N, 1]; 0 where masked
        masked_embeddings = object_embeddings * mask_expanded

        # Encode unmasked context: mean-pool unmasked objects
        unmasked_count = mask_expanded.squeeze(-1).sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        context = (masked_embeddings.sum(dim=1, keepdim=True)
                   / unmasked_count.unsqueeze(-1))  # [B, 1, D]
        context = self.context_encoder(context)  # [B, 1, D]
        context = context.expand_as(object_embeddings)  # [B, N, D]

        # Reconstruct all embeddings (loss only on masked ones)
        reconstructed = self.reconstruction_head(context)  # [B, N, D]

        # MSE loss only on masked positions
        diff = (reconstructed - targets) ** 2  # [B, N, D]
        per_object_loss = diff.mean(dim=-1)  # [B, N]

        # Weight by masking (only masked objects contribute)
        masked_loss = (per_object_loss * masking).sum() / masking.sum().clamp(min=1)

        return masked_loss, reconstructed, masking


class NumericalHead(nn.Module):
    """
    Dedicated regression head for numerical physics values.
    Outputs precise floating-point values directly from physics embeddings.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_outputs: int = 6):
        """
        Args:
            input_dim: Physics feature dimension
            hidden_dim: Hidden layer dimension
            num_outputs: Number of numerical outputs:
                0: distance (min between objects)
                1: speed (max in scene)
                2: time_to_collision (timesteps)
                3: kinetic_energy (total)
                4: momentum (magnitude)
                5: object_count
        """
        super().__init__()
        
        self.num_outputs = num_outputs
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_outputs)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, physics_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            physics_features: [batch, physics_dim]
        Returns:
            numerical_outputs: [batch, num_outputs]
        """
        return self.network(physics_features)
