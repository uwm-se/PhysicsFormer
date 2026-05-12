"""
MCQ Scoring Head — Direct classifier for multiple-choice questions.

Replaces LLM perplexity-based MCQ scoring with a dedicated MLP that
directly scores each choice from physics features + text embeddings.

Much faster and more trainable than running the full LLM forward pass
per choice, especially with a frozen LLM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class MCQScoringHead(nn.Module):
    """Score MCQ choices directly from physics features + text embeddings.

    Pipeline:
    1. Encode question and choices via LLM embedding table (frozen, no transformer)
    2. Project physics features + question into a context vector
    3. Score each choice against the context via MLP
    4. Cross-entropy loss on choice scores

    This is dramatically faster than the perplexity approach which runs
    the full LLM autoregressive forward pass for each choice separately.
    """

    def __init__(self, physics_dim: int, text_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.physics_dim = physics_dim
        self.text_dim = text_dim

        # Project physics + question into a context vector
        self.context_proj = nn.Sequential(
            nn.Linear(physics_dim + text_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # Project each choice text
        self.choice_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # Score: context ⊕ choice → scalar
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, physics_features, question_embed, choice_embeds):
        """
        Args:
            physics_features: [B, physics_dim]
            question_embed:   [B, text_dim]   mean-pooled LLM embeddings of question
            choice_embeds:    [B, K, text_dim] mean-pooled LLM embeddings per choice
        Returns:
            scores: [B, K]
        """
        B, K, _ = choice_embeds.shape

        # Physics-informed question context
        context = torch.cat([physics_features, question_embed], dim=-1)  # [B, phys+text]
        context = self.context_proj(context)  # [B, hidden]

        # Project choices
        choices = self.choice_proj(choice_embeds)  # [B, K, hidden]

        # Score each choice against context
        context_exp = context.unsqueeze(1).expand(-1, K, -1)  # [B, K, hidden]
        combined = torch.cat([context_exp, choices], dim=-1)  # [B, K, hidden*2]
        scores = self.scorer(combined).squeeze(-1)  # [B, K]

        return scores


def encode_texts(texts: List[str], tokenizer, model_or_embedding, device, max_length: int = 64):
    """Encode texts via LLM transformer + mean pooling.

    Args:
        texts: List of strings to encode
        tokenizer: LLM tokenizer
        model_or_embedding: Either:
            - nn.Embedding: raw token lookup (fast but weak — no context)
            - nn.Module (transformer): full contextual encoding (recommended)
              e.g. model.transformer (GPT2Model) for rich representations
        device: Target device
        max_length: Max token length

    Returns:
        pooled: [len(texts), embed_dim] mean-pooled embeddings
    """
    tokens = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_length
    )
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    with torch.no_grad():
        if isinstance(model_or_embedding, nn.Embedding):
            # Raw token embeddings only (no contextual information)
            hidden = model_or_embedding(input_ids)
        else:
            # Full transformer forward pass — produces contextual embeddings
            outputs = model_or_embedding(input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state  # [B, seq_len, embed_dim]

    mask = attention_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B, embed_dim]
    return pooled
