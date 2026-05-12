"""
Physics-LLM Adapter V3 — Mixed-format training variant of V2.

V2 trains exclusively on the prompt shape ``{question} Answer:`` (no choices
visible), while CLEVRER MCQ inference uses ``{question} Options: c1, c2, c3
Answer:`` (choices inserted by ``run_adapter_evaluation.py``). This train/
inference asymmetry is the documented cause of the 99.45%% ``"unknown"``
emission rate under free-form prompting (see ``free_form_transfer_test.py``
results + ``article/main.tex:464``).

V3 keeps the V2 architecture, state-dict layout, and frozen-module discipline
byte-identical, and only changes the *training-time* prompt distribution:

* **Format A** (V2's default): ``{question} Answer:`` -> target ``{answer}``
* **Format B** (NEW):         ``{question} Options: c1, c2, c3 Answer:`` -> target ``{answer}``

At each training forward pass, each sample's format is drawn independently
from ``Bernoulli(include_choices_prob)``. ``include_choices_prob=0.0``
reproduces V2 byte-identically (no Format B ever). ``0.5`` is the recipe
recommended in ``ADAPTER_GENERALIZATION_PLAN.md`` Tier 1a.

The V3 state-dict is identical to V2's — ``include_choices_prob`` and the
mixed-format RNG are Python attributes, not ``nn.Parameter``s. This means:

* V3 checkpoints can be loaded by V2 code.
* V2 checkpoints can be warm-loaded into V3 (recommended: start Phase 4 from
  ``adapter_phase3.pt``).

Inference behavior is unchanged — the generate path
(``PhysicsLLMAdapterV2.forward``) continues to prepend physics prefix and
append ``" Answer:"``; the caller continues to inject any ``Options: ...``
clause into ``question_text`` before calling forward. No inference-path
override is required because V2's inference prompt already matches Format B.

Key rollback invariant: **never overwrite ``adapter_phase3.pt``**. V3 writes
to ``checkpoints/phase4_*/``. See ``ADAPTER_GENERALIZATION_PLAN.md`` §6.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Match V2's dual-import pattern: `from adapter_v2 import ...` works when
# train_adapter_v2.py loads us as a top-level script module; the package-
# qualified path works when imported via `physics_llm_adapter.adapter_v3`
# (e.g., from the package __init__.py or evaluation scripts).
try:
    from adapter_v2 import PhysicsLLMAdapterV2
    from adapter_heads import (
        CLEVRERQuestionCategory,
        classify_clevrer_question,
    )
    from scene_summary import build_scene_summary
except ImportError:
    from physics_llm_adapter.adapter_v2 import PhysicsLLMAdapterV2
    from physics_llm_adapter.adapter_heads import (
        CLEVRERQuestionCategory,
        classify_clevrer_question,
    )
    from physics_llm_adapter.scene_summary import build_scene_summary


class PhysicsLLMAdapterV3(PhysicsLLMAdapterV2):
    """Mixed-format training subclass of ``PhysicsLLMAdapterV2``.

    Only the training-loss path is changed. See module docstring for the
    rationale and the ``ADAPTER_GENERALIZATION_PLAN.md`` Tier 1a section for
    the end-to-end recipe.

    Parameters added on top of V2:

    include_choices_prob: float
        Per-sample probability of using Format B (choices in prompt). Must be
        in ``[0.0, 1.0]``. Default 0.5.
    mixed_format_seed: int
        Seed for the Bernoulli RNG that selects Format A vs B per sample. A
        dedicated RNG avoids perturbing any other stochastic path in V2. The
        RNG is reset to this seed each time the class is instantiated, so
        resuming training from a checkpoint re-seeds deterministically. This
        is acceptable because mixed-format is a stochastic augmentation — the
        expected Format-A/B ratio is preserved across runs.

    All other ``PhysicsLLMAdapterV2.__init__`` arguments are passed through
    unchanged via ``**kwargs``.
    """

    def __init__(
        self,
        *args,
        include_choices_prob: float = 0.5,
        mixed_format_seed: int = 42,
        answer_token_dropout: float = 0.0,
        answer_token_dropout_seed: int = 4242,
        force_llm_only_routing: bool = False,
        inject_scene_text: bool = False,
        scene_text_style: str = 'comma_list',
        scene_text_include_color: bool = True,
        scene_text_include_material: bool = True,
        descriptor_strip_prob: float = 0.0,
        descriptor_strip_seed: int = 4747,
        **kwargs,
    ):
        if not 0.0 <= include_choices_prob <= 1.0:
            raise ValueError(
                f"include_choices_prob must be in [0, 1], got {include_choices_prob}"
            )
        if not 0.0 <= answer_token_dropout <= 0.5:
            raise ValueError(
                f"answer_token_dropout must be in [0, 0.5], got {answer_token_dropout}. "
                f"Values above 0.5 destroy the supervisory signal."
            )
        if scene_text_style not in ('comma_list', 'numbered'):
            raise ValueError(
                f"scene_text_style must be 'comma_list' or 'numbered', "
                f"got {scene_text_style!r}"
            )
        if not 0.0 <= descriptor_strip_prob <= 1.0:
            raise ValueError(
                f"descriptor_strip_prob must be in [0, 1], "
                f"got {descriptor_strip_prob}"
            )
        super().__init__(*args, **kwargs)
        self.include_choices_prob = float(include_choices_prob)
        self.mixed_format_seed = int(mixed_format_seed)
        # Dedicated RNG so mixed-format sampling never perturbs the global
        # torch / numpy / Python RNG states used elsewhere in training
        # (notably the physics noise injection and LoRA dropout).
        self._mix_rng = random.Random(self.mixed_format_seed)

        # Answer-token dropout. Per-token Bernoulli on input-side
        # answer tokens; labels stay un-perturbed so the model has to predict
        # the original answer despite noisy autoregressive context. This
        # attacks the verbatim-copy shortcut documented in Phase 4 free-form
        # transfer (100% "unknown" emission). RNG is dedicated to keep
        # determinism orthogonal to the format-decision RNG.
        self.answer_token_dropout = float(answer_token_dropout)
        self.answer_token_dropout_seed = int(answer_token_dropout_seed)
        self._dropout_rng = random.Random(self.answer_token_dropout_seed)

        # LLM-only routing knob. When True, ``compute_combined_loss`` skips
        # the ``classify_clevrer_question`` router and sends every sample
        # through the LLM causal-LM path (``compute_loss``). Required for
        # training on free-form prose targets: the regex classifier at
        # ``adapter_heads.classify_clevrer_question`` defaults to DESCRIPTIVE
        # for any question that doesn't match a pattern (see line 161),
        # which mis-routes most causal-QA questions (e.g. "Will A and B
        # collide?", "What happens after...?") to the descriptive head. The
        # descriptive head expects categorical labels (count / color /
        # shape / ...) not prose, so it produces zero-grad loss and training
        # stalls.
        #
        # Default False preserves V2 / mixed-format-ablation behavior
        # byte-identically (the router still runs, samples split between
        # descriptive and LLM as they did in Phases 1-3).
        self.force_llm_only_routing = bool(force_llm_only_routing)

        # Scene-text injection (Phase B). When enabled, ``compute_loss``
        # builds a deterministic per-sample scene-summary string from the
        # input state tensor (color, shape, material per object) and
        # prepends it to every training prompt:
        #     "Scene contains: red metal cube, blue rubber sphere. <question> Answer:"
        # This is the training-time counterpart to the eval-time
        # ``--inject_scene_text`` flag on ``free_form_transfer_test.py``.
        # Motivation: the encoder OOD probe
        # (``clevrer_benchmark/results/encoder_ood_probing_v2_surface.json``)
        # showed the frozen PhysicsFormer encoder destroys per-object
        # color (R^2 1.00 input -> -0.004 encoder_per_obj on CLEVRER) and
        # that the adapter prefix loses scene-presence color entirely
        # (AUC 0.81 input_sum -> 0.52 prefix_pool). Text injection routes
        # color/shape/material into the LLM as plain tokens, bypassing
        # the encoder bottleneck for surface features. Default False
        # preserves V2 / mixed-format-ablation behavior byte-identically.
        self.inject_scene_text = bool(inject_scene_text)
        self.scene_text_style = str(scene_text_style)
        self.scene_text_include_color = bool(scene_text_include_color)
        self.scene_text_include_material = bool(scene_text_include_material)

        # Phase 8: per-sample probability of stripping descriptors from the
        # scene-text. With probability ``descriptor_strip_prob`` we replace
        # the regular per-sample summary with the descriptor-free variant
        # ``"Object 1. Object 2. ..."``. The slot count and ordering are
        # preserved (so pronoun-style questions still have anchors), but
        # the color / material / shape words are absent -- forcing the
        # LoRA to read the per-object prefix slots to identify objects.
        # See ``physics_llm_adapter/scene_summary.py`` for the
        # ``style='numbered_no_descriptors'`` branch.
        #
        # Motivation (Phase 7 NLI audit, free-form transfer N=300): the
        # Phase 7 LoRA learned to use the prefix as a binary "scene
        # exists" signal but not for slot->object binding -- the failure
        # mode was template-following with INVENTED object descriptors
        # (predictions like ``"the cylinder collides with the brown
        # cube"`` in scenes containing no brown objects). Descriptor
        # stripping closes that loophole by ensuring the descriptor
        # tokens cannot be copied from the prompt on a fraction of
        # samples.
        #
        # Default 0.0 reproduces Phase 7 byte-identically; only ``training
        # mode`` enables stripping (eval always sees the full descriptors
        # to match real-world inference). Dedicated ``random.Random``
        # instance keeps determinism orthogonal to mixed-format and
        # answer-token dropout RNGs.
        self.descriptor_strip_prob = float(descriptor_strip_prob)
        self.descriptor_strip_seed = int(descriptor_strip_seed)
        self._descriptor_strip_rng = random.Random(self.descriptor_strip_seed)

        # Diagnostic counters: realised number of stripped vs full samples
        # across the lifetime of the model. Useful for confirming the
        # actual stripping rate matches ``descriptor_strip_prob``.
        self._n_strip = 0
        self._n_strip_eligible = 0

        # Counters for diagnostics; useful for confirming the realised
        # Format-A vs Format-B ratio matches include_choices_prob.
        self._n_format_a = 0
        self._n_format_b = 0
        # Counter for token-dropout realised rate -- diagnostic, useful for
        # confirming the dropout actually fired in training and at what rate.
        self._n_dropout_replaced = 0
        self._n_dropout_eligible = 0

        print(
            f"  Mixed-format training: include_choices_prob={self.include_choices_prob}"
            f" (seed={self.mixed_format_seed})"
        )
        if self.force_llm_only_routing:
            print(
                "  LLM-only routing: enabled (descriptive/numerical heads bypassed)"
            )
        if self.answer_token_dropout > 0.0:
            print(
                f"  Answer-token dropout: p={self.answer_token_dropout}"
                f" (seed={self.answer_token_dropout_seed})"
            )
        if self.inject_scene_text:
            attrs = []
            if self.scene_text_include_color:
                attrs.append('color')
            if self.scene_text_include_material:
                attrs.append('material')
            attrs.append('shape')
            print(
                f"  Scene-text injection: style={self.scene_text_style}, "
                f"attrs=[{','.join(attrs)}]"
            )
            if self.descriptor_strip_prob > 0.0:
                print(
                    f"  Descriptor stripping (Phase 8): "
                    f"p={self.descriptor_strip_prob} "
                    f"(seed={self.descriptor_strip_seed}) -- per-sample "
                    f"replacement of scene-text with "
                    f"'Object N.'-only form to force prefix grounding"
                )
        if getattr(self, 'tokens_per_object', 0) > 0:
            print(
                f"  Per-object prefix: tokens_per_object="
                f"{self.tokens_per_object}, total prefix tokens="
                f"{self.num_prefix_tokens}"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decide_format(self, sample_choices: Optional[List[str]]) -> Optional[List[str]]:
        """Per-sample Bernoulli draw: return choices (Format B) or None (Format A).

        A sample with no valid choices (``None`` or empty list) is always
        Format A — we cannot build a Format-B prompt without choice text.
        """
        if not isinstance(sample_choices, list) or not sample_choices:
            return None
        if self._mix_rng.random() < self.include_choices_prob:
            return sample_choices
        return None

    @staticmethod
    def _build_prompt(
        question: str,
        format_choices: Optional[List[str]],
        scene_text: Optional[str] = None,
    ) -> str:
        """Construct the training prompt up to (but not including) the answer text.

        If ``scene_text`` is provided, prepend it to the question with a
        single space separator. The ``Answer:`` sentinel remains at the
        very end so the label-masking logic in ``compute_loss`` (which
        masks out everything up to the answer tokens) doesn't need to
        change.
        """
        if format_choices is not None:
            prompt = f"{question} Options: {', '.join(format_choices)} Answer:"
        else:
            prompt = f"{question} Answer:"
        if scene_text:
            return f"{scene_text} {prompt}"
        return prompt

    def _build_scene_texts(
        self,
        physics_states: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> List[str]:
        """Build a deterministic scene-summary string per sample.

        Mirrors the eval-time injection in
        ``clevrer_benchmark/scripts/free_form_transfer_test.py`` but works
        on a batched physics-state tensor. For each scene in the batch we
        take the **union of visibility across time** (``mask.max(dim=0)``)
        so objects that exited the camera view mid-trajectory still appear
        in the description -- CLEVRER questions routinely reference such
        objects (e.g. "If the cylinder is removed..." when the cylinder
        bounced out at frame 40).

        Args:
            physics_states: ``[B, T, N, D]`` or ``[B, N, D]`` state tensor.
            object_mask:    ``[B, T, N]`` or ``[B, N]`` visibility mask.

        Returns:
            List of ``B`` summary strings, one per scene. The strings are
            deterministic given the states, so training is reproducible.
        """
        # We do the heavy lifting on CPU as numpy -- build_scene_summary
        # is pure numpy and the per-batch overhead is negligible next to
        # the LLM forward.
        states_np = physics_states.detach().cpu().numpy()
        mask_np = object_mask.detach().cpu().numpy() if object_mask is not None else None
        summaries: List[str] = []
        B = states_np.shape[0]

        # Phase 8: decide per-sample whether to strip descriptors. Only
        # active in training mode -- eval always sees the full descriptor
        # form (matches what the deployment-time prompt looks like). When
        # stripping fires, the per-sample summary becomes the
        # descriptor-free ``"Object N."`` form regardless of the configured
        # ``self.scene_text_style`` (the strip override is always the
        # numbered no-descriptors variant -- the cleanest forcing function).
        if self.training and self.descriptor_strip_prob > 0.0:
            strip_flags = [
                self._descriptor_strip_rng.random() < self.descriptor_strip_prob
                for _ in range(B)
            ]
            self._n_strip += sum(strip_flags)
            self._n_strip_eligible += B
        else:
            strip_flags = [False] * B

        for b in range(B):
            s_b = states_np[b]
            m_b = mask_np[b] if mask_np is not None else None
            if strip_flags[b]:
                # Descriptor-stripped variant. ``include_color`` and
                # ``include_material`` are ignored by the
                # ``numbered_no_descriptors`` style (it never calls
                # ``describe_object``); pass the same values for clarity.
                style_b = 'numbered_no_descriptors'
                inc_color_b = False
                inc_material_b = False
            else:
                style_b = self.scene_text_style
                inc_color_b = self.scene_text_include_color
                inc_material_b = self.scene_text_include_material

            if s_b.ndim == 3 and m_b is not None and m_b.ndim == 2:
                # Sequence input -- use union of masks across time as the
                # scene composition and frame 0 for static attributes.
                union = m_b.max(axis=0)
                summaries.append(
                    build_scene_summary(
                        s_b[0], union,
                        style=style_b,
                        include_color=inc_color_b,
                        include_material=inc_material_b,
                    )
                )
            else:
                # Single-frame slice. build_scene_summary handles both 2D
                # and 3D state inputs internally.
                summaries.append(
                    build_scene_summary(
                        s_b, m_b,
                        style=style_b,
                        include_color=inc_color_b,
                        include_material=inc_material_b,
                    )
                )
        return summaries

    def _apply_answer_token_dropout(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        question_lengths: List[int],
    ) -> None:
        """In-place perturb answer-region input tokens to random vocab IDs.

        For each sample i and each position j such that:
            j >= question_lengths[i]  (answer region)
            attention_mask[i, j] == 1 (non-padding)
        replace input_ids[i, j] with a uniformly-sampled vocab token with
        probability self.answer_token_dropout.

        Labels are NOT perturbed (they are captured from input_ids before
        this call in compute_loss), so cross-entropy still penalizes the
        original answer. This attacks the verbatim-copy shortcut documented
        in Phase 4 free-form transfer: with corrupted context the model
        cannot rely on autoregressive copy from preceding tokens and must
        condition on the physics prefix instead.

        Uses self._dropout_rng (dedicated Random instance seeded in
        __init__) so perturbation is deterministic per training seed and
        orthogonal to the format-decision RNG. Updates the diagnostic
        counters _n_dropout_replaced and _n_dropout_eligible so the
        realised rate can be verified post-training via the token-dropout
        stats helper.
        """
        p = self.answer_token_dropout
        if p <= 0.0:
            return
        vocab_size = len(self.tokenizer)
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        n_eligible = 0
        n_replaced = 0
        for i in range(batch_size):
            q_len = question_lengths[i]
            for j in range(q_len, seq_len):
                if attention_mask[i, j].item() != 1:
                    continue
                n_eligible += 1
                if self._dropout_rng.random() < p:
                    # Sample a vocab id uniformly. Exclude special tokens
                    # is unnecessary -- any random token works as noise,
                    # and occasional pad/eos insertions are themselves
                    # useful perturbation signal.
                    input_ids[i, j] = self._dropout_rng.randrange(vocab_size)
                    n_replaced += 1
        self._n_dropout_eligible += n_eligible
        self._n_dropout_replaced += n_replaced

    def token_dropout_stats(self) -> Dict[str, float]:
        """Return realised token-dropout rate so far.

        Useful to confirm during training that self.answer_token_dropout
        actually fired and at the expected rate (expected replaced/eligible
        ratio = self.answer_token_dropout).
        """
        total = self._n_dropout_eligible
        return {
            "n_eligible": float(total),
            "n_replaced": float(self._n_dropout_replaced),
            "realised_rate": (self._n_dropout_replaced / total) if total else 0.0,
            "target_rate": self.answer_token_dropout,
        }

    def _compute_question_lengths_mixed(
        self,
        question_text: List[str],
        format_choices: List[Optional[List[str]]],
        scene_texts: Optional[List[Optional[str]]] = None,
    ) -> List[int]:
        """Per-sample prompt-token-length computation.

        Replaces ``PhysicsLLMAdapterV2._compute_question_lengths`` for the
        training-loss path. Needed because Format B and Format A have
        different prefix lengths and label masking must mask exactly the
        prompt portion (everything up to the answer tokens). When
        ``scene_texts`` is provided, each sample's scene-summary prefix
        also contributes to the prompt length.
        """
        lengths: List[int] = []
        if scene_texts is None:
            scene_texts = [None] * len(question_text)
        for q, fc, st in zip(question_text, format_choices, scene_texts):
            prompt = self._build_prompt(q, fc, scene_text=st)
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            lengths.append(len(ids))
        return lengths

    # ------------------------------------------------------------------
    # Inference override (needed when tokens_per_object > 0)
    # ------------------------------------------------------------------

    def forward(
        self,
        physics_states,
        object_mask,
        question_text,
        max_length: int = 50,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_beams: int = 1,
        logits_processor=None,
    ):
        """Generation-path override so ``tokens_per_object > 0`` works at eval time.

        ``PhysicsLLMAdapterV2.forward`` always calls
        ``create_prefix_tokens(physics_features)`` + builds a scalar
        all-ones prefix mask. For a per-object-prefix adapter that path
        breaks in two ways:

          1. ``self.adapter`` doesn't exist (only ``self.adapter_per_object``
             was built in V2.__init__), so the call raises
             ``AttributeError``.
          2. The padded-object slots require a variable prefix mask so
             the LLM's attention skips them.

        Everything else matches V2.forward byte-for-byte: stochastic
        generation defaults, ``" Answer:"`` appended to the question,
        same decoding kwargs, same ``Answer:`` split on the generated
        text. Scene-text injection at inference is done caller-side (the
        eval scripts prepend the summary to ``question_text`` before
        calling), so V3.forward does not need to touch that path.
        """
        batch_size = physics_states.size(0)
        device = physics_states.device

        # Dispatch prefix construction. Same logic as compute_loss.
        if getattr(self, 'tokens_per_object', 0) > 0:
            prefix_tokens, prefix_mask = self.create_prefix_tokens_per_object(
                physics_states, object_mask
            )
        else:
            physics_features = self.extract_physics_features(
                physics_states, object_mask
            )
            prefix_tokens = self.create_prefix_tokens(physics_features)
            prefix_mask = torch.ones(
                batch_size, self.num_prefix_tokens, device=device,
                dtype=prefix_tokens.dtype,
            )

        prompted_questions = [q + " Answer:" for q in question_text]
        question_tokens = self.tokenizer(
            prompted_questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        question_embeds = self._get_input_embeddings()(question_tokens.input_ids)
        # Phase 7 dtype fix: when the LLM is bf16 (Qwen2.5 default) but the
        # adapter MLP outputs fp32 prefix tokens, ``cat`` promotes to fp32
        # and downstream LoRA Linear layers raise the dtype mismatch error.
        # Training masks this with autocast; eval doesn't autocast.
        prefix_tokens = prefix_tokens.to(dtype=question_embeds.dtype)
        combined_embeds = torch.cat([prefix_tokens, question_embeds], dim=1)
        combined_mask = torch.cat([prefix_mask, question_tokens.attention_mask], dim=1)
        # HuggingFace ``generate`` builds ``position_ids`` by casting the
        # attention mask to long and running ``cumsum`` on it. Our per-object
        # prefix mask inherits CLEVRER's lifecycle fractional weights (0.0,
        # 0.5, 1.0 from ``clevrer_scene_to_state_tensor``) for partial-visibility
        # objects. The ``.long()`` cast truncates 0.5 to 0, leaving a run of
        # zeros at the front of the prefix that makes cumsum-1 negative and
        # crashes ``wpe(position_ids)`` with a CUDA device-side assert
        # (``vectorized gather kernel index out of bounds``). Any slot with a
        # non-zero mask represents a real object — binarize the combined mask
        # so cumsum increments monotonically and all real positions attend.
        # Training is unaffected: ``compute_loss`` calls the LLM via its plain
        # forward path, which does not run cumsum on the mask.
        combined_mask = (combined_mask > 0).to(combined_mask.dtype)

        gen_kwargs = dict(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=do_sample,
            num_beams=num_beams,
        )
        if do_sample:
            gen_kwargs['temperature'] = temperature
            gen_kwargs['top_p'] = top_p
        if logits_processor is not None:
            # transformers accepts either a single LogitsProcessor
            # subclass or a LogitsProcessorList. Wrap a bare processor
            # in a list so the call is canonical.
            from transformers import LogitsProcessorList
            if not isinstance(logits_processor, LogitsProcessorList):
                logits_processor = LogitsProcessorList([logits_processor])
            gen_kwargs['logits_processor'] = logits_processor
        outputs = self.llm.generate(**gen_kwargs)

        generated_text = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        answers: List[str] = []
        for text in generated_text:
            if "Answer:" in text:
                answers.append(text.split("Answer:")[-1].strip())
            else:
                answers.append(text.strip())
        return answers

    # ------------------------------------------------------------------
    # Training loss overrides
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        physics_states,
        object_mask,
        question_text,
        answer_text,
        choices: Optional[List[Optional[List[str]]]] = None,
    ):
        """V3 override of ``PhysicsLLMAdapterV2.compute_loss``.

        New parameter:

        choices : Optional[List[Optional[List[str]]]]
            Per-sample MCQ choice lists, or ``None``/missing for non-MCQ
            samples. When provided, each sample's prompt format is drawn
            from ``Bernoulli(include_choices_prob)``. When ``None``, the
            behavior reproduces V2 (Format A for every sample).

        Everything else — physics feature extraction, LLM causal-LM loss
        with optional label smoothing, token-level accuracy tracking —
        matches V2 exactly, so the loss curves and optimization dynamics
        remain comparable to the Phase-3 training log.
        """
        batch_size = physics_states.size(0)
        device = physics_states.device

        # Decide prompt format per sample. If choices aren't provided at
        # all, fall back to Format A for every sample (V2 behavior).
        if choices is None:
            format_choices: List[Optional[List[str]]] = [None] * batch_size
        else:
            if len(choices) != batch_size:
                raise ValueError(
                    f"choices list length {len(choices)} != batch_size {batch_size}"
                )
            format_choices = [self._decide_format(c) for c in choices]

        # Update diagnostic counters (cheap; module-level, not per-backward).
        for fc in format_choices:
            if fc is None:
                self._n_format_a += 1
            else:
                self._n_format_b += 1

        # Prefix construction. Two mutually-exclusive paths:
        #   (a) scene-level prefix (V2 default): one mean-pooled physics
        #       vector expanded to num_prefix_tokens via self.adapter.
        #       prefix_mask is an all-ones [B, num_prefix_tokens] tensor.
        #   (b) per-object prefix (Phase C, tokens_per_object > 0): each
        #       object emits K tokens via self.adapter_per_object. Padded
        #       object slots get a zero attention mask so the LLM's
        #       attention skips them.
        # We construct prefix_tokens + prefix_mask up front so the rest of
        # the loss code can treat both paths uniformly.
        if getattr(self, 'tokens_per_object', 0) > 0:
            prefix_tokens, prefix_mask = self.create_prefix_tokens_per_object(
                physics_states, object_mask
            )
        else:
            physics_features = self.extract_physics_features(
                physics_states, object_mask
            )
            prefix_tokens = self.create_prefix_tokens(physics_features)
            prefix_mask = torch.ones(
                batch_size, self.num_prefix_tokens, device=device,
                dtype=prefix_tokens.dtype,
            )

        # Phase 7: prefix dropout. With probability ``self.prefix_dropout`` per
        # sample, zero the prefix VALUES (keep the attention mask). This forces
        # the LLM-side LoRA to learn that the prefix carries the discriminative
        # signal: if we always send a non-trivial prefix, a stronger LLM will
        # often answer correctly using only the question text + scene-text
        # injection, and the prefix becomes a no-op. Dropping it ~10% of the
        # time creates a measurable train-time loss difference between
        # "real prefix" and "zero prefix" conditions, and the only way the
        # model can reduce loss in the "real prefix" case is to actually
        # condition on the prefix tokens.
        #
        # Why not also zero the mask? Then the model just sees "shorter
        # prompt" and never learns to differentiate. Keeping the mask = 1
        # over a zero-valued prefix is the contrast signal we want.
        if self.training and self.prefix_dropout > 0.0:
            drop_mask = (
                torch.rand(batch_size, device=device) >= self.prefix_dropout
            ).to(prefix_tokens.dtype).view(batch_size, 1, 1)
            prefix_tokens = prefix_tokens * drop_mask

        # Scene-text injection. Build per-sample strings from the physics
        # states; they are prepended to every prompt the LLM sees. Done
        # once per batch so the CPU -> numpy detour in _build_scene_texts
        # amortizes over the full forward.
        if self.inject_scene_text:
            scene_texts: Optional[List[Optional[str]]] = self._build_scene_texts(
                physics_states, object_mask
            )
        else:
            scene_texts = None

        # Build full concatenated strings (prompt + answer) for the tokenizer.
        full_text: List[str] = []
        for idx, (q, a, fc) in enumerate(zip(question_text, answer_text, format_choices)):
            st = scene_texts[idx] if scene_texts is not None else None
            prompt = self._build_prompt(q, fc, scene_text=st)
            full_text.append(f"{prompt} {a}")

        tokens = self.tokenizer(
            full_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        question_lengths = self._compute_question_lengths_mixed(
            question_text, format_choices, scene_texts=scene_texts,
        )

        # Labels are derived from the ORIGINAL input_ids before any input-side
        # dropout, so the model is supervised against the un-perturbed answer
        # even when its autoregressive context has been corrupted.
        labels = tokens.input_ids.clone()
        for i in range(batch_size):
            q_len = question_lengths[i]
            labels[i, :q_len] = -100

        # Phase 5a: answer-token dropout on the input side. Replace answer
        # tokens (positions >= q_len, attention-mask=1) with random vocab
        # tokens at probability self.answer_token_dropout. Labels are NOT
        # perturbed (computed above) so cross-entropy still penalizes the
        # original answer. This breaks the verbatim-copy shortcut from
        # Phase 4 (model previously learned that copying choice text earned
        # zero loss; with dropout it cannot rely on intact preceding context
        # and must use the physics prefix).
        if self.answer_token_dropout > 0.0:
            self._apply_answer_token_dropout(
                tokens.input_ids, tokens.attention_mask, question_lengths
            )

        text_embeds = self._get_input_embeddings()(tokens.input_ids)
        # Phase 7 dtype fix (see V3.forward concat above for rationale).
        prefix_tokens = prefix_tokens.to(dtype=text_embeds.dtype)
        combined_embeds = torch.cat([prefix_tokens, text_embeds], dim=1)

        # prefix_mask already constructed above (all-ones for scene-level
        # prefix, variable for per-object prefix). Size is [B, P] where
        # P = self.num_prefix_tokens in both cases.
        combined_mask = torch.cat([prefix_mask, tokens.attention_mask], dim=1)

        prefix_labels = torch.full(
            (batch_size, self.num_prefix_tokens),
            -100,
            dtype=torch.long,
            device=device,
        )
        labels = torch.cat([prefix_labels, labels], dim=1)

        outputs = self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=labels if self.label_smoothing == 0 else None,
            return_dict=True,
        )

        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_mask = shift_labels != -100

        if self.label_smoothing > 0:
            loss_fct = nn.CrossEntropyLoss(
                ignore_index=-100, label_smoothing=self.label_smoothing
            )
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        else:
            loss = outputs.loss

        if shift_mask.any():
            preds = shift_logits.argmax(dim=-1)
            token_correct = ((preds == shift_labels) & shift_mask).sum().item()
            token_total = shift_mask.sum().item()
        else:
            token_correct = 0
            token_total = 0

        return loss, token_correct, token_total

    def compute_combined_loss(
        self,
        physics_states,
        object_mask,
        question_text,
        answer_text,
        choices=None,
        correct_choice_idx=None,
        numerical_targets: Optional[Dict[str, torch.Tensor]] = None,
        categorical_weight: float = 1.0,
        numerical_weight: float = 1.0,
        descriptive_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """V3 override: thread per-sample ``choices`` through to ``compute_loss``.

        Routing is identical to V2:

        1. Descriptive questions -> descriptive_head classification.
        2. All other questions   -> LLM causal-LM loss (V3's mixed-format one).
        3. Numerical loss        -> unchanged.

        The only behavioral change is at step 2: the LLM causal-LM loss is
        computed on Format-A or Format-B prompts per sample, where V2
        always used Format A.
        """
        device = physics_states.device
        batch_size = len(question_text)

        # Step 1: split descriptive vs reasoning (V2 logic, reimplemented
        # to avoid re-invoking super().compute_combined_loss, which would
        # call the V2-flavoured compute_loss instead of V3's).
        #
        # force_llm_only_routing short-circuits the classifier: every sample
        # goes through the LLM causal-LM path. Required for free-form prose
        # training (see __init__ docstring for the classifier-mis-routing
        # rationale).
        desc_idx: List[int] = []
        llm_idx: List[int] = []
        if self.force_llm_only_routing:
            llm_idx = list(range(batch_size))
        else:
            for i in range(batch_size):
                cat = classify_clevrer_question(question_text[i])
                if cat == CLEVRERQuestionCategory.DESCRIPTIVE:
                    desc_idx.append(i)
                else:
                    llm_idx.append(i)

        # Step 2: descriptive head loss (unchanged from V2).
        desc_loss = torch.tensor(0.0, device=device)
        if desc_idx:
            desc_loss = self.compute_descriptive_loss(
                physics_states[desc_idx],
                object_mask[desc_idx],
                [question_text[i] for i in desc_idx],
                [answer_text[i] for i in desc_idx],
            )

        # Step 3: LLM causal-LM loss for reasoning items, *with mixed-format
        # prompt construction when per-sample choices are provided*.
        cat_loss = torch.tensor(0.0, device=device)
        token_correct = 0
        token_total = 0
        if llm_idx:
            llm_choices: Optional[List[Optional[List[str]]]] = None
            if choices is not None:
                # choices may be a List[Optional[List[str]]] (the current
                # collate_fn shape) or None. Filter to the reasoning subset.
                llm_choices = [choices[i] for i in llm_idx]
            cat_loss, token_correct, token_total = self.compute_loss(
                physics_states[llm_idx],
                object_mask[llm_idx],
                [question_text[i] for i in llm_idx],
                [answer_text[i] for i in llm_idx],
                choices=llm_choices,
            )

        # Step 4: numerical loss (unchanged from V2).
        num_loss = torch.tensor(0.0, device=device)
        if numerical_targets is not None and numerical_weight > 0:
            has_nonzero = any(
                v.abs().sum().item() > 1e-8
                for v in numerical_targets.values()
                if isinstance(v, torch.Tensor)
            )
            if has_nonzero:
                num_loss = self.compute_numerical_loss(
                    physics_states, object_mask, numerical_targets
                )

        total_loss = (
            categorical_weight * cat_loss
            + descriptive_weight * desc_loss
            + numerical_weight * num_loss
        )

        return total_loss, {
            "categorical": cat_loss,
            "descriptive": desc_loss,
            "numerical": num_loss,
            "mcq_correct": token_correct,
            "mcq_total": token_total,
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def mixed_format_stats(self) -> Dict[str, float]:
        """Return realised Format-A vs Format-B ratio so far.

        Useful to confirm during training that the stochastic draw is
        producing the intended distribution (expected Format-B fraction =
        include_choices_prob averaged across MCQ-eligible samples).
        """
        total = self._n_format_a + self._n_format_b
        return {
            "n_format_a": self._n_format_a,
            "n_format_b": self._n_format_b,
            "format_b_fraction": (self._n_format_b / total) if total else 0.0,
            "configured_prob": self.include_choices_prob,
        }

    def reset_mixed_format_stats(self) -> None:
        """Zero the Format-A/B counters (e.g., at the start of each epoch)."""
        self._n_format_a = 0
        self._n_format_b = 0


def create_adapter_v3(
    physics_model,
    include_choices_prob: float = 0.5,
    mixed_format_seed: int = 42,
    inject_scene_text: bool = False,
    scene_text_style: str = 'comma_list',
    scene_text_include_color: bool = True,
    scene_text_include_material: bool = True,
    descriptor_strip_prob: float = 0.0,
    descriptor_strip_seed: int = 4747,
    **kwargs,
) -> PhysicsLLMAdapterV3:
    """Factory mirroring ``create_adapter_v2``.

    Default ``include_choices_prob=0.5`` is the Tier 1a recipe from
    ``ADAPTER_GENERALIZATION_PLAN.md``. Set to ``0.0`` to reproduce V2
    byte-identically (useful for regression testing).

    Phase B / scene-text injection args (all default off, so omitting
    them reproduces the V3 mixed-format-ablation behavior byte-identically):

    inject_scene_text: bool
        When True, prepend a deterministic scene-summary to every prompt
        at training time. See ``PhysicsLLMAdapterV3.__init__`` for the
        full motivation and probe evidence.
    scene_text_style: {'comma_list', 'numbered'}
        Template. ``'comma_list'`` -> "Scene contains: red cube, blue sphere."
        ``'numbered'`` -> "Object 1: red cube. Object 2: blue sphere." The
        numbered form makes pronoun references unambiguous in causal QA.
    scene_text_include_color / scene_text_include_material: bool
        Ablation knobs. Dropping color tests whether the LLM is actually
        learning to attend to the injected color tokens; dropping material
        tests the same for the rubber/metal distinction. Shape is always
        present because shape is load-bearing for every CLEVRER question.

    Per-object prefix (``tokens_per_object``) is passed through via
    ``**kwargs`` to the V2 base-class constructor.
    """
    return PhysicsLLMAdapterV3(
        physics_model=physics_model,
        include_choices_prob=include_choices_prob,
        mixed_format_seed=mixed_format_seed,
        inject_scene_text=inject_scene_text,
        scene_text_style=scene_text_style,
        scene_text_include_color=scene_text_include_color,
        scene_text_include_material=scene_text_include_material,
        descriptor_strip_prob=descriptor_strip_prob,
        descriptor_strip_seed=descriptor_strip_seed,
        **kwargs,
    )


__all__ = ["PhysicsLLMAdapterV3", "create_adapter_v3"]
