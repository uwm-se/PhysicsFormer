"""ComPhy zero-shot OOD evaluation for the Physics-LLM adapter.

The adapter was trained only on CLEVRER. This module wraps ComPhy scenes /
questions in the same 35-D state-tensor + MCQ format the adapter expects, so
the Phase 3 checkpoint can be evaluated on a strictly different benchmark
without any retraining.

What transfers, what does not
-----------------------------
* **Scene schema**: ComPhy uses Bullet (same as CLEVRER) and exposes
  motion_trajectory + object_property in the same JSON layout. The 35-D
  state tensor and the per-frame mask both port over unchanged.
* **Hidden mass**: ComPhy adds a per-object ``mass`` attribute (1.0 light /
  5.0 heavy). We override state[13] with that value, so the encoder sees a
  ground-truth mass signal in OOD evaluation.
* **Hidden charge**: ComPhy also defines per-object ``charge`` (neutral /
  positive / negative). The 35-D state schema has **no charge slot** -- this
  is an honest architectural limitation. ComPhy questions that depend on
  charge are answered with charge invisible to the model. This is a
  partial-observability OOD setting, not a charge-blind cherry-pick.
"""
