"""
Training Pipelines for PhysicsFormer

PRIMARY PIPELINE (Use This):
- cls_pipeline: CLS-based training with catastrophic forgetting prevention
  * Implements Complementary Learning Systems theory
  * Experience replay (hippocampal consolidation)
  * Encoder freezing (schema protection)
  * Multi-task validation (forgetting detection)
  * Adaptive consolidation (prioritize weak tasks)

SUPPORTING PIPELINES:
- full_pipeline: Base pipeline (used by CLS pipeline)

LEGACY PIPELINES (in legacy/ folder):
- ablation_pipeline: Old ablation study (has TODOs)
- improved_ablation_pipeline: Improved version (superseded by CLS)
- complete_ablation_pipeline: Complete version (superseded by CLS)
- enhanced_pipeline: Enhanced physics (superseded by CLS)

RECOMMENDED USAGE:
    from physics_former.training.pipelines.cls_pipeline import CLSTrainingPipeline
    
    pipeline = CLSTrainingPipeline(
        config='aggressive',
        use_cls=True,
        consolidation_frequency=0.2
    )
    
    pipeline.train_stage_1(physics_loader)
    pipeline.train_stage_2(counting_loader)
    pipeline.train_stage_3(arithmetic_loader)
    pipeline.train_stage_4(symbolic_loader)
"""

# Use lazy imports to avoid circular import issues when running modules as __main__
def __getattr__(name):
    if name == 'CLSTrainingPipeline':
        from .cls_pipeline import CLSTrainingPipeline
        return CLSTrainingPipeline
    elif name == 'FullPipeline':
        from .full_pipeline import FullPipeline
        return FullPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['CLSTrainingPipeline', 'FullPipeline']
