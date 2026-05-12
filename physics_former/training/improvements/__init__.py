"""
Training Improvements for PhysicsFormer

Phase 1: Quick Wins (+9-17%)
- Curriculum Learning
- Better Optimization

Phase 2: Architecture Improvements (+9-14%)
- Auxiliary Tasks
- Contrastive Learning
- Enhanced Number Encoder

Phase 3: Compositional Reasoning (+10-15%)
- Multi-Step Arithmetic
- Relational Reasoning
- Algebraic Problem Solving

Phase 4: Meta-Learning (+5-10%)
- Few-Shot Operation Learning
- Self-Verification
- Adaptive Learning

Phase 5: Conceptual Understanding (+15-20%)
- Concept Formation
- Analogical Reasoning
- Pattern Recognition

Phase 6: Physics-Informed Architecture (NEW)
- Physics Auxiliary Loss: Explicit dynamics prediction objectives
- Spatiotemporal Encoding: Object identity + temporal position
- Factorized Attention: Separate spatial (object) and temporal attention
"""

from .curriculum_learning import (
    CurriculumScheduler,
    CurriculumDataLoader,
    train_with_curriculum
)

from .better_optimization import (
    create_optimizer,
    create_scheduler,
    WarmupCosineScheduler,
    GradientClipping,
    EarlyStopping
)

from .auxiliary_tasks import (
    AuxiliaryTaskHeads,
    AuxiliaryTaskLoss,
    generate_auxiliary_targets
)

from .contrastive_learning import (
    ContrastiveLoss,
    TripletLoss,
    InfoNCELoss,
    OrderingLoss,
    CombinedContrastiveLoss,
    generate_triplets
)

from .enhanced_number_encoder import (
    EnhancedNumberEncoder,
    PositionalNumberEncoder,
    HybridEnhancedEncoder
)

from .compositional_reasoning import (
    MultiStepArithmeticHead,
    RelationalReasoningModule,
    AlgebraicReasoningModule,
    CompositionalReasoningLoss
)

from .meta_learning import (
    MetaOperationLearner,
    SelfVerificationModule,
    AdaptiveLearningRate,
    MetaLearningLoss
)

from .conceptual_understanding import (
    ConceptFormationModule,
    AnalogicalReasoningModule,
    AbstractPatternRecognizer,
    ConceptualUnderstandingLoss
)

from .physics_auxiliary_loss import (
    PhysicsAuxiliaryLoss,
    PhysicsFormerWithAuxLoss,
    create_physics_aux_loss
)

from .spatiotemporal_encoding import (
    SpatioTemporalEncoding,
    RelativePositionalEncoding,
    create_spatiotemporal_encoding
)

from .factorized_attention import (
    SpatialAttention,
    TemporalAttention,
    FactorizedPhysicsBlock,
    FactorizedPhysicsEncoder,
    create_factorized_encoder
)

from .enhanced_physics_former import (
    EnhancedPhysicsFormer,
    EnhancedStateEncoder,
    create_enhanced_physics_former
)

__all__ = [
    # Curriculum learning
    'CurriculumScheduler',
    'CurriculumDataLoader',
    'train_with_curriculum',
    
    # Optimization
    'create_optimizer',
    'create_scheduler',
    'WarmupCosineScheduler',
    'GradientClipping',
    'EarlyStopping',
    
    # Auxiliary tasks
    'AuxiliaryTaskHeads',
    'AuxiliaryTaskLoss',
    'generate_auxiliary_targets',
    
    # Contrastive learning
    'ContrastiveLoss',
    'TripletLoss',
    'InfoNCELoss',
    'OrderingLoss',
    'CombinedContrastiveLoss',
    'generate_triplets',
    
    # Enhanced encoders
    'EnhancedNumberEncoder',
    'PositionalNumberEncoder',
    'HybridEnhancedEncoder',
    
    # Compositional reasoning
    'MultiStepArithmeticHead',
    'RelationalReasoningModule',
    'AlgebraicReasoningModule',
    'CompositionalReasoningLoss',
    
    # Meta-learning
    'MetaOperationLearner',
    'SelfVerificationModule',
    'AdaptiveLearningRate',
    'MetaLearningLoss',
    
    # Conceptual understanding
    'ConceptFormationModule',
    'AnalogicalReasoningModule',
    'AbstractPatternRecognizer',
    'ConceptualUnderstandingLoss',
    
    # Physics-informed architecture (Phase 6)
    'PhysicsAuxiliaryLoss',
    'PhysicsFormerWithAuxLoss',
    'create_physics_aux_loss',
    'SpatioTemporalEncoding',
    'RelativePositionalEncoding',
    'create_spatiotemporal_encoding',
    'SpatialAttention',
    'TemporalAttention',
    'FactorizedPhysicsBlock',
    'FactorizedPhysicsEncoder',
    'create_factorized_encoder',
    
    # Enhanced PhysicsFormer (integrated improvements)
    'EnhancedPhysicsFormer',
    'EnhancedStateEncoder',
    'create_enhanced_physics_former',
]
