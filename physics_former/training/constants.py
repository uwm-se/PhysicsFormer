"""
Central Constants for PhysicsFormer

All magic numbers should be defined here for easy maintenance and configuration.
Import these constants instead of using hard-coded values.
"""

# ============================================================================
# Model Architecture Constants
# ============================================================================

# Compositional Output
DIGIT_CLASSES = 11  # 0-9 digits + padding token (10)
PAD_TOKEN = 10      # Padding token for compositional output
MAX_DIGITS = 10     # Maximum number of digits (up to 10 billion)

# Number Encoding
FIXED_EMBEDDING_RANGE = 100  # Use fixed embeddings for 0-100
MAX_NUMBER_VALUE = 10_000_000_000  # 10 billion (10 digits)

# Object State Dimensions
STATE_DIM = 35  # Object state dimensions (35D comprehensive physics state)
POSITION_DIM = 3  # x, y, z
VELOCITY_DIM = 3  # vx, vy, vz
ORIENTATION_DIM = 4  # quaternion
ANGULAR_VELOCITY_DIM = 3  # wx, wy, wz
MASS_DIM = 1
# Total: 13 (motion) + 8 (properties) + 3 (bbox) + 5 (contact) + 6 (accel) = 35

# State Vector Indices (35D - comprehensive physics state)
# Motion State (13 dims)
POSITION_START_IDX = 0
POSITION_END_IDX = 3
VELOCITY_START_IDX = 3
VELOCITY_END_IDX = 6
ORIENTATION_START_IDX = 6
ORIENTATION_END_IDX = 10
ANGULAR_VELOCITY_START_IDX = 10
ANGULAR_VELOCITY_END_IDX = 13

# Dynamic features for delta prediction (indices 0-12: position, velocity, quaternion, angular_vel)
# These are the only features that change meaningfully frame-to-frame
# Static properties (mass, shape, friction) and corrupted features (force, torque) are excluded
DYNAMIC_FEATURE_INDICES = list(range(0, 13))  # position(0-2), velocity(3-5), quat(6-9), ang_vel(10-12)
DYNAMIC_FEATURE_END_IDX = 13  # First 13 features are dynamic

# Object Properties (8 dims: 13-16, 21-24)
MASS_IDX = 13
SIZE_IDX = 14
FRICTION_IDX = 15
RESTITUTION_IDX = 16
SHAPE_TYPE_IDX = 21
DENSITY_IDX = 22
LINEAR_DAMPING_IDX = 23
ANGULAR_DAMPING_IDX = 24

# Collision Geometry (3 dims: 25-27)
BOUNDING_BOX_START_IDX = 25
BOUNDING_BOX_END_IDX = 28

# Contact Info (5 dims: 17-20, 34)
IS_IN_CONTACT_IDX = 17
CONTACT_FORCE_START_IDX = 18
CONTACT_FORCE_END_IDX = 21
CONTACT_COUNT_IDX = 34

# Dynamics - Derived (6 dims: 28-33)
LINEAR_ACCEL_START_IDX = 28
LINEAR_ACCEL_END_IDX = 31
ANGULAR_ACCEL_START_IDX = 31
ANGULAR_ACCEL_END_IDX = 34

# Legacy aliases for backward compatibility
SIZE_START_IDX = 14
SIZE_END_IDX = 15

# Physics Schemas
NUM_PHYSICS_SCHEMAS = 85  # Total number of Isaac Sim physics schemas (15 groups)

# Operations
NUM_OPERATIONS = 4  # add, subtract, multiply, divide
OPERATION_NAMES = ['add', 'subtract', 'multiply', 'divide']
OPERATION_IDS = {
    'add': 0,
    'subtract': 1,
    'multiply': 2,
    'divide': 3
}

# ============================================================================
# Training Constants
# ============================================================================

# Gradient Clipping
MAX_GRAD_NORM = 1.0  # Maximum gradient norm for clipping

# Logging
LOG_INTERVAL = 100  # Log every N batches
CHECKPOINT_INTERVAL = 5  # Save checkpoint every N epochs
VALIDATION_INTERVAL = 1000  # Validate every N steps

# Curriculum Learning
CURRICULUM_STAGES = 3  # Easy, medium, hard
CURRICULUM_EPOCHS_PER_STAGE = 33  # For 100 epoch training

# Loss Weights (default)
PHYSICS_PREDICTION_WEIGHT = 1.0
ENERGY_CONSERVATION_WEIGHT = 0.1
MOMENTUM_CONSERVATION_WEIGHT = 0.1
COUNTING_WEIGHT = 0.5
ARITHMETIC_WEIGHT = 0.5
SYMBOLIC_WEIGHT = 1.0
DIGIT_LOSS_WEIGHT = 1.0
LENGTH_LOSS_WEIGHT = 0.1

# ============================================================================
# Data Constants
# ============================================================================

# Counting Range
MIN_COUNT = 0
MAX_COUNT = 100

# Arithmetic Training Ranges
ARITHMETIC_EASY_MAX = 10
ARITHMETIC_MEDIUM_MAX = 50
ARITHMETIC_HARD_MAX = 100

# Symbolic Training Ranges
SYMBOLIC_EASY_MAX = 10
SYMBOLIC_MEDIUM_MAX = 100
SYMBOLIC_HARD_MAX = 10_000

# Zero-Shot Test Ranges
ZERO_SHOT_RANGE_1 = (0, 10)  # Training range
ZERO_SHOT_RANGE_2 = (11, 100)  # 10× beyond training
ZERO_SHOT_RANGE_3 = (101, 1_000)  # 100× beyond training
ZERO_SHOT_RANGE_4 = (1_001, 10_000)  # 1000× beyond training
ZERO_SHOT_RANGE_5 = (10_001, 100_000)  # 10,000× beyond training

# Data Generation
PHYSICS_TRAJECTORIES_TRAIN = 1_000_000
PHYSICS_TRAJECTORIES_VAL = 50_000
PHYSICS_TRAJECTORIES_TEST = 50_000

COUNTING_PROBLEMS_TRAIN = 500_000
COUNTING_PROBLEMS_VAL = 25_000
COUNTING_PROBLEMS_TEST = 25_000

ARITHMETIC_PROBLEMS_TRAIN = 2_000_000
ARITHMETIC_PROBLEMS_VAL = 100_000
ARITHMETIC_PROBLEMS_TEST = 100_000

SYMBOLIC_PROBLEMS_TRAIN = 1_000_000
SYMBOLIC_PROBLEMS_VAL = 50_000
SYMBOLIC_PROBLEMS_TEST = 50_000

# ============================================================================
# Physics Simulation Constants
# ============================================================================

# PyBullet
GRAVITY = -9.81  # m/s^2
TIME_STEP = 1.0 / 240.0  # 240 Hz simulation
MAX_TIMESTEPS = 400  # Maximum timesteps per episode

# Object Properties
MIN_MASS = 0.1  # kg
MAX_MASS = 10.0  # kg
MIN_RADIUS = 0.1  # m
MAX_RADIUS = 0.5  # m

# Scene Bounds
SCENE_MIN_X = -5.0
SCENE_MAX_X = 5.0
SCENE_MIN_Y = -5.0
SCENE_MAX_Y = 5.0
SCENE_MIN_Z = 0.0
SCENE_MAX_Z = 10.0

# State Value Bounds (for output clamping and validation)
# Positions: [-5, 10] (scene bounds)
# Velocities: can spike to 200+ m/s during high-energy collisions
# Using ±500 as safe upper bound to accommodate physics while catching true explosions
MAX_STATE_VALUE = 500.0
MIN_STATE_VALUE = -500.0

# ============================================================================
# Ablation Study Constants
# ============================================================================

NUM_ABLATION_CONFIGS = 6  # Total ablation configurations

ABLATION_CONFIG_NAMES = [
    "Config 1: Full Model (All Stages)",
    "Config 2: Skip Counting",
    "Config 3: Skip Arithmetic",
    "Config 4: Skip Physics",
    "Config 5: Symbolic Only",
    "Config 6: Simultaneous Training"
]

# ============================================================================
# File Paths (relative to project root)
# ============================================================================

DEFAULT_CHECKPOINT_DIR = './checkpoints'
DEFAULT_DATA_DIR = './data'
DEFAULT_LOG_DIR = './logs'
DEFAULT_RESULTS_DIR = './results'

# Checkpoint Names
STAGE_1_CHECKPOINT = 'stage1_final.pt'
STAGE_2_CHECKPOINT = 'stage2_final.pt'
STAGE_3_CHECKPOINT = 'stage3_final.pt'
STAGE_4_CHECKPOINT = 'stage4_final.pt'

# Ablation Checkpoints
ABLATION_MODEL_A = 'model_a_full.pt'
ABLATION_MODEL_B = 'model_b_no_stage3.pt'
ABLATION_MODEL_C = 'model_c_symbolic_only.pt'

# ============================================================================
# Numerical Constants
# ============================================================================

# Epsilon for numerical stability
EPSILON = 1e-6
SMALL_NUMBER = 1e-8

# Infinity representations
LARGE_NUMBER = 1e10
NEG_INF = float('-inf')
POS_INF = float('inf')

# ============================================================================
# Attention Constants
# ============================================================================

# Attention patterns (for interpretability analysis)
ATTENTION_UNIFORM_THRESHOLD = 0.1  # Difference from uniform for classification
ATTENTION_SELECTIVE_THRESHOLD = 0.3  # Minimum difference for selective attention

# ============================================================================
# Performance Thresholds
# ============================================================================

# Expected accuracies (for validation)
EXPECTED_PHYSICS_ACCURACY = 0.85  # 85%
EXPECTED_COUNTING_ACCURACY = 0.90  # 90%
EXPECTED_ARITHMETIC_ACCURACY = 0.85  # 85%
EXPECTED_SYMBOLIC_ACCURACY = 0.80  # 80%

# Minimum acceptable accuracies (for early stopping)
MIN_ACCEPTABLE_ACCURACY = 0.50  # 50%
MIN_ZERO_SHOT_ACCURACY = 0.40  # 40%

# ============================================================================
# String Constants
# ============================================================================

# Task Names
TASK_PHYSICS = 'physics'
TASK_COUNTING = 'counting'
TASK_ARITHMETIC = 'arithmetic'
TASK_SYMBOLIC = 'symbolic'

TASK_NAMES = [TASK_PHYSICS, TASK_COUNTING, TASK_ARITHMETIC, TASK_SYMBOLIC]

# Batch Dictionary Keys
BATCH_KEY_OBJECT_STATES = 'object_states'
BATCH_KEY_OBJECT_MASK = 'object_mask'
BATCH_KEY_NEXT_STATES = 'next_states'
BATCH_KEY_DELTA_STATES = 'delta_states'  # next_states - current_states (position-invariant)
BATCH_KEY_TARGET_STATES = 'target_states'
BATCH_KEY_CURRENT_STATES = 'current_states'
BATCH_KEY_MASSES = 'masses'
BATCH_KEY_SCHEMA = 'schema'
BATCH_KEY_SCHEMA_NAME = 'schema_name'
BATCH_KEY_COUNTS = 'counts'
BATCH_KEY_RESULTS = 'results'
BATCH_KEY_NUMBERS = 'numbers'
BATCH_KEY_ANSWER = 'answer'
BATCH_KEY_ANSWERS = 'answers'  # Plural for symbolic task

# Output Dictionary Keys
OUTPUT_KEY_PREDICTED_STATES = 'predicted_next_states'
OUTPUT_KEY_PREDICTED_ANSWER = 'predicted_answer'
OUTPUT_KEY_NUMBER_EMBEDDINGS = 'number_embeddings'

# Loss Dictionary Keys
LOSS_KEY_PREDICTION = 'prediction'
LOSS_KEY_ENERGY = 'energy'
LOSS_KEY_MOMENTUM = 'momentum'
LOSS_KEY_COUNTING = 'counting'
LOSS_KEY_ARITHMETIC = 'arithmetic'
LOSS_KEY_ARITHMETIC_DIGIT = 'arithmetic_digit'
LOSS_KEY_ARITHMETIC_LENGTH = 'arithmetic_length'
LOSS_KEY_SYMBOLIC = 'symbolic'
LOSS_KEY_SYMBOLIC_DIGIT = 'symbolic_digit'
LOSS_KEY_SYMBOLIC_LENGTH = 'symbolic_length'
LOSS_KEY_AUXILIARY = 'auxiliary'
LOSS_KEY_CONTRASTIVE = 'contrastive'

# Metrics Dictionary Keys
METRICS_KEY_AUX_LOSS = 'aux_loss'
METRICS_KEY_CONTRAST_LOSS = 'contrast_loss'
METRICS_KEY_DIGIT_LOSS = 'digit_loss'
METRICS_KEY_LENGTH_LOSS = 'length_loss'
METRICS_KEY_TOTAL_LOSS = 'total_loss'

# Default/Fallback Values
DEFAULT_SCHEMA_NAME = 'unknown'
DEFAULT_SCHEMA_ID = 0
DEFAULT_BATCH_SIZE = 32

# Stage Names
STAGE_NAMES = [
    "Stage 1: Physics Understanding",
    "Stage 2: Counting (Quantity Abstraction)",
    "Stage 3: Arithmetic Operations (Grounded Math)",
    "Stage 4: Symbolic Transfer (Abstract Math)"
]

# Progress Messages
MSG_STAGE_COMPLETE = " {stage} complete: {checkpoint}"
MSG_STAGE_SKIPPED = " {stage} already complete. Use --force to retrain."
MSG_STAGE_REQUIRED = " Must complete {stage} first!"
MSG_TRAINING_START = "=" * 70 + "\nTRAINING {stage}\n" + "=" * 70

# Error Messages
ERR_INVALID_OPERATION = "Invalid operation: {op}. Must be one of {valid_ops}"
ERR_INVALID_STAGE = "Invalid stage: {stage}. Must be 1-4"
ERR_CHECKPOINT_NOT_FOUND = "Checkpoint not found: {path}"
ERR_CUDA_OOM = "CUDA out of memory. Try reducing batch_size or enabling gradient_checkpointing"
ERR_INVALID_NUMBER_RANGE = "Number {num} out of valid range [0, {max_val}]"

# ============================================================================
# Validation Constants
# ============================================================================

# Input validation ranges
VALID_BATCH_SIZE_RANGE = (1, 512)
VALID_LEARNING_RATE_RANGE = (1e-6, 1e-2)
VALID_HIDDEN_DIM_RANGE = (64, 2048)
VALID_NUM_LAYERS_RANGE = (1, 48)
VALID_NUM_HEADS_RANGE = (1, 64)

# ============================================================================
# Helper Functions
# ============================================================================

def validate_operation(operation: str) -> None:
    """Validate operation name."""
    if operation not in OPERATION_NAMES:
        raise ValueError(
            ERR_INVALID_OPERATION.format(
                op=operation,
                valid_ops=OPERATION_NAMES
            )
        )

def validate_stage(stage: int) -> None:
    """Validate stage number."""
    if stage not in [1, 2, 3, 4]:
        raise ValueError(ERR_INVALID_STAGE.format(stage=stage))

def validate_number_range(number: int, max_val: int = MAX_NUMBER_VALUE) -> None:
    """Validate number is in acceptable range."""
    if number < 0 or number > max_val:
        raise ValueError(
            ERR_INVALID_NUMBER_RANGE.format(num=number, max_val=max_val)
        )

def get_stage_name(stage: int) -> str:
    """Get human-readable stage name."""
    validate_stage(stage)
    return STAGE_NAMES[stage - 1]

def get_operation_id(operation: str) -> int:
    """Get operation ID from name."""
    validate_operation(operation)
    return OPERATION_IDS[operation]

def get_operation_name(operation_id: int) -> str:
    """Get operation name from ID."""
    if operation_id < 0 or operation_id >= NUM_OPERATIONS:
        raise ValueError(f"Invalid operation ID: {operation_id}")
    return OPERATION_NAMES[operation_id]

# ============================================================================
# Export All Constants
# ============================================================================

__all__ = [
    # Compositional Output
    'DIGIT_CLASSES',
    'PAD_TOKEN',
    'MAX_DIGITS',
    
    # Number Encoding
    'FIXED_EMBEDDING_RANGE',
    'MAX_NUMBER_VALUE',
    
    # Dimensions
    'STATE_DIM',
    'NUM_PHYSICS_SCHEMAS',
    'NUM_OPERATIONS',
    
    # State Vector Indices (21D Isaac Sim)
    'POSITION_START_IDX',
    'POSITION_END_IDX',
    'VELOCITY_START_IDX',
    'VELOCITY_END_IDX',
    'ORIENTATION_START_IDX',
    'ORIENTATION_END_IDX',
    'ANGULAR_VELOCITY_START_IDX',
    'ANGULAR_VELOCITY_END_IDX',
    'MASS_IDX',
    'SIZE_IDX',
    'SIZE_START_IDX',
    'SIZE_END_IDX',
    'FRICTION_IDX',
    'RESTITUTION_IDX',
    'IS_IN_CONTACT_IDX',
    'CONTACT_FORCE_START_IDX',
    'CONTACT_FORCE_END_IDX',
    'LINEAR_DAMPING_IDX',
    'ANGULAR_DAMPING_IDX',
    
    # Task Names
    'TASK_PHYSICS',
    'TASK_COUNTING',
    'TASK_ARITHMETIC',
    'TASK_SYMBOLIC',
    'TASK_NAMES',
    
    # Batch Keys
    'BATCH_KEY_OBJECT_STATES',
    'BATCH_KEY_OBJECT_MASK',
    'BATCH_KEY_NEXT_STATES',
    'BATCH_KEY_DELTA_STATES',
    'BATCH_KEY_TARGET_STATES',
    'BATCH_KEY_CURRENT_STATES',
    'BATCH_KEY_MASSES',
    'BATCH_KEY_SCHEMA',
    'BATCH_KEY_SCHEMA_NAME',
    'BATCH_KEY_COUNTS',
    'BATCH_KEY_RESULTS',
    'BATCH_KEY_NUMBERS',
    'BATCH_KEY_ANSWER',
    'BATCH_KEY_ANSWERS',
    
    # Output Keys
    'OUTPUT_KEY_PREDICTED_STATES',
    'OUTPUT_KEY_PREDICTED_ANSWER',
    'OUTPUT_KEY_NUMBER_EMBEDDINGS',
    
    # Loss Keys
    'LOSS_KEY_PREDICTION',
    'LOSS_KEY_ENERGY',
    'LOSS_KEY_MOMENTUM',
    'LOSS_KEY_COUNTING',
    'LOSS_KEY_ARITHMETIC',
    'LOSS_KEY_ARITHMETIC_DIGIT',
    'LOSS_KEY_ARITHMETIC_LENGTH',
    'LOSS_KEY_SYMBOLIC',
    'LOSS_KEY_SYMBOLIC_DIGIT',
    'LOSS_KEY_SYMBOLIC_LENGTH',
    'LOSS_KEY_AUXILIARY',
    'LOSS_KEY_CONTRASTIVE',
    
    # Metrics Keys
    'METRICS_KEY_AUX_LOSS',
    'METRICS_KEY_CONTRAST_LOSS',
    'METRICS_KEY_DIGIT_LOSS',
    'METRICS_KEY_LENGTH_LOSS',
    'METRICS_KEY_TOTAL_LOSS',
    
    # Default Values
    'DEFAULT_SCHEMA_NAME',
    'DEFAULT_SCHEMA_ID',
    'DEFAULT_BATCH_SIZE',
    
    # Operations
    'OPERATION_NAMES',
    'OPERATION_IDS',
    
    # Training
    'MAX_GRAD_NORM',
    'LOG_INTERVAL',
    'CHECKPOINT_INTERVAL',
    'VALIDATION_INTERVAL',
    
    # Loss Weights
    'PHYSICS_PREDICTION_WEIGHT',
    'ENERGY_CONSERVATION_WEIGHT',
    'MOMENTUM_CONSERVATION_WEIGHT',
    'COUNTING_WEIGHT',
    'ARITHMETIC_WEIGHT',
    'SYMBOLIC_WEIGHT',
    'DIGIT_LOSS_WEIGHT',
    'LENGTH_LOSS_WEIGHT',
    
    # Data Ranges
    'MIN_COUNT',
    'MAX_COUNT',
    'ARITHMETIC_EASY_MAX',
    'ARITHMETIC_MEDIUM_MAX',
    'ARITHMETIC_HARD_MAX',
    'SYMBOLIC_EASY_MAX',
    'SYMBOLIC_MEDIUM_MAX',
    'SYMBOLIC_HARD_MAX',
    
    # Zero-Shot Ranges
    'ZERO_SHOT_RANGE_1',
    'ZERO_SHOT_RANGE_2',
    'ZERO_SHOT_RANGE_3',
    'ZERO_SHOT_RANGE_4',
    'ZERO_SHOT_RANGE_5',
    
    # Physics
    'GRAVITY',
    'TIME_STEP',
    'MAX_TIMESTEPS',
    
    # Paths
    'DEFAULT_CHECKPOINT_DIR',
    'DEFAULT_DATA_DIR',
    'DEFAULT_LOG_DIR',
    'DEFAULT_RESULTS_DIR',
    
    # Checkpoints
    'STAGE_1_CHECKPOINT',
    'STAGE_2_CHECKPOINT',
    'STAGE_3_CHECKPOINT',
    'STAGE_4_CHECKPOINT',
    
    # Numerical
    'EPSILON',
    'SMALL_NUMBER',
    
    # Messages
    'STAGE_NAMES',
    'MSG_STAGE_COMPLETE',
    'MSG_STAGE_SKIPPED',
    'MSG_STAGE_REQUIRED',
    'MSG_TRAINING_START',
    
    # Errors
    'ERR_INVALID_OPERATION',
    'ERR_INVALID_STAGE',
    'ERR_CHECKPOINT_NOT_FOUND',
    'ERR_CUDA_OOM',
    'ERR_INVALID_NUMBER_RANGE',
    
    # Validation
    'VALID_BATCH_SIZE_RANGE',
    'VALID_LEARNING_RATE_RANGE',
    'VALID_HIDDEN_DIM_RANGE',
    'VALID_NUM_LAYERS_RANGE',
    'VALID_NUM_HEADS_RANGE',
    
    # Helper Functions
    'validate_operation',
    'validate_stage',
    'validate_number_range',
    'get_stage_name',
    'get_operation_id',
    'get_operation_name',
]
