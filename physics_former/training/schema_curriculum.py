"""
Schema Curriculum Definitions for Physics-LLM Training

Single source of truth for all Isaac Sim schema groups and curriculum ordering.
Import this module in datasets and training pipelines to ensure consistency.

37 Isaac Sim Physics Schemas in 11 Groups + 2 Advanced Training Modes:
- Group 1: Gravity & Free Fall (4)
- Group 2: Collisions (6)
- Group 3: Stacking (5)
- Group 4: Rolling & Sliding (4)
- Group 5: Projectiles (4)
- Group 6: Domino (1)
- Group 7: Scattering (4)
- Group 8: Physics Variety (5)
- Group 9: Articulated (1)
- Group 10: Rotation (1)
- Group 11: Complex Dynamics (2)
- Level 12: CAUSAL TRAINING - Object dropout with intervention loss (uses all schemas)
- Level 13: COUNTERFACTUAL TRAINING - "What if" reasoning with contrastive learning (uses all schemas)
"""

from typing import List, Dict

NUM_SCHEMA_GROUPS = 11
NUM_SCHEMAS = 36  # 37 - 1 (excluded: isaac_hinged_door)

# Excluded schemas (max delta > 50 causes gradient instability):
# - isaac_hinged_door: articulated object with joint constraints
# - orbit_elliptical: extreme orbital velocities (max=122)
# - gravitational_slingshot: extreme gravity assist velocities (max=78)
# - attraction_repulsion: extreme force field velocities (max=51)
# Note: orbit_circular is OK (max=6.0)
EXCLUDED_SCHEMAS = {
    'isaac_hinged_door',
    'orbit_elliptical',
    'gravitational_slingshot',
    'attraction_repulsion'
}

ISAAC_CURRICULUM_ORDER = [
    # GROUP 1: Gravity & Free Fall (4 schemas)
    "isaac_multi_drop", "isaac_varied_mass_drop",
    "isaac_varied_height_drop", "isaac_simultaneous_drop",
    # GROUP 2: Collisions (6 schemas)
    "isaac_head_on_collision", "isaac_angled_collision", "isaac_multi_body_collision",
    "isaac_chain_collision", "isaac_cluster_collision", "isaac_asymmetric_collision",
    # GROUP 3: Stacking (5 schemas)
    "isaac_simple_stack", "isaac_tall_stack", "isaac_pyramid_stack",
    "isaac_unstable_stack", "isaac_offset_stack",
    # GROUP 4: Rolling & Sliding (4 schemas)
    "isaac_cube_slide", "isaac_ramp_roll",
    "isaac_ramp_slide", "isaac_friction_compare",
    # GROUP 5: Projectiles (4 schemas)
    "isaac_horizontal_throw", "isaac_angled_throw", "isaac_lob_throw", "isaac_multi_projectile",
    # GROUP 6: Domino (1 schema)
    "isaac_domino_line",
    # GROUP 7: Scattering (4 schemas)
    "isaac_explosion_scatter", "isaac_impact_scatter", "isaac_funnel_scatter",
    "isaac_directed_scatter",
    # GROUP 8: Physics Variety (5 schemas)
    "isaac_obstacle_drop", "isaac_wedge_deflect", "isaac_block_stacking",
    "isaac_projectile_trajectory", "isaac_friction_variety",
    # GROUP 9: Articulated (1 schema)
    "isaac_hinged_door",
    # GROUP 10: Rotation (1 schema)
    "isaac_angular_momentum",
    # GROUP 11: Complex Dynamics (2 schemas)
    "isaac_billiard_break", "isaac_bowling_strike",
]

ISAAC_SCHEMA_GROUPS = [
    ISAAC_CURRICULUM_ORDER[0:4],   # Group 1: Gravity & Free Fall (4 schemas)
    ISAAC_CURRICULUM_ORDER[4:10],  # Group 2: Collisions (6 schemas)
    ISAAC_CURRICULUM_ORDER[10:15], # Group 3: Stacking (5 schemas)
    ISAAC_CURRICULUM_ORDER[15:19], # Group 4: Rolling & Sliding (4 schemas)
    ISAAC_CURRICULUM_ORDER[19:23], # Group 5: Projectiles (4 schemas)
    ISAAC_CURRICULUM_ORDER[23:24], # Group 6: Domino (1 schema)
    ISAAC_CURRICULUM_ORDER[24:28], # Group 7: Scattering (4 schemas)
    ISAAC_CURRICULUM_ORDER[28:33], # Group 8: Physics Variety (5 schemas)
    ISAAC_CURRICULUM_ORDER[33:34], # Group 9: Articulated (1 schema)
    ISAAC_CURRICULUM_ORDER[34:35], # Group 10: Rotation (1 schema)
    ISAAC_CURRICULUM_ORDER[35:37], # Group 11: Complex Dynamics (2 schemas)
]

GROUP_NAMES = {
    1: "Gravity & Free Fall",
    2: "Collisions",
    3: "Stacking",
    4: "Rolling & Sliding",
    5: "Projectiles",
    6: "Domino",
    7: "Scattering",
    8: "Physics Variety",
    9: "Articulated",
    10: "Rotation",
    11: "Complex Dynamics",
    12: "Causal Training",
    13: "Counterfactual Training",
}

# Total curriculum levels (11 schema groups + 2 advanced training modes)
NUM_CURRICULUM_LEVELS = 13


def get_schemas_for_level(level: int) -> List[str]:
    """Get all schemas up to and including the given curriculum level.
    
    Levels 1-11: Progressive schema groups
    Levels 12-13: All schemas (causal/counterfactual training modes)
    
    Excludes articulated schemas (e.g., hinged_door) which have joint constraints.
    """
    schemas = []
    for group_idx in range(min(level, NUM_SCHEMA_GROUPS)):
        schemas.extend(ISAAC_SCHEMA_GROUPS[group_idx])
    # Filter out excluded schemas
    return [s for s in schemas if s not in EXCLUDED_SCHEMAS]


def get_schema_groups_for_level(level: int) -> List[List[str]]:
    """Get schema groups up to and including the given curriculum level.
    
    Excludes articulated schemas (e.g., hinged_door) which have joint constraints.
    """
    groups = ISAAC_SCHEMA_GROUPS[:min(level, NUM_SCHEMA_GROUPS)]
    # Filter out excluded schemas from each group
    return [[s for s in group if s not in EXCLUDED_SCHEMAS] for group in groups]


def get_group_for_schema(schema_name: str) -> int:
    """Get the group number (1-11) for a given schema name."""
    for group_idx, group in enumerate(ISAAC_SCHEMA_GROUPS):
        if schema_name in group:
            return group_idx + 1
    return -1


def print_curriculum_summary(level: int = NUM_SCHEMA_GROUPS):
    """Print a summary of schemas at the given curriculum level."""
    schemas = get_schemas_for_level(level)
    groups = get_schema_groups_for_level(level)
    
    print(f"[CURRICULUM] Loading {len(schemas)} schemas from {len(groups)} groups:")
    for i, group in enumerate(groups):
        print(f"  Group {i+1} ({GROUP_NAMES[i+1]}): {len(group)} schemas")
    print()
