"""
Copyright (c) 2026 Anonymous. All rights reserved.
Author: Anonymous

PROPRIETARY AND CONFIDENTIAL. This software is provided for academic review
and research purposes only. Unauthorized copying, modification, distribution,
or use of this software, via any medium, is strictly prohibited without prior
written permission from Anonymous.
"""

"""
Schema Registry for Isaac Sim

Centralized registry of all Isaac Sim physics schemas with their configurations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any

from .state_extraction import ObjectProperties, StateExtractor, SHAPE_TYPES


@dataclass
class SchemaConfig:
    name: str
    task_name: str
    description: str
    max_steps: int = 200
    num_objects_range: tuple = (2, 10)
    object_extractors: List[Dict[str, Any]] = field(default_factory=list)
    custom_extractor: Optional[Callable] = None
    
    def create_state_extractor(self) -> StateExtractor:
        extractor = StateExtractor()
        for obj_config in self.object_extractors:
            extractor.add_object_source(**obj_config)
        return extractor


ROBOT_PROPERTIES = ObjectProperties(
    mass=10.0,
    radius=0.1,
    color=(0.3, 0.3, 0.8),
    shape_type=SHAPE_TYPES['robot'],
    is_static=False,
    friction=0.5
)

CABINET_PROPERTIES = ObjectProperties(
    mass=50.0,
    radius=0.5,
    color=(0.6, 0.4, 0.2),
    shape_type=SHAPE_TYPES['box'],
    is_static=True,
    friction=0.8
)

BALL_PROPERTIES = ObjectProperties(
    mass=0.5,
    radius=0.1,
    color=(0.9, 0.6, 0.2),
    shape_type=SHAPE_TYPES['sphere'],
    is_static=False,
    friction=0.5
)

HAND_PROPERTIES = ObjectProperties(
    mass=2.0,
    radius=0.15,
    color=(0.8, 0.7, 0.6),
    shape_type=SHAPE_TYPES['articulation'],
    is_static=False,
    friction=0.8
)

OBJECT_PROPERTIES = ObjectProperties(
    mass=1.0,
    radius=0.05,
    color=(0.8, 0.2, 0.2),
    shape_type=SHAPE_TYPES['sphere'],
    is_static=False,
    friction=0.5
)

DEFORMABLE_PROPERTIES = ObjectProperties(
    mass=0.3,
    radius=0.1,
    color=(0.2, 0.8, 0.2),
    shape_type=SHAPE_TYPES['mesh'],
    is_static=False,
    friction=0.6
)

PLATFORM_PROPERTIES = ObjectProperties(
    mass=5.0,
    radius=0.3,
    color=(0.5, 0.5, 0.5),
    shape_type=SHAPE_TYPES['box'],
    is_static=False,
    friction=0.7
)

CONTAINER_PROPERTIES = ObjectProperties(
    mass=10.0,
    radius=0.3,
    color=(0.4, 0.4, 0.4),
    shape_type=SHAPE_TYPES['box'],
    is_static=True,
    friction=0.6
)

SMALL_BALL_PROPERTIES = ObjectProperties(
    mass=0.1,
    radius=0.03,
    color=(0.9, 0.3, 0.3),
    shape_type=SHAPE_TYPES['sphere'],
    is_static=False,
    friction=0.4
)

DOMINO_PROPERTIES = ObjectProperties(
    mass=0.2,
    radius=0.04,
    color=(0.1, 0.1, 0.1),
    shape_type=SHAPE_TYPES['box'],
    is_static=False,
    friction=0.7,
    dimensions=(0.01, 0.04, 0.08),
    restitution=0.3
)

BLOCK_PROPERTIES = ObjectProperties(
    mass=0.5,
    radius=0.05,
    color=(0.8, 0.6, 0.2),
    shape_type=SHAPE_TYPES['box'],
    is_static=False,
    friction=0.6
)

RAMP_PROPERTIES = ObjectProperties(
    mass=100.0,
    radius=0.5,
    color=(0.5, 0.5, 0.6),
    shape_type=SHAPE_TYPES['box'],
    is_static=True,
    friction=0.3
)

PENDULUM_PROPERTIES = ObjectProperties(
    mass=1.0,
    radius=0.08,
    color=(0.7, 0.2, 0.7),
    shape_type=SHAPE_TYPES['sphere'],
    is_static=False,
    friction=0.1
)


ISAAC_SCHEMAS: Dict[str, SchemaConfig] = {
    'isaac_cabinet_manipulation': SchemaConfig(
        name='isaac_cabinet_manipulation',
        task_name='FrankaCabinet',
        description='Franka robot arm opens cabinet drawer',
        max_steps=200,
        num_objects_range=(2, 4),
        object_extractors=[
            {
                'name': 'franka_robot',
                'source_type': 'articulation_view',
                'view_attr': '_frankas',
                'properties': ROBOT_PROPERTIES,
            },
            {
                'name': 'cabinet',
                'source_type': 'articulation_view',
                'view_attr': '_cabinets',
                'properties': CABINET_PROPERTIES,
            },
        ]
    ),
    
    'isaac_deformable_manipulation': SchemaConfig(
        name='isaac_deformable_manipulation',
        task_name='FrankaDeformable',
        description='Franka robot manipulates deformable tube',
        max_steps=200,
        num_objects_range=(2, 3),
        object_extractors=[
            {
                'name': 'franka_robot',
                'source_type': 'articulation_view',
                'view_attr': '_frankas',
                'properties': ROBOT_PROPERTIES,
            },
            {
                'name': 'deformable_object',
                'source_type': 'rigid_prim_view',
                'view_attr': '_objects',
                'properties': DEFORMABLE_PROPERTIES,
            },
        ]
    ),
    
    'isaac_hand_rotation': SchemaConfig(
        name='isaac_hand_rotation',
        task_name='ShadowHand',
        description='Shadow hand rotates object in-hand',
        max_steps=300,
        num_objects_range=(2, 2),
        object_extractors=[
            {
                'name': 'shadow_hand',
                'source_type': 'articulation_view',
                'view_attr': '_hands',
                'properties': HAND_PROPERTIES,
            },
            {
                'name': 'manipulated_object',
                'source_type': 'tensor',
                'view_attr': None,
                'properties': OBJECT_PROPERTIES,
                'position_attr': 'object_pos',
                'velocity_attr': 'object_linvel',
            },
        ]
    ),
    
    'isaac_hand_grasping': SchemaConfig(
        name='isaac_hand_grasping',
        task_name='AllegroHand',
        description='Allegro hand grasps and manipulates object',
        max_steps=300,
        num_objects_range=(2, 2),
        object_extractors=[
            {
                'name': 'allegro_hand',
                'source_type': 'articulation_view',
                'view_attr': '_hands',
                'properties': HAND_PROPERTIES,
            },
            {
                'name': 'grasped_object',
                'source_type': 'tensor',
                'view_attr': None,
                'properties': OBJECT_PROPERTIES,
                'position_attr': 'object_pos',
                'velocity_attr': 'object_linvel',
            },
        ]
    ),
    
    'isaac_ball_balance': SchemaConfig(
        name='isaac_ball_balance',
        task_name='BallBalance',
        description='Balance ball on tilting platform',
        max_steps=200,
        num_objects_range=(2, 2),
        object_extractors=[
            {
                'name': 'balance_platform',
                'source_type': 'articulation_view',
                'view_attr': '_balance_bots',
                'properties': PLATFORM_PROPERTIES,
            },
            {
                'name': 'ball',
                'source_type': 'rigid_prim_view',
                'view_attr': '_balls',
                'properties': BALL_PROPERTIES,
            },
        ]
    ),
    
    'isaac_cartpole': SchemaConfig(
        name='isaac_cartpole',
        task_name='Cartpole',
        description='Classic cartpole balancing task',
        max_steps=500,
        num_objects_range=(2, 2),
        object_extractors=[
            {
                'name': 'cart',
                'source_type': 'articulation_view',
                'view_attr': '_cartpoles',
                'properties': ObjectProperties(
                    mass=1.0, radius=0.1, color=(0.4, 0.4, 0.8),
                    shape_type=SHAPE_TYPES['box'], is_static=False, friction=0.5
                ),
            },
        ]
    ),
    
    'isaac_ant_locomotion': SchemaConfig(
        name='isaac_ant_locomotion',
        task_name='Ant',
        description='Ant robot locomotion',
        max_steps=1000,
        num_objects_range=(1, 1),
        object_extractors=[
            {
                'name': 'ant',
                'source_type': 'articulation_view',
                'view_attr': '_ants',
                'properties': ObjectProperties(
                    mass=5.0, radius=0.3, color=(0.6, 0.3, 0.1),
                    shape_type=SHAPE_TYPES['articulation'], is_static=False, friction=0.8
                ),
            },
        ]
    ),
    
    'isaac_humanoid': SchemaConfig(
        name='isaac_humanoid',
        task_name='Humanoid',
        description='Humanoid robot locomotion',
        max_steps=1000,
        num_objects_range=(1, 1),
        object_extractors=[
            {
                'name': 'humanoid',
                'source_type': 'articulation_view',
                'view_attr': '_humanoids',
                'properties': ObjectProperties(
                    mass=50.0, radius=0.5, color=(0.8, 0.6, 0.5),
                    shape_type=SHAPE_TYPES['articulation'], is_static=False, friction=0.8
                ),
            },
        ]
    ),
    
    'isaac_quadcopter': SchemaConfig(
        name='isaac_quadcopter',
        task_name='Quadcopter',
        description='Quadcopter flight control',
        max_steps=500,
        num_objects_range=(1, 1),
        object_extractors=[
            {
                'name': 'quadcopter',
                'source_type': 'articulation_view',
                'view_attr': '_copters',
                'properties': ObjectProperties(
                    mass=1.0, radius=0.2, color=(0.2, 0.2, 0.2),
                    shape_type=SHAPE_TYPES['mesh'], is_static=False, friction=0.3
                ),
            },
        ]
    ),
    
    'isaac_anymal': SchemaConfig(
        name='isaac_anymal',
        task_name='Anymal',
        description='ANYmal quadruped locomotion',
        max_steps=1000,
        num_objects_range=(1, 1),
        object_extractors=[
            {
                'name': 'anymal',
                'source_type': 'articulation_view',
                'view_attr': '_anymals',
                'properties': ObjectProperties(
                    mass=30.0, radius=0.4, color=(0.3, 0.3, 0.3),
                    shape_type=SHAPE_TYPES['articulation'], is_static=False, friction=0.9
                ),
            },
        ]
    ),
    
    'isaac_container_fill_spill': SchemaConfig(
        name='isaac_container_fill_spill',
        task_name='ContainerFillSpill',
        description='Objects fall into container, pile up, and spill over edges',
        max_steps=300,
        num_objects_range=(10, 20),
        object_extractors=[
            {
                'name': 'container',
                'source_type': 'rigid_prim_view',
                'view_attr': '_container',
                'properties': CONTAINER_PROPERTIES,
            },
            {
                'name': 'falling_objects',
                'source_type': 'rigid_prim_view',
                'view_attr': '_falling_objects',
                'properties': SMALL_BALL_PROPERTIES,
            },
        ]
    ),
    
    'isaac_domino_topple': SchemaConfig(
        name='isaac_domino_topple',
        task_name='DominoTopple',
        description='Chain reaction of falling dominoes transferring momentum',
        max_steps=400,
        num_objects_range=(10, 20),
        object_extractors=[
            {
                'name': 'dominoes',
                'source_type': 'rigid_prim_view',
                'view_attr': '_dominoes',
                'properties': DOMINO_PROPERTIES,
            },
        ]
    ),
    
    'isaac_domino_line': SchemaConfig(
        name='isaac_domino_line',
        task_name='DominoTopple',
        description='Line of dominoes that topple in sequence when first is pushed',
        max_steps=400,
        num_objects_range=(10, 20),
        object_extractors=[
            {
                'name': 'dominoes',
                'source_type': 'rigid_prim_view',
                'view_attr': '_dominoes',
                'properties': DOMINO_PROPERTIES,
            },
        ]
    ),
    
    'isaac_domino_curve': SchemaConfig(
        name='isaac_domino_curve',
        task_name='DominoTopple',
        description='Curved arrangement of dominoes that topple in sequence',
        max_steps=400,
        num_objects_range=(10, 20),
        object_extractors=[
            {
                'name': 'dominoes',
                'source_type': 'rigid_prim_view',
                'view_attr': '_dominoes',
                'properties': DOMINO_PROPERTIES,
            },
        ]
    ),
    
    'isaac_cascade_fall': SchemaConfig(
        name='isaac_cascade_fall',
        task_name='DominoTopple',
        description='Cascading fall of dominoes arranged in tiers or levels',
        max_steps=400,
        num_objects_range=(10, 20),
        object_extractors=[
            {
                'name': 'dominoes',
                'source_type': 'rigid_prim_view',
                'view_attr': '_dominoes',
                'properties': DOMINO_PROPERTIES,
            },
        ]
    ),
    
    'isaac_trigger_chain': SchemaConfig(
        name='isaac_trigger_chain',
        task_name='DominoTopple',
        description='Chain reaction triggered by initial domino push',
        max_steps=400,
        num_objects_range=(10, 20),
        object_extractors=[
            {
                'name': 'dominoes',
                'source_type': 'rigid_prim_view',
                'view_attr': '_dominoes',
                'properties': DOMINO_PROPERTIES,
            },
        ]
    ),
    
    'isaac_block_stacking': SchemaConfig(
        name='isaac_block_stacking',
        task_name='BlockStacking',
        description='Blocks stacked and tested for stability under perturbation',
        max_steps=300,
        num_objects_range=(5, 12),
        object_extractors=[
            {
                'name': 'blocks',
                'source_type': 'rigid_prim_view',
                'view_attr': '_blocks',
                'properties': BLOCK_PROPERTIES,
            },
            {
                'name': 'ground_plane',
                'source_type': 'rigid_prim_view',
                'view_attr': '_ground',
                'properties': ObjectProperties(
                    mass=1000.0, radius=5.0, color=(0.3, 0.3, 0.3),
                    shape_type=SHAPE_TYPES['box'], is_static=True, friction=0.8
                ),
            },
        ]
    ),
    
    'isaac_ramp_rolling': SchemaConfig(
        name='isaac_ramp_rolling',
        task_name='RampRolling',
        description='Objects roll/slide down inclined ramp with varying friction',
        max_steps=200,
        num_objects_range=(2, 5),
        object_extractors=[
            {
                'name': 'ramp',
                'source_type': 'rigid_prim_view',
                'view_attr': '_ramp',
                'properties': RAMP_PROPERTIES,
            },
            {
                'name': 'rolling_objects',
                'source_type': 'rigid_prim_view',
                'view_attr': '_rolling_objects',
                'properties': BALL_PROPERTIES,
            },
        ]
    ),
    
    'isaac_pendulum_chain': SchemaConfig(
        name='isaac_pendulum_chain',
        task_name='PendulumChain',
        description='Coupled pendulums exhibiting energy transfer and oscillation',
        max_steps=500,
        num_objects_range=(3, 7),
        object_extractors=[
            {
                'name': 'pendulum_bobs',
                'source_type': 'rigid_prim_view',
                'view_attr': '_pendulum_bobs',
                'properties': PENDULUM_PROPERTIES,
            },
            {
                'name': 'pendulum_frame',
                'source_type': 'rigid_prim_view',
                'view_attr': '_frame',
                'properties': ObjectProperties(
                    mass=50.0, radius=0.5, color=(0.2, 0.2, 0.2),
                    shape_type=SHAPE_TYPES['box'], is_static=True, friction=0.5
                ),
            },
        ]
    ),
    
    'isaac_newtons_cradle': SchemaConfig(
        name='isaac_newtons_cradle',
        task_name='NewtonsCradle',
        description='Classic momentum and energy conservation demonstration',
        max_steps=400,
        num_objects_range=(5, 7),
        object_extractors=[
            {
                'name': 'cradle_balls',
                'source_type': 'rigid_prim_view',
                'view_attr': '_cradle_balls',
                'properties': ObjectProperties(
                    mass=1.0, radius=0.05, color=(0.8, 0.8, 0.8),
                    shape_type=SHAPE_TYPES['sphere'], is_static=False, friction=0.1
                ),
            },
            {
                'name': 'cradle_frame',
                'source_type': 'rigid_prim_view',
                'view_attr': '_frame',
                'properties': ObjectProperties(
                    mass=20.0, radius=0.3, color=(0.1, 0.1, 0.1),
                    shape_type=SHAPE_TYPES['box'], is_static=True, friction=0.5
                ),
            },
        ]
    ),
    
    'isaac_multi_ball_collision': SchemaConfig(
        name='isaac_multi_ball_collision',
        task_name='MultiBallCollision',
        description='Multiple balls colliding on a surface like billiards',
        max_steps=300,
        num_objects_range=(5, 16),
        object_extractors=[
            {
                'name': 'balls',
                'source_type': 'rigid_prim_view',
                'view_attr': '_balls',
                'properties': ObjectProperties(
                    mass=0.3, radius=0.03, color=(0.2, 0.5, 0.8),
                    shape_type=SHAPE_TYPES['sphere'], is_static=False, friction=0.2
                ),
            },
            {
                'name': 'table',
                'source_type': 'rigid_prim_view',
                'view_attr': '_table',
                'properties': ObjectProperties(
                    mass=100.0, radius=1.0, color=(0.1, 0.5, 0.2),
                    shape_type=SHAPE_TYPES['box'], is_static=True, friction=0.3
                ),
            },
        ]
    ),
    
    'isaac_projectile_trajectory': SchemaConfig(
        name='isaac_projectile_trajectory',
        task_name='ProjectileTrajectory',
        description='Projectile motion with varying launch angles and velocities',
        max_steps=200,
        num_objects_range=(1, 3),
        object_extractors=[
            {
                'name': 'projectile',
                'source_type': 'rigid_prim_view',
                'view_attr': '_projectiles',
                'properties': ObjectProperties(
                    mass=0.5, radius=0.05, color=(1.0, 0.5, 0.0),
                    shape_type=SHAPE_TYPES['sphere'], is_static=False, friction=0.3
                ),
            },
            {
                'name': 'launcher',
                'source_type': 'rigid_prim_view',
                'view_attr': '_launcher',
                'properties': ObjectProperties(
                    mass=10.0, radius=0.2, color=(0.3, 0.3, 0.3),
                    shape_type=SHAPE_TYPES['box'], is_static=True, friction=0.5
                ),
            },
        ]
    ),
    
    'isaac_friction_variety': SchemaConfig(
        name='isaac_friction_variety',
        task_name='FrictionVariety',
        description='Objects sliding on surfaces with different friction coefficients',
        max_steps=200,
        num_objects_range=(4, 8),
        object_extractors=[
            {
                'name': 'sliding_objects',
                'source_type': 'rigid_prim_view',
                'view_attr': '_sliders',
                'properties': ObjectProperties(
                    mass=1.0, radius=0.05, color=(0.6, 0.4, 0.2),
                    shape_type=SHAPE_TYPES['box'], is_static=False, friction=0.5
                ),
            },
            {
                'name': 'friction_surfaces',
                'source_type': 'rigid_prim_view',
                'view_attr': '_surfaces',
                'properties': ObjectProperties(
                    mass=100.0, radius=0.5, color=(0.5, 0.5, 0.5),
                    shape_type=SHAPE_TYPES['box'], is_static=True, friction=0.5
                ),
            },
        ]
    ),
    
    'isaac_wall_collision': SchemaConfig(
        name='isaac_wall_collision',
        task_name='WallCollision',
        description='High-velocity ball strikes massive stationary cube horizontally',
        max_steps=200,
        num_objects_range=(2, 2),
        object_extractors=[
            {
                'name': 'massive_cube',
                'source_type': 'rigid_prim_view',
                'view_attr': '_cube',
                'properties': ObjectProperties(
                    mass=10000.0, radius=0.15, color=(0.2, 0.2, 0.2),
                    shape_type=SHAPE_TYPES['box'], is_static=False, friction=0.5
                ),
            },
            {
                'name': 'projectile_ball',
                'source_type': 'rigid_prim_view',
                'view_attr': '_ball',
                'properties': ObjectProperties(
                    mass=1.0, radius=0.05, color=(1.0, 0.2, 0.2),
                    shape_type=SHAPE_TYPES['sphere'], is_static=False, friction=0.4
                ),
            },
        ]
    ),
}


def get_schema_config(schema_name: str) -> Optional[SchemaConfig]:
    return ISAAC_SCHEMAS.get(schema_name)


def get_task_to_schema_mapping() -> Dict[str, str]:
    return {config.task_name: name for name, config in ISAAC_SCHEMAS.items()}


def register_schema(config: SchemaConfig):
    ISAAC_SCHEMAS[config.name] = config


def list_available_schemas() -> List[str]:
    return list(ISAAC_SCHEMAS.keys())


def list_available_tasks() -> List[str]:
    return [config.task_name for config in ISAAC_SCHEMAS.values()]
