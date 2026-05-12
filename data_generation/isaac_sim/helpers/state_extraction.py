"""
Copyright (c) 2026 Style Machine LLC. All rights reserved.

PROPRIETARY AND CONFIDENTIAL. This software is provided for academic review
and research purposes only. Unauthorized copying, modification, distribution,
or use of this software, via any medium, is strictly prohibited without prior
written permission from Style Machine LLC.
"""

"""
State Extraction Helpers for Isaac Sim

Provides unified state extraction from Isaac Sim environments for physics training.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

MAX_OBJECTS = 20
STATE_DIM = 35

STATE_INDICES = {
    'position': slice(0, 3),
    'linear_velocity': slice(3, 6),
    'orientation': slice(6, 10),
    'angular_velocity': slice(10, 13),
    'mass': 13,
    'radius': 14,
    'color': slice(15, 18),
    'shape_type': 18,
    'is_static': 19,
    'friction': 20,
    'is_active': 21,
    'inertia': 22,
    'bbox_min': 23,
    'bbox_max': 24,
    'dimensions': slice(25, 28),
    'force': slice(28, 31),
    'torque': slice(31, 34),
    'restitution': 34,
}

SHAPE_TYPES = {
    'sphere': 0,
    'box': 1,
    'cylinder': 2,
    'capsule': 3,
    'mesh': 4,
    'robot': 5,
    'articulation': 6,
}


@dataclass
class ObjectProperties:
    mass: float = 1.0
    radius: float = 0.1
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    shape_type: int = 0
    is_static: bool = False
    friction: float = 0.5
    is_active: bool = True
    inertia: float = 0.0
    dimensions: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    restitution: float = 0.5


@dataclass
class ExtractedState:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1]))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    properties: ObjectProperties = field(default_factory=ObjectProperties)


def create_state_array(max_objects: int = MAX_OBJECTS, state_dim: int = STATE_DIM) -> np.ndarray:
    return np.zeros((max_objects, state_dim), dtype=np.float32)


def create_mask_array(max_objects: int = MAX_OBJECTS) -> np.ndarray:
    return np.zeros(max_objects, dtype=np.float32)


def pack_state_vector(state: ExtractedState) -> np.ndarray:
    vec = np.zeros(STATE_DIM, dtype=np.float32)
    vec[STATE_INDICES['position']] = state.position
    vec[STATE_INDICES['linear_velocity']] = state.linear_velocity
    vec[STATE_INDICES['orientation']] = state.orientation
    vec[STATE_INDICES['angular_velocity']] = state.angular_velocity
    vec[STATE_INDICES['mass']] = state.properties.mass
    vec[STATE_INDICES['radius']] = state.properties.radius
    vec[STATE_INDICES['color']] = state.properties.color
    vec[STATE_INDICES['shape_type']] = state.properties.shape_type
    vec[STATE_INDICES['is_static']] = 1.0 if state.properties.is_static else 0.0
    vec[STATE_INDICES['friction']] = state.properties.friction
    vec[STATE_INDICES['is_active']] = 1.0 if state.properties.is_active else 0.0
    vec[STATE_INDICES['inertia']] = state.properties.inertia
    vec[STATE_INDICES['dimensions']] = state.properties.dimensions
    vec[STATE_INDICES['restitution']] = state.properties.restitution
    return vec


def extract_rigid_body_state(
    rigid_prim_view,
    env_idx: int,
    properties: ObjectProperties
) -> Optional[ExtractedState]:
    try:
        pos, rot = rigid_prim_view.get_world_poses(indices=[env_idx])
        vel = rigid_prim_view.get_velocities(indices=[env_idx])
        
        if pos is None:
            return None
        
        # Create a copy of properties to avoid modifying the original
        updated_props = ObjectProperties(
            mass=properties.mass,
            radius=properties.radius,
            color=properties.color,
            shape_type=properties.shape_type,
            is_static=properties.is_static,
            friction=properties.friction,
            is_active=properties.is_active,
            inertia=properties.inertia,
            dimensions=properties.dimensions,
            restitution=properties.restitution
        )
        
        # Try to extract actual physics properties from Isaac Sim
        try:
            # Get mass from rigid body
            masses = rigid_prim_view.get_masses(indices=[env_idx])
            if masses is not None and len(masses) > 0:
                updated_props.mass = float(masses[0].cpu().numpy())
            
            # Get material properties (friction, restitution)
            # Note: This requires the prim to have a PhysicsMaterial applied
            if hasattr(rigid_prim_view, 'get_friction_coefficients'):
                frictions = rigid_prim_view.get_friction_coefficients(indices=[env_idx])
                if frictions is not None and len(frictions) > 0:
                    updated_props.friction = float(frictions[0].cpu().numpy())
            
            if hasattr(rigid_prim_view, 'get_restitutions'):
                restitutions = rigid_prim_view.get_restitutions(indices=[env_idx])
                if restitutions is not None and len(restitutions) > 0:
                    updated_props.restitution = float(restitutions[0].cpu().numpy())
        except Exception:
            # If property extraction fails, use the provided defaults
            pass
        
        state = ExtractedState(properties=updated_props)
        state.position = pos[0].cpu().numpy()[:3]
        state.orientation = rot[0].cpu().numpy()[:4]
        
        if vel is not None:
            state.linear_velocity = vel[0, :3].cpu().numpy()
            state.angular_velocity = vel[0, 3:6].cpu().numpy()
        
        return state
    except Exception:
        return None


def extract_articulation_state(
    articulation_view,
    env_idx: int,
    properties: ObjectProperties
) -> Optional[ExtractedState]:
    try:
        pos, rot = articulation_view.get_world_poses(indices=[env_idx])
        vel = articulation_view.get_velocities(indices=[env_idx])
        
        if pos is None:
            return None
        
        # Create a copy of properties to avoid modifying the original
        updated_props = ObjectProperties(
            mass=properties.mass,
            radius=properties.radius,
            color=properties.color,
            shape_type=properties.shape_type,
            is_static=properties.is_static,
            friction=properties.friction,
            is_active=properties.is_active,
            inertia=properties.inertia,
            dimensions=properties.dimensions,
            restitution=properties.restitution
        )
        
        # Try to extract actual physics properties from Isaac Sim
        try:
            # Articulations may have different API for mass
            if hasattr(articulation_view, 'get_body_masses'):
                masses = articulation_view.get_body_masses(indices=[env_idx])
                if masses is not None and len(masses) > 0:
                    # Sum all body masses for total articulation mass
                    updated_props.mass = float(masses[0].sum().cpu().numpy())
        except Exception:
            pass
        
        state = ExtractedState(properties=updated_props)
        state.position = pos[0].cpu().numpy()[:3]
        state.orientation = rot[0].cpu().numpy()[:4]
        
        if vel is not None:
            state.linear_velocity = vel[0, :3].cpu().numpy()
            state.angular_velocity = vel[0, 3:6].cpu().numpy()
        
        return state
    except Exception:
        return None


def extract_tensor_state(
    position_tensor,
    env_idx: int,
    properties: ObjectProperties,
    velocity_tensor=None,
    orientation_tensor=None
) -> Optional[ExtractedState]:
    try:
        state = ExtractedState(properties=properties)
        state.position = position_tensor[env_idx].cpu().numpy()[:3]
        
        if velocity_tensor is not None:
            vel = velocity_tensor[env_idx].cpu().numpy()
            state.linear_velocity = vel[:3] if len(vel) >= 3 else vel
            if len(vel) >= 6:
                state.angular_velocity = vel[3:6]
        
        if orientation_tensor is not None:
            state.orientation = orientation_tensor[env_idx].cpu().numpy()[:4]
        
        return state
    except Exception:
        return None


class StateExtractor:
    def __init__(self, max_objects: int = MAX_OBJECTS):
        self.max_objects = max_objects
        self.object_configs: List[Dict[str, Any]] = []
    
    def add_object_source(
        self,
        name: str,
        source_type: str,
        view_attr: str,
        properties: ObjectProperties,
        position_attr: Optional[str] = None,
        velocity_attr: Optional[str] = None
    ):
        self.object_configs.append({
            'name': name,
            'source_type': source_type,
            'view_attr': view_attr,
            'properties': properties,
            'position_attr': position_attr,
            'velocity_attr': velocity_attr,
        })
    
    def extract_all(self, task, env_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        states = create_state_array(self.max_objects)
        masks = create_mask_array(self.max_objects)
        obj_idx = 0
        
        for config in self.object_configs:
            if obj_idx >= self.max_objects:
                break
            
            state = self._extract_single(task, env_idx, config)
            if state is not None:
                states[obj_idx] = pack_state_vector(state)
                masks[obj_idx] = 1.0
                obj_idx += 1
        
        return states, masks
    
    def _extract_single(self, task, env_idx: int, config: Dict) -> Optional[ExtractedState]:
        source_type = config['source_type']
        view_attr = config['view_attr']
        properties = config['properties']
        
        if source_type == 'rigid_prim_view':
            view = getattr(task, view_attr, None)
            if view is not None:
                return extract_rigid_body_state(view, env_idx, properties)
        
        elif source_type == 'articulation_view':
            view = getattr(task, view_attr, None)
            if view is not None:
                return extract_articulation_state(view, env_idx, properties)
        
        elif source_type == 'tensor':
            pos_attr = config.get('position_attr')
            vel_attr = config.get('velocity_attr')
            
            pos_tensor = getattr(task, pos_attr, None) if pos_attr else None
            vel_tensor = getattr(task, vel_attr, None) if vel_attr else None
            
            if pos_tensor is not None:
                return extract_tensor_state(pos_tensor, env_idx, properties, vel_tensor)
        
        return None
