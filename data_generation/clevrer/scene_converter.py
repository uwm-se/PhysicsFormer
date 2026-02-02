"""
Copyright (c) 2026 Anonymous. All rights reserved.
Author: Anonymous

PROPRIETARY AND CONFIDENTIAL. This software is provided for academic review
and research purposes only. Unauthorized copying, modification, distribution,
or use of this software, via any medium, is strictly prohibited without prior
written permission from Anonymous.
"""

"""
CLEVRER Scene to Physics-LLM State Converter

Converts CLEVRER scene JSON format to Physics-LLM state tensor format.
CLEVRER uses Bullet physics engine (same as PyBullet), so physics are compatible.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


SHAPE_MAP = {
    'sphere': 0,
    'cube': 1,
    'cylinder': 2
}

COLOR_MAP = {
    'gray': [0.5, 0.5, 0.5],
    'red': [1.0, 0.0, 0.0],
    'blue': [0.0, 0.0, 1.0],
    'green': [0.0, 1.0, 0.0],
    'brown': [0.6, 0.3, 0.1],
    'purple': [0.5, 0.0, 0.5],
    'cyan': [0.0, 1.0, 1.0],
    'yellow': [1.0, 1.0, 0.0]
}

MATERIAL_MAP = {
    'rubber': {'mass': 1.0, 'friction': 0.8, 'restitution': 0.9},
    'metal': {'mass': 2.0, 'friction': 0.4, 'restitution': 0.3}
}

DEFAULT_SIZE = 0.3
CLEVRER_FPS = 24.0


def load_clevrer_scene(scene_path: str) -> Dict[str, Any]:
    """Load a CLEVRER scene JSON file."""
    with open(scene_path, 'r') as f:
        return json.load(f)


def derive_velocity(positions: List[List[float]], frame_idx: int, fps: float = CLEVRER_FPS) -> np.ndarray:
    """Derive velocity from position deltas."""
    if frame_idx == 0 or frame_idx >= len(positions):
        return np.zeros(3)
    
    current_pos = np.array(positions[frame_idx])
    prev_pos = np.array(positions[frame_idx - 1])
    return (current_pos - prev_pos) * fps


def get_object_bbox(shape: str, size: float = DEFAULT_SIZE) -> np.ndarray:
    """Get bounding box dimensions based on shape."""
    if shape == 'sphere':
        return np.array([size * 2, size * 2, size * 2])
    elif shape == 'cylinder':
        return np.array([size, size, size * 1.5])
    return np.array([size, size, size])


def construct_state_vector(
    position: List[float],
    velocity: np.ndarray,
    shape: str,
    color: str,
    material: str
) -> np.ndarray:
    """
    Construct a 35-dim state vector matching Physics-LLM input format.
    
    State vector layout (35 dimensions) - matches state_extraction.py:
    [0:3]   - Position (x, y, z)
    [3:6]   - Linear velocity (vx, vy, vz)
    [6:10]  - Orientation quaternion (qx, qy, qz, qw)
    [10:13] - Angular velocity (wx, wy, wz)
    [13]    - Mass
    [14]    - Radius
    [15:18] - Color (r, g, b)
    [18]    - Shape type
    [19]    - Is static
    [20]    - Friction
    [21]    - Is active
    [22]    - Inertia
    [23]    - Bbox min
    [24]    - Bbox max
    [25:28] - Dimensions (width, height, depth)
    [28:31] - Force
    [31:34] - Torque
    [34]    - Restitution
    """
    state = np.zeros(35, dtype=np.float32)
    
    state[0:3] = position
    state[3:6] = velocity
    state[6:10] = [0.0, 0.0, 0.0, 1.0]  # Identity quaternion (xyzw)
    state[10:13] = [0.0, 0.0, 0.0]  # No angular velocity data in CLEVRER
    
    mat_props = MATERIAL_MAP.get(material, MATERIAL_MAP['rubber'])
    state[13] = mat_props['mass']
    state[14] = DEFAULT_SIZE  # Radius
    state[15:18] = COLOR_MAP.get(color, [0.5, 0.5, 0.5])  # Color RGB
    state[18] = SHAPE_MAP.get(shape, 0)  # Shape type
    state[19] = 0.0  # Is static (all CLEVRER objects are dynamic)
    state[20] = mat_props['friction']
    state[21] = 1.0  # Is active
    state[22] = 0.0  # Inertia (not provided by CLEVRER)
    state[25:28] = get_object_bbox(shape)  # Dimensions
    state[34] = mat_props['restitution']
    
    return state


def clevrer_scene_to_state_tensor(scene: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Convert CLEVRER scene to Physics-LLM state tensor format.
    
    Args:
        scene: CLEVRER scene dictionary containing objects and their trajectories
        
    Returns:
        states: [T, N, 35] state tensor
        masks: [T, N] validity masks
        metadata: Scene metadata including collisions and object info
    """
    # Handle new CLEVRER annotation format with motion_trajectory
    if 'motion_trajectory' in scene:
        return _convert_motion_trajectory_format(scene)
    
    # Legacy format with objects list
    objects = scene.get('objects', [])
    if not objects:
        raise ValueError("Scene contains no objects")
    
    # Determine number of frames from first object's trajectory
    first_obj = objects[0]
    if 'locations' in first_obj:
        num_frames = len(first_obj['locations'])
        position_key = 'locations'
    elif 'positions' in first_obj:
        num_frames = len(first_obj['positions'])
        position_key = 'positions'
    else:
        raise ValueError("Cannot find position data in scene objects")
    
    num_objects = len(objects)
    states = np.zeros((num_frames, num_objects, 35), dtype=np.float32)
    masks = np.ones((num_frames, num_objects), dtype=np.float32)
    
    for obj_idx, obj in enumerate(objects):
        positions = obj.get(position_key, [])
        shape = obj.get('shape', 'sphere')
        color = obj.get('color', 'gray')
        material = obj.get('material', 'rubber')
        
        for t in range(min(num_frames, len(positions))):
            pos = positions[t]
            vel = derive_velocity(positions, t)
            states[t, obj_idx] = construct_state_vector(pos, vel, shape, color, material)
        
        # Mark frames beyond available positions as invalid
        if len(positions) < num_frames:
            masks[len(positions):, obj_idx] = 0.0
    
    metadata = {
        'num_objects': num_objects,
        'num_frames': num_frames,
        'objects': [
            {
                'shape': obj.get('shape', 'sphere'),
                'color': obj.get('color', 'gray'),
                'material': obj.get('material', 'rubber')
            }
            for obj in objects
        ],
        'collisions': scene.get('collisions', [])
    }
    
    return states, masks, metadata


def _convert_motion_trajectory_format(scene: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Convert CLEVRER annotation format with motion_trajectory."""
    motion_trajectory = scene['motion_trajectory']
    object_properties = {obj['object_id']: obj for obj in scene.get('object_property', [])}
    
    num_frames = len(motion_trajectory)
    if num_frames == 0:
        raise ValueError("Scene contains no frames")
    
    # Get number of objects from first frame
    first_frame = motion_trajectory[0]
    num_objects = len(first_frame.get('objects', []))
    if num_objects == 0:
        raise ValueError("Scene contains no objects")
    
    states = np.zeros((num_frames, num_objects, 35), dtype=np.float32)
    masks = np.ones((num_frames, num_objects), dtype=np.float32)
    
    for t, frame in enumerate(motion_trajectory):
        for obj_data in frame.get('objects', []):
            obj_id = obj_data['object_id']
            if obj_id >= num_objects:
                continue
            
            # Get static properties
            props = object_properties.get(obj_id, {})
            shape = props.get('shape', 'sphere')
            color = props.get('color', 'gray')
            material = props.get('material', 'rubber')
            
            # Get dynamic state
            location = obj_data.get('location', [0, 0, 0])
            velocity = obj_data.get('velocity', [0, 0, 0])
            
            states[t, obj_id] = construct_state_vector(location, np.array(velocity), shape, color, material)
            
            # Mark as invisible if outside camera view
            if not obj_data.get('inside_camera_view', True):
                masks[t, obj_id] = 0.5  # Partially visible
    
    metadata = {
        'num_objects': num_objects,
        'num_frames': num_frames,
        'objects': [
            {
                'shape': object_properties.get(i, {}).get('shape', 'sphere'),
                'color': object_properties.get(i, {}).get('color', 'gray'),
                'material': object_properties.get(i, {}).get('material', 'rubber')
            }
            for i in range(num_objects)
        ],
        'collisions': scene.get('collision', [])
    }
    
    return states, masks, metadata


def convert_clevrer_dataset(
    scene_dir: Path,
    output_dir: Path,
    max_scenes: Optional[int] = None
) -> List[str]:
    """
    Convert multiple CLEVRER scenes to Physics-LLM format.
    
    Args:
        scene_dir: Directory containing CLEVRER scene JSON files
        output_dir: Directory to save converted state tensors
        max_scenes: Maximum number of scenes to convert (None for all)
        
    Returns:
        List of converted scene IDs
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_files = sorted(scene_dir.glob('*.json'))
    
    if max_scenes:
        scene_files = scene_files[:max_scenes]
    
    converted = []
    for scene_file in scene_files:
        try:
            scene = load_clevrer_scene(str(scene_file))
            states, masks, metadata = clevrer_scene_to_state_tensor(scene)
            
            scene_id = scene_file.stem
            np.savez(
                output_dir / f'{scene_id}.npz',
                states=states,
                masks=masks,
                metadata=json.dumps(metadata)
            )
            converted.append(scene_id)
        except Exception as e:
            print(f"Failed to convert {scene_file.name}: {e}")
    
    return converted
