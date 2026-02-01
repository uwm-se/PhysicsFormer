"""
Copyright (c) 2026 Style Machine LLC. All rights reserved.
Author: Jesse Pokora

PROPRIETARY AND CONFIDENTIAL. This software is provided for academic review
and research purposes only. Unauthorized copying, modification, distribution,
or use of this software, via any medium, is strictly prohibited without prior
written permission from Style Machine LLC.
"""

"""
Generate Robotics-Focused Physics Schemas

New schemas for grounding robotics concepts:
- Manipulation (pickup, push, rotate)
- Visibility/Occlusion
- Material properties (fragility)
- Containment/Capacity

These extend the existing PyBullet simulation with additional properties
needed for embodied AI and robotic manipulation.
"""

import sys
from pathlib import Path

helpers_dir = Path(__file__).parent / "helpers"
if str(helpers_dir) not in sys.path:
    sys.path.insert(0, str(helpers_dir))

import numpy as np
import json
import argparse
from tqdm import tqdm
import pybullet as p
import pybullet_data
import gc
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional


class ShapeType(Enum):
    SPHERE = "sphere"
    BOX = "box"
    CYLINDER = "cylinder"


class MaterialType(Enum):
    METAL = "metal"
    WOOD = "wood"
    GLASS = "glass"
    PLASTIC = "plastic"
    CERAMIC = "ceramic"


MATERIAL_PROPERTIES = {
    MaterialType.METAL: {"fragility_threshold": 1000.0, "density": 7800, "friction": 0.4, "restitution": 0.3},
    MaterialType.WOOD: {"fragility_threshold": 50.0, "density": 600, "friction": 0.5, "restitution": 0.2},
    MaterialType.GLASS: {"fragility_threshold": 5.0, "density": 2500, "friction": 0.3, "restitution": 0.1},
    MaterialType.PLASTIC: {"fragility_threshold": 30.0, "density": 1200, "friction": 0.4, "restitution": 0.4},
    MaterialType.CERAMIC: {"fragility_threshold": 8.0, "density": 2400, "friction": 0.5, "restitution": 0.1},
}


ROBOTICS_SCHEMAS = [
    "object_pickup",
    "fragile_objects",
    "occlusion_static",
    "object_placement",
    "rotation_manipulation",
]


def get_timesteps_for_robotics_schema(schema: str) -> int:
    """Return appropriate timesteps for robotics schemas."""
    timestep_config = {
        "object_pickup": 150,
        "fragile_objects": 100,
        "occlusion_static": 50,
        "object_placement": 150,
        "rotation_manipulation": 200,
        "default": 100
    }
    return timestep_config.get(schema, timestep_config["default"])


class RoboticsPhysicsGenerator:
    """Generate robotics-focused physics episodes with extended state vectors."""
    
    def __init__(self, use_gui: bool = False):
        if use_gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(0.01, physicsClientId=self.client)
        
        self.observer_position = [0, -3, 1.5]
    
    def cleanup(self):
        try:
            p.disconnect(physicsClientId=self.client)
        except:
            pass
    
    def generate_episode(self, schema: str, episode_id: int, num_steps: int = 100) -> dict:
        """Generate a robotics physics episode."""
        seed = (hash(schema) % 1000000) + episode_id
        np.random.seed(seed % (2**32))
        
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        
        setup_result = self._setup_schema(schema)
        objects, object_properties, schema_metadata = setup_result
        
        states = []
        previous_state = None
        
        for step in range(num_steps):
            self._apply_schema_forces(schema, objects, object_properties, step, schema_metadata)
            
            p.stepSimulation(physicsClientId=self.client)
            
            state = self._capture_extended_state(objects, object_properties, previous_state, schema_metadata)
            states.append(state)
            previous_state = state
        
        for obj_id in objects:
            try:
                p.removeBody(obj_id, physicsClientId=self.client)
            except:
                pass
        
        return {
            'states': states,
            'schema': schema,
            'object_properties': object_properties,
            'schema_metadata': schema_metadata,
            'metadata': {
                'episode_id': episode_id,
                'num_steps': num_steps,
                'seed': seed
            }
        }
    
    def _setup_schema(self, schema: str) -> Tuple[List[int], List[Dict], Dict]:
        """Setup objects for robotics schema."""
        
        if schema == "object_pickup":
            return self._setup_object_pickup()
        elif schema == "pushing_sliding":
            return self._setup_pushing_sliding()
        elif schema == "fragile_objects":
            return self._setup_fragile_objects()
        elif schema == "occlusion_static":
            return self._setup_occlusion_static()
        elif schema == "object_placement":
            return self._setup_object_placement()
        elif schema == "rotation_manipulation":
            return self._setup_rotation_manipulation()
        elif schema == "stacking_precision":
            return self._setup_stacking_precision()
        elif schema == "container_filling":
            return self._setup_container_filling()
        else:
            return self._setup_object_pickup()
    
    def _create_ground(self) -> int:
        """Create ground plane."""
        ground_shape = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self.client)
        ground = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=ground_shape, physicsClientId=self.client)
        p.changeDynamics(ground, -1, lateralFriction=0.5, physicsClientId=self.client)
        return ground
    
    def _create_object(
        self,
        shape_type: ShapeType,
        position: List[float],
        dimensions: List[float],
        material: MaterialType,
        velocity: List[float] = None
    ) -> Tuple[int, Dict]:
        """Create an object with extended properties."""
        
        mat_props = MATERIAL_PROPERTIES[material]
        
        if shape_type == ShapeType.SPHERE:
            radius = dimensions[0]
            volume = (4/3) * np.pi * radius**3
            shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=self.client)
        elif shape_type == ShapeType.BOX:
            half_extents = [d/2 for d in dimensions]
            volume = dimensions[0] * dimensions[1] * dimensions[2]
            shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.client)
        elif shape_type == ShapeType.CYLINDER:
            radius, height = dimensions[0], dimensions[1]
            volume = np.pi * radius**2 * height
            shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=self.client)
        else:
            radius = 0.1
            volume = (4/3) * np.pi * radius**3
            shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=self.client)
        
        mass = volume * mat_props["density"]
        mass = np.clip(mass, 0.1, 100.0)
        
        obj_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=shape,
            basePosition=position,
            physicsClientId=self.client
        )
        
        p.changeDynamics(
            obj_id, -1,
            lateralFriction=mat_props["friction"],
            restitution=mat_props["restitution"],
            physicsClientId=self.client
        )
        
        if velocity:
            p.resetBaseVelocity(obj_id, linearVelocity=velocity, physicsClientId=self.client)
        
        max_dim = max(dimensions)
        graspable = bool(max_dim < 0.3 and mass < 10.0)
        
        properties = {
            "mass": float(mass),
            "shape_type": shape_type.value,
            "dimensions": [float(d) for d in dimensions],
            "material_type": material.value,
            "fragility_threshold": float(mat_props["fragility_threshold"]),
            "friction": float(mat_props["friction"]),
            "restitution": float(mat_props["restitution"]),
            "graspable": graspable,
            "is_container": False,
            "container_volume": 0.0,
            "is_broken": False,
            "max_impact_force": 0.0
        }
        
        return obj_id, properties
    
    def _setup_object_pickup(self) -> Tuple[List[int], List[Dict], Dict]:
        """Setup for STATIC graspability assessment scenario.
        
        Objects rest on ground - no invisible forces applied.
        Model learns to predict graspability from object properties.
        """
        objects = []
        properties = []
        
        ground = self._create_ground()
        objects.append(ground)
        properties.append({
            "mass": 0, "shape_type": "plane", "dimensions": [10, 10, 0],
            "material_type": "ground", "fragility_threshold": 1e9,
            "friction": 0.5, "restitution": 0.0, "graspable": False,
            "is_container": False, "container_volume": 0.0,
            "is_broken": False, "max_impact_force": 0.0
        })
        
        num_objects = np.random.randint(3, 8)
        graspable_count = 0
        
        for i in range(num_objects):
            shape = np.random.choice(list(ShapeType))
            material = np.random.choice(list(MaterialType))
            
            size_category = np.random.choice(["small", "medium", "large"])
            
            if size_category == "small":
                scale = np.random.uniform(0.02, 0.08)
            elif size_category == "medium":
                scale = np.random.uniform(0.08, 0.2)
            else:
                scale = np.random.uniform(0.2, 0.5)
            
            if shape == ShapeType.SPHERE:
                dims = [scale]
                z_offset = scale
            elif shape == ShapeType.BOX:
                dims = [scale * np.random.uniform(0.8, 1.2) for _ in range(3)]
                z_offset = dims[2] / 2
            else:
                radius = scale * 0.5
                height = scale * np.random.uniform(1.5, 3.0)
                dims = [radius, height]
                z_offset = height / 2
            
            x = np.random.uniform(-1.5, 1.5)
            y = np.random.uniform(-1.5, 1.5)
            pos = [x, y, z_offset + 0.01]
            
            obj_id, props = self._create_object(shape, pos, dims, material)
            
            p.resetBaseVelocity(obj_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
            
            objects.append(obj_id)
            properties.append(props)
            
            if props["graspable"]:
                graspable_count += 1
        
        metadata = {
            "scenario": "graspability_assessment",
            "num_objects": num_objects,
            "graspable_count": graspable_count,
            "static_scene": True
        }
        
        return objects, properties, metadata
    
    def _setup_pushing_sliding(self) -> Tuple[List[int], List[Dict], Dict]:
        """Setup for pushing/sliding scenario."""
        objects = []
        properties = []
        
        ground = self._create_ground()
        objects.append(ground)
        properties.append({
            "mass": 0, "shape_type": "plane", "dimensions": [10, 10, 0],
            "material_type": "ground", "fragility_threshold": 1e9,
            "friction": 0.5, "restitution": 0.0, "graspable": False,
            "is_container": False, "container_volume": 0.0,
            "is_broken": False, "max_impact_force": 0.0
        })
        
        shape = np.random.choice([ShapeType.BOX, ShapeType.CYLINDER])
        material = np.random.choice(list(MaterialType))
        
        if shape == ShapeType.BOX:
            dims = [np.random.uniform(0.1, 0.3) for _ in range(3)]
            z_offset = dims[2] / 2
        else:
            radius = np.random.uniform(0.05, 0.15)
            height = np.random.uniform(0.1, 0.25)
            dims = [radius, height]
            z_offset = height / 2
        
        pos = [0, 0, z_offset + 0.01]
        obj_id, props = self._create_object(shape, pos, dims, material)
        objects.append(obj_id)
        properties.append(props)
        
        push_direction = np.random.uniform(-np.pi, np.pi)
        push_force = np.random.uniform(1, 20)
        
        metadata = {
            "scenario": "pushing",
            "push_direction": push_direction,
            "push_force": push_force,
            "push_start_step": 10,
            "push_duration": 30
        }
        
        return objects, properties, metadata
    
    def _setup_fragile_objects(self) -> Tuple[List[int], List[Dict], Dict]:
        """Setup for fragile objects collision scenario."""
        objects = []
        properties = []
        
        ground = self._create_ground()
        objects.append(ground)
        properties.append({
            "mass": 0, "shape_type": "plane", "dimensions": [10, 10, 0],
            "material_type": "ground", "fragility_threshold": 1e9,
            "friction": 0.5, "restitution": 0.0, "graspable": False,
            "is_container": False, "container_volume": 0.0,
            "is_broken": False, "max_impact_force": 0.0
        })
        
        fragile_material = np.random.choice([MaterialType.GLASS, MaterialType.CERAMIC])
        shape = np.random.choice(list(ShapeType))
        
        if shape == ShapeType.SPHERE:
            dims = [np.random.uniform(0.05, 0.15)]
        elif shape == ShapeType.BOX:
            dims = [np.random.uniform(0.05, 0.15) for _ in range(3)]
        else:
            dims = [np.random.uniform(0.03, 0.1), np.random.uniform(0.1, 0.2)]
        
        drop_height = np.random.uniform(0.5, 2.0)
        pos = [0, 0, drop_height]
        
        obj_id, props = self._create_object(shape, pos, dims, fragile_material)
        objects.append(obj_id)
        properties.append(props)
        
        metadata = {
            "scenario": "fragile_drop",
            "drop_height": drop_height,
            "fragility_threshold": props["fragility_threshold"]
        }
        
        return objects, properties, metadata
    
    def _setup_occlusion_static(self) -> Tuple[List[int], List[Dict], Dict]:
        """Setup for occlusion/visibility scenario."""
        objects = []
        properties = []
        
        ground = self._create_ground()
        objects.append(ground)
        properties.append({
            "mass": 0, "shape_type": "plane", "dimensions": [10, 10, 0],
            "material_type": "ground", "fragility_threshold": 1e9,
            "friction": 0.5, "restitution": 0.0, "graspable": False,
            "is_container": False, "container_volume": 0.0,
            "is_broken": False, "max_impact_force": 0.0
        })
        
        target_pos = [np.random.uniform(0.5, 2), 0, np.random.uniform(0.2, 1)]
        target_dims = [np.random.uniform(0.05, 0.15)]
        target_id, target_props = self._create_object(
            ShapeType.SPHERE, target_pos, target_dims, MaterialType.PLASTIC
        )
        objects.append(target_id)
        properties.append(target_props)
        
        num_occluders = np.random.randint(1, 4)
        occluder_positions = []
        
        for i in range(num_occluders):
            occ_x = np.random.uniform(0.2, target_pos[0] - 0.2)
            occ_y = np.random.uniform(-0.5, 0.5)
            occ_z = np.random.uniform(0.2, 1.5)
            occ_pos = [occ_x, occ_y, occ_z]
            occluder_positions.append(occ_pos)
            
            occ_dims = [np.random.uniform(0.1, 0.4) for _ in range(3)]
            occ_id, occ_props = self._create_object(
                ShapeType.BOX, occ_pos, occ_dims, MaterialType.WOOD
            )
            objects.append(occ_id)
            properties.append(occ_props)
        
        metadata = {
            "scenario": "occlusion",
            "observer_position": self.observer_position,
            "target_object_idx": 1,
            "occluder_indices": list(range(2, 2 + num_occluders)),
            "static_scene": True
        }
        
        return objects, properties, metadata
    
    def _setup_object_placement(self) -> Tuple[List[int], List[Dict], Dict]:
        """Setup for STATIC placement feasibility assessment.
        
        Objects already placed on surfaces - no invisible release.
        Model learns to assess placement stability from object properties and positions.
        """
        objects = []
        properties = []
        
        ground = self._create_ground()
        objects.append(ground)
        properties.append({
            "mass": 0, "shape_type": "plane", "dimensions": [10, 10, 0],
            "material_type": "ground", "fragility_threshold": 1e9,
            "friction": 0.5, "restitution": 0.0, "graspable": False,
            "is_container": False, "container_volume": 0.0,
            "is_broken": False, "max_impact_force": 0.0
        })
        
        table_dims = [0.8, 0.6, 0.02]
        table_height = 0.7
        table_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[d/2 for d in table_dims],
            physicsClientId=self.client
        )
        table_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=table_shape,
            basePosition=[0, 0, table_height],
            physicsClientId=self.client
        )
        objects.append(table_id)
        properties.append({
            "mass": 0, "shape_type": "box", "dimensions": table_dims,
            "material_type": "wood", "fragility_threshold": 1e9,
            "friction": 0.5, "restitution": 0.1, "graspable": False,
            "is_container": False, "container_volume": 0.0,
            "is_broken": False, "max_impact_force": 0.0
        })
        
        num_objects = np.random.randint(2, 5)
        stable_placements = 0
        
        for i in range(num_objects):
            shape = np.random.choice(list(ShapeType))
            material = np.random.choice(list(MaterialType))
            
            if shape == ShapeType.SPHERE:
                dims = [np.random.uniform(0.03, 0.1)]
                z_offset = dims[0]
            elif shape == ShapeType.BOX:
                dims = [np.random.uniform(0.03, 0.1) for _ in range(3)]
                z_offset = dims[2] / 2
            else:
                dims = [np.random.uniform(0.02, 0.06), np.random.uniform(0.05, 0.15)]
                z_offset = dims[1] / 2
            
            placement_type = np.random.choice(["on_table", "edge", "off_table"])
            
            if placement_type == "on_table":
                x = np.random.uniform(-0.25, 0.25)
                y = np.random.uniform(-0.2, 0.2)
                z = table_height + table_dims[2]/2 + z_offset + 0.001
                is_stable = True
            elif placement_type == "edge":
                edge_side = np.random.choice(["x", "y"])
                if edge_side == "x":
                    x = np.random.choice([-1, 1]) * (table_dims[0]/2 - 0.02)
                    y = np.random.uniform(-0.2, 0.2)
                else:
                    x = np.random.uniform(-0.25, 0.25)
                    y = np.random.choice([-1, 1]) * (table_dims[1]/2 - 0.02)
                z = table_height + table_dims[2]/2 + z_offset + 0.001
                is_stable = shape != ShapeType.SPHERE
            else:
                x = np.random.uniform(-1.5, 1.5)
                y = np.random.uniform(-1.5, 1.5)
                while abs(x) < 0.5 and abs(y) < 0.4:
                    x = np.random.uniform(-1.5, 1.5)
                    y = np.random.uniform(-1.5, 1.5)
                z = z_offset + 0.001
                is_stable = True
            
            pos = [x, y, z]
            obj_id, props = self._create_object(shape, pos, dims, material)
            
            p.resetBaseVelocity(obj_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
            
            props["placement_stable"] = is_stable
            objects.append(obj_id)
            properties.append(props)
            
            if is_stable:
                stable_placements += 1
        
        metadata = {
            "scenario": "placement_assessment",
            "table_height": table_height,
            "table_dims": table_dims,
            "num_objects": num_objects,
            "stable_placements": stable_placements,
            "static_scene": True
        }
        
        return objects, properties, metadata
    
    def _setup_rotation_manipulation(self) -> Tuple[List[int], List[Dict], Dict]:
        """Setup for rotation/orientation manipulation."""
        objects = []
        properties = []
        
        ground = self._create_ground()
        objects.append(ground)
        properties.append({
            "mass": 0, "shape_type": "plane", "dimensions": [10, 10, 0],
            "material_type": "ground", "fragility_threshold": 1e9,
            "friction": 0.5, "restitution": 0.0, "graspable": False,
            "is_container": False, "container_volume": 0.0,
            "is_broken": False, "max_impact_force": 0.0
        })
        
        dims = [np.random.uniform(0.05, 0.15) for _ in range(3)]
        dims[2] = dims[2] * 2
        
        initial_tilt = np.random.uniform(0, np.pi/3)
        orn = p.getQuaternionFromEuler([initial_tilt, 0, 0])
        
        shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[d/2 for d in dims],
            physicsClientId=self.client
        )
        
        material = np.random.choice(list(MaterialType))
        mat_props = MATERIAL_PROPERTIES[material]
        volume = dims[0] * dims[1] * dims[2]
        mass = volume * mat_props["density"]
        mass = np.clip(mass, 0.1, 50.0)
        
        obj_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=shape,
            basePosition=[0, 0, dims[2]/2 + 0.01],
            baseOrientation=orn,
            physicsClientId=self.client
        )
        objects.append(obj_id)
        
        properties.append({
            "mass": mass,
            "shape_type": "box",
            "dimensions": dims,
            "material_type": material.value,
            "fragility_threshold": mat_props["fragility_threshold"],
            "friction": mat_props["friction"],
            "restitution": mat_props["restitution"],
            "graspable": True,
            "is_container": False,
            "container_volume": 0.0,
            "is_broken": False,
            "max_impact_force": 0.0
        })
        
        metadata = {
            "scenario": "rotation",
            "initial_tilt": initial_tilt,
            "target_upright": True
        }
        
        return objects, properties, metadata
    
    def _setup_stacking_precision(self) -> Tuple[List[int], List[Dict], Dict]:
        """Setup for stacking scenario."""
        objects = []
        properties = []
        
        ground = self._create_ground()
        objects.append(ground)
        properties.append({
            "mass": 0, "shape_type": "plane", "dimensions": [10, 10, 0],
            "material_type": "ground", "fragility_threshold": 1e9,
            "friction": 0.5, "restitution": 0.0, "graspable": False,
            "is_container": False, "container_volume": 0.0,
            "is_broken": False, "max_impact_force": 0.0
        })
        
        num_blocks = np.random.randint(2, 5)
        current_height = 0
        
        for i in range(num_blocks):
            size = np.random.uniform(0.08, 0.15)
            dims = [size, size, size]
            
            x_offset = np.random.uniform(-0.02, 0.02)
            y_offset = np.random.uniform(-0.02, 0.02)
            pos = [x_offset, y_offset, current_height + size/2 + 0.01]
            
            material = np.random.choice([MaterialType.WOOD, MaterialType.PLASTIC])
            obj_id, props = self._create_object(ShapeType.BOX, pos, dims, material)
            objects.append(obj_id)
            properties.append(props)
            
            current_height += size
        
        metadata = {
            "scenario": "stacking",
            "num_blocks": num_blocks,
            "initial_height": current_height
        }
        
        return objects, properties, metadata
    
    def _setup_container_filling(self) -> Tuple[List[int], List[Dict], Dict]:
        """Setup for container filling scenario."""
        objects = []
        properties = []
        
        ground = self._create_ground()
        objects.append(ground)
        properties.append({
            "mass": 0, "shape_type": "plane", "dimensions": [10, 10, 0],
            "material_type": "ground", "fragility_threshold": 1e9,
            "friction": 0.5, "restitution": 0.0, "graspable": False,
            "is_container": False, "container_volume": 0.0,
            "is_broken": False, "max_impact_force": 0.0
        })
        
        container_size = np.random.uniform(0.2, 0.4)
        wall_thickness = 0.02
        
        floor_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[container_size/2, container_size/2, wall_thickness/2],
            physicsClientId=self.client
        )
        floor_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=floor_shape,
            basePosition=[0, 0, wall_thickness/2],
            physicsClientId=self.client
        )
        objects.append(floor_id)
        
        container_volume = container_size ** 3
        properties.append({
            "mass": 0, "shape_type": "box", "dimensions": [container_size, container_size, wall_thickness],
            "material_type": "plastic", "fragility_threshold": 1e9,
            "friction": 0.5, "restitution": 0.1, "graspable": False,
            "is_container": True, "container_volume": container_volume,
            "is_broken": False, "max_impact_force": 0.0
        })
        
        wall_height = container_size * 0.8
        wall_positions = [
            ([container_size/2, 0, wall_height/2 + wall_thickness], [wall_thickness/2, container_size/2, wall_height/2]),
            ([-container_size/2, 0, wall_height/2 + wall_thickness], [wall_thickness/2, container_size/2, wall_height/2]),
            ([0, container_size/2, wall_height/2 + wall_thickness], [container_size/2, wall_thickness/2, wall_height/2]),
            ([0, -container_size/2, wall_height/2 + wall_thickness], [container_size/2, wall_thickness/2, wall_height/2]),
        ]
        
        for pos, half_extents in wall_positions:
            wall_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.client)
            wall_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_shape, basePosition=pos, physicsClientId=self.client)
            objects.append(wall_id)
            properties.append({
                "mass": 0, "shape_type": "box", "dimensions": [h*2 for h in half_extents],
                "material_type": "plastic", "fragility_threshold": 1e9,
                "friction": 0.5, "restitution": 0.1, "graspable": False,
                "is_container": False, "container_volume": 0.0,
                "is_broken": False, "max_impact_force": 0.0
            })
        
        num_balls = np.random.randint(3, 8)
        ball_radius = container_size * 0.1
        
        for i in range(num_balls):
            x = np.random.uniform(-container_size/3, container_size/3)
            y = np.random.uniform(-container_size/3, container_size/3)
            z = wall_height + 0.5 + i * 0.1
            
            obj_id, props = self._create_object(
                ShapeType.SPHERE, [x, y, z], [ball_radius], MaterialType.PLASTIC
            )
            objects.append(obj_id)
            properties.append(props)
        
        metadata = {
            "scenario": "container_filling",
            "container_volume": container_volume,
            "num_objects": num_balls,
            "container_idx": 1
        }
        
        return objects, properties, metadata
    
    def _apply_schema_forces(
        self,
        schema: str,
        objects: List[int],
        properties: List[Dict],
        step: int,
        metadata: Dict
    ):
        """Apply schema-specific forces during simulation.
        
        Static assessment schemas (object_pickup, object_placement, occlusion_static)
        do NOT apply any forces - they are purely observational.
        """
        
        if metadata.get("static_scene", False):
            return
        
        if schema in ["object_pickup", "object_placement", "occlusion_static"]:
            return
    
    def _capture_extended_state(
        self,
        objects: List[int],
        properties: List[Dict],
        previous_state: Optional[Dict],
        metadata: Dict
    ) -> Dict:
        """Capture extended state including robotics-specific fields."""
        
        all_objects = []
        
        for i, obj_id in enumerate(objects):
            try:
                pos, orn = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.client)
                vel, ang_vel = p.getBaseVelocity(obj_id, physicsClientId=self.client)
                
                orn_euler = p.getEulerFromQuaternion(orn)
                world_up = [0, 0, 1]
                obj_up = p.multiplyTransforms([0,0,0], orn, [0,0,1], [0,0,0,1])[0]
                uprightness = max(0, obj_up[2])
                
                contact_points = p.getContactPoints(bodyA=obj_id, physicsClientId=self.client)
                contact_count = len(contact_points)
                is_grounded = any(cp[2] == objects[0] for cp in contact_points) if len(objects) > 0 else False
                
                max_impact_force = 0
                for cp in contact_points:
                    normal_force = cp[9]
                    max_impact_force = max(max_impact_force, abs(normal_force))
                
                if i < len(properties):
                    if max_impact_force > properties[i].get("max_impact_force", 0):
                        properties[i]["max_impact_force"] = max_impact_force
                    
                    if max_impact_force > properties[i].get("fragility_threshold", 1e9):
                        properties[i]["is_broken"] = True
                
                force = [0, 0, 0]
                if previous_state and i < len(previous_state.get('objects', [])):
                    prev_vel = previous_state['objects'][i].get('linear_velocity', [0,0,0])
                    mass = properties[i].get('mass', 1.0) if i < len(properties) else 1.0
                    dt = 0.01
                    force = [(vel[j] - prev_vel[j]) * mass / dt for j in range(3)]
                
                visible = self._check_visibility(obj_id, objects, i)
                
                all_objects.append({
                    'position': list(pos),
                    'linear_velocity': list(vel),
                    'orientation': list(orn),
                    'angular_velocity': list(ang_vel),
                    'force': force,
                    'uprightness': float(uprightness),
                    'contact_count': int(contact_count),
                    'is_grounded': bool(is_grounded),
                    'visible_from_origin': bool(visible),
                    'max_impact_force': float(max_impact_force)
                })
                
            except Exception as e:
                all_objects.append({
                    'position': [0, 0, 0],
                    'linear_velocity': [0, 0, 0],
                    'orientation': [0, 0, 0, 1],
                    'angular_velocity': [0, 0, 0],
                    'force': [0, 0, 0],
                    'uprightness': 1.0,
                    'contact_count': 0,
                    'is_grounded': False,
                    'visible_from_origin': True,
                    'max_impact_force': 0
                })
        
        return {'objects': all_objects}
    
    def _check_visibility(self, target_id: int, all_objects: List[int], target_idx: int) -> bool:
        """Check if target is visible from observer position."""
        if target_idx == 0:
            return True
        
        try:
            target_pos, _ = p.getBasePositionAndOrientation(target_id, physicsClientId=self.client)
            
            ray_result = p.rayTest(
                self.observer_position,
                target_pos,
                physicsClientId=self.client
            )
            
            if ray_result and ray_result[0][0] != -1:
                hit_id = ray_result[0][0]
                return hit_id == target_id
            
            return True
            
        except:
            return True


def generate_robotics_schema_data(
    schema: str,
    num_episodes: int,
    output_dir: Path,
    use_gui: bool = False
):
    """Generate episodes for a robotics schema."""
    schema_file = output_dir / f"{schema}.json"
    
    timesteps = get_timesteps_for_robotics_schema(schema)
    print(f"Generating {num_episodes:,} episodes for '{schema}' ({timesteps} timesteps each)...")
    
    generator = RoboticsPhysicsGenerator(use_gui=use_gui)
    
    try:
        with open(schema_file, 'w', encoding='utf-8') as f:
            f.write('[')
            
            for episode_id in tqdm(range(num_episodes), desc=schema, leave=False):
                try:
                    episode = generator.generate_episode(schema, episode_id, num_steps=timesteps)
                    
                    if episode_id > 0:
                        f.write(',')
                    json.dump(episode, f)
                    
                    if episode_id % 100 == 0:
                        f.flush()
                        gc.collect()
                    
                except Exception as e:
                    print(f"\nError on episode {episode_id}: {e}")
                    continue
            
            f.write(']')
        
        print(f"  Saved {num_episodes:,} episodes to {schema_file}")
        
    finally:
        generator.cleanup()
        gc.collect()


def main():
    print("=" * 70)
    print("DEPRECATED: Use generate_robotics_hdf5.py instead")
    print("=" * 70)
    print("\nThis script outputs JSON which causes memory issues.")
    print("Use the HDF5 version for better performance:\n")
    print("  python generate_robotics_hdf5.py --episodes 1200")
    print("\nOr use generate_all_data.py which handles everything:\n")
    print("  python generate_all_data.py --physics-only")
    print("=" * 70)


if __name__ == "__main__":
    main()
