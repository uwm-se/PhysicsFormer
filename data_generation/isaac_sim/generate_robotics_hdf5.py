"""
Copyright (c) 2026 Anonymous. All rights reserved.
Author: Anonymous

PROPRIETARY AND CONFIDENTIAL. This software is provided for academic review
and research purposes only. Unauthorized copying, modification, distribution,
or use of this software, via any medium, is strictly prohibited without prior
written permission from Anonymous.
"""

"""
Generate Robotics Schema Data Directly to HDF5

Bypasses JSON intermediate format to avoid memory issues.
Generates data in batches and writes directly to HDF5.
"""

import sys
from pathlib import Path

helpers_dir = Path(__file__).parent / "helpers"
if str(helpers_dir) not in sys.path:
    sys.path.insert(0, str(helpers_dir))

import numpy as np
import h5py
import argparse
from tqdm import tqdm
import pybullet as p
import pybullet_data
import gc
from typing import List, Dict, Tuple
from enum import Enum


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

MAX_SEQ_LENGTH = 512
MAX_OBJECTS = 20
STATE_DIM = 21


def get_timesteps_for_schema(schema: str) -> int:
    timestep_config = {
        "object_pickup": 150,
        "fragile_objects": 100,
        "occlusion_static": 50,
        "object_placement": 150,
        "rotation_manipulation": 200,
    }
    return timestep_config.get(schema, 100)


class RoboticsGenerator:
    def __init__(self):
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
    
    def _create_ground(self) -> int:
        ground_shape = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self.client)
        ground = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=ground_shape, physicsClientId=self.client)
        p.changeDynamics(ground, -1, lateralFriction=0.5, physicsClientId=self.client)
        return ground
    
    def _create_object(self, shape_type: ShapeType, position: List[float], 
                       dimensions: List[float], material: MaterialType,
                       velocity: List[float] = None, color: List[float] = None) -> Tuple[int, Dict]:
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
        
        p.changeDynamics(obj_id, -1, lateralFriction=mat_props["friction"],
                        restitution=mat_props["restitution"], physicsClientId=self.client)
        
        if velocity:
            p.resetBaseVelocity(obj_id, linearVelocity=velocity, physicsClientId=self.client)
        
        max_dim = max(dimensions)
        graspable = bool(max_dim < 0.3 and mass < 10.0)
        
        properties = {
            "mass": float(mass),
            "radius": float(dimensions[0]),
            "color": color if color else [0.5, 0.5, 0.5],
            "graspable": graspable,
            "is_static": mass == 0,
            "shape_type": 0 if shape_type == ShapeType.SPHERE else (1 if shape_type == ShapeType.BOX else 2),
        }
        
        return obj_id, properties
    
    def _get_object_state(self, obj_id: int, properties: Dict) -> np.ndarray:
        """Get 21-dim state vector for object."""
        state = np.zeros(STATE_DIM, dtype=np.float32)
        
        pos, orn = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.client)
        vel, ang_vel = p.getBaseVelocity(obj_id, physicsClientId=self.client)
        
        state[0:3] = pos
        state[3:6] = vel
        state[6:10] = orn
        state[10:13] = ang_vel
        state[13] = properties.get("mass", 1.0)
        state[14] = properties.get("radius", 0.1)
        color = properties.get("color", [0.5, 0.5, 0.5])
        state[15:18] = color[:3]
        state[18] = properties.get("shape_type", 0)
        state[19] = 1.0 if properties.get("is_static", False) else 0.0
        state[20] = 1.0 if properties.get("graspable", False) else 0.0
        
        return state
    
    def _setup_object_pickup(self) -> Tuple[List[int], List[Dict]]:
        objects = []
        properties = []
        
        ground = self._create_ground()
        objects.append(ground)
        properties.append({"mass": 0, "radius": 10.0, "color": [0.5, 0.5, 0.5], 
                          "graspable": False, "is_static": True, "shape_type": 1})
        
        num_objects = np.random.randint(3, 8)
        for i in range(num_objects):
            x = np.random.uniform(-0.5, 0.5)
            y = np.random.uniform(-0.5, 0.5)
            z = np.random.uniform(0.1, 0.3)
            
            shape = np.random.choice([ShapeType.SPHERE, ShapeType.BOX, ShapeType.CYLINDER])
            material = np.random.choice(list(MaterialType))
            size = np.random.uniform(0.03, 0.15)
            
            if shape == ShapeType.BOX:
                dims = [size, size * np.random.uniform(0.5, 1.5), size * np.random.uniform(0.5, 1.5)]
            elif shape == ShapeType.CYLINDER:
                dims = [size * 0.5, size]
            else:
                dims = [size]
            
            color = [np.random.random(), np.random.random(), np.random.random()]
            obj_id, props = self._create_object(shape, [x, y, z], dims, material, color=color)
            objects.append(obj_id)
            properties.append(props)
        
        return objects, properties
    
    def _setup_fragile_objects(self) -> Tuple[List[int], List[Dict]]:
        objects = []
        properties = []
        
        ground = self._create_ground()
        objects.append(ground)
        properties.append({"mass": 0, "radius": 10.0, "color": [0.5, 0.5, 0.5],
                          "graspable": False, "is_static": True, "shape_type": 1})
        
        fragile_materials = [MaterialType.GLASS, MaterialType.CERAMIC]
        robust_materials = [MaterialType.METAL, MaterialType.PLASTIC]
        
        for i in range(np.random.randint(2, 5)):
            x = np.random.uniform(-0.3, 0.3)
            y = np.random.uniform(-0.3, 0.3)
            z = 0.3 + i * 0.1
            
            material = np.random.choice(fragile_materials)
            size = np.random.uniform(0.03, 0.08)
            color = [0.8, 0.9, 1.0] if material == MaterialType.GLASS else [0.9, 0.85, 0.8]
            
            obj_id, props = self._create_object(ShapeType.SPHERE, [x, y, z], [size], material, color=color)
            objects.append(obj_id)
            properties.append(props)
        
        for i in range(np.random.randint(2, 4)):
            x = np.random.uniform(-0.5, 0.5)
            y = np.random.uniform(-0.5, 0.5)
            z = 0.1
            
            material = np.random.choice(robust_materials)
            size = np.random.uniform(0.05, 0.12)
            color = [0.6, 0.6, 0.7] if material == MaterialType.METAL else [0.9, 0.7, 0.3]
            
            obj_id, props = self._create_object(ShapeType.BOX, [x, y, z], [size, size, size], material, color=color)
            objects.append(obj_id)
            properties.append(props)
        
        return objects, properties
    
    def _setup_occlusion_static(self) -> Tuple[List[int], List[Dict]]:
        objects = []
        properties = []
        
        ground = self._create_ground()
        objects.append(ground)
        properties.append({"mass": 0, "radius": 10.0, "color": [0.5, 0.5, 0.5],
                          "graspable": False, "is_static": True, "shape_type": 1})
        
        wall_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.02, 0.25], physicsClientId=self.client)
        wall = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_shape, 
                                basePosition=[0, 0, 0.25], physicsClientId=self.client)
        objects.append(wall)
        properties.append({"mass": 0, "radius": 0.3, "color": [0.3, 0.3, 0.3],
                          "graspable": False, "is_static": True, "shape_type": 1})
        
        for i in range(np.random.randint(2, 5)):
            x = np.random.uniform(-0.2, 0.2)
            y = np.random.uniform(0.1, 0.4)
            z = np.random.uniform(0.05, 0.2)
            
            size = np.random.uniform(0.03, 0.08)
            color = [np.random.random(), np.random.random(), np.random.random()]
            
            obj_id, props = self._create_object(ShapeType.SPHERE, [x, y, z], [size], 
                                               MaterialType.PLASTIC, color=color)
            objects.append(obj_id)
            properties.append(props)
        
        for i in range(np.random.randint(1, 3)):
            x = np.random.uniform(-0.3, 0.3)
            y = np.random.uniform(-0.4, -0.1)
            z = np.random.uniform(0.05, 0.15)
            
            size = np.random.uniform(0.04, 0.1)
            color = [np.random.random(), np.random.random(), np.random.random()]
            
            obj_id, props = self._create_object(ShapeType.BOX, [x, y, z], [size, size, size],
                                               MaterialType.WOOD, color=color)
            objects.append(obj_id)
            properties.append(props)
        
        return objects, properties
    
    def _setup_object_placement(self) -> Tuple[List[int], List[Dict]]:
        objects = []
        properties = []
        
        ground = self._create_ground()
        objects.append(ground)
        properties.append({"mass": 0, "radius": 10.0, "color": [0.5, 0.5, 0.5],
                          "graspable": False, "is_static": True, "shape_type": 1})
        
        table_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.2, 0.01], physicsClientId=self.client)
        table = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=table_shape,
                                 basePosition=[0, 0, 0.25], physicsClientId=self.client)
        objects.append(table)
        properties.append({"mass": 0, "radius": 0.3, "color": [0.6, 0.4, 0.2],
                          "graspable": False, "is_static": True, "shape_type": 1})
        
        for i in range(np.random.randint(2, 5)):
            x = np.random.uniform(-0.2, 0.2)
            y = np.random.uniform(-0.15, 0.15)
            z = 0.3 + i * 0.08
            
            size = np.random.uniform(0.03, 0.07)
            color = [np.random.random(), np.random.random(), np.random.random()]
            
            obj_id, props = self._create_object(ShapeType.BOX, [x, y, z], [size, size, size],
                                               MaterialType.PLASTIC, color=color)
            objects.append(obj_id)
            properties.append(props)
        
        return objects, properties
    
    def _setup_rotation_manipulation(self) -> Tuple[List[int], List[Dict]]:
        objects = []
        properties = []
        
        ground = self._create_ground()
        objects.append(ground)
        properties.append({"mass": 0, "radius": 10.0, "color": [0.5, 0.5, 0.5],
                          "graspable": False, "is_static": True, "shape_type": 1})
        
        for i in range(np.random.randint(2, 5)):
            x = np.random.uniform(-0.3, 0.3)
            y = np.random.uniform(-0.3, 0.3)
            z = 0.1 + i * 0.1
            
            w = np.random.uniform(0.06, 0.12)
            h = np.random.uniform(0.03, 0.06)
            d = np.random.uniform(0.03, 0.06)
            
            color = [np.random.random(), np.random.random(), np.random.random()]
            
            obj_id, props = self._create_object(ShapeType.BOX, [x, y, z], [w, h, d],
                                               MaterialType.WOOD, color=color)
            
            ang_vel = [np.random.uniform(-2, 2) for _ in range(3)]
            p.resetBaseVelocity(obj_id, angularVelocity=ang_vel, physicsClientId=self.client)
            
            objects.append(obj_id)
            properties.append(props)
        
        return objects, properties
    
    def generate_episode(self, schema: str, episode_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate episode and return states and masks arrays."""
        seed = (hash(schema) % 1000000) + episode_id
        np.random.seed(seed % (2**32))
        
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        
        if schema == "object_pickup":
            objects, properties = self._setup_object_pickup()
        elif schema == "fragile_objects":
            objects, properties = self._setup_fragile_objects()
        elif schema == "occlusion_static":
            objects, properties = self._setup_occlusion_static()
        elif schema == "object_placement":
            objects, properties = self._setup_object_placement()
        elif schema == "rotation_manipulation":
            objects, properties = self._setup_rotation_manipulation()
        else:
            objects, properties = self._setup_object_pickup()
        
        num_steps = get_timesteps_for_schema(schema)
        
        states = np.zeros((MAX_SEQ_LENGTH, MAX_OBJECTS, STATE_DIM), dtype=np.float32)
        masks = np.zeros((MAX_SEQ_LENGTH, MAX_OBJECTS), dtype=np.float32)
        
        num_objects = min(len(objects), MAX_OBJECTS)
        
        for step in range(min(num_steps, MAX_SEQ_LENGTH)):
            p.stepSimulation(physicsClientId=self.client)
            
            for obj_idx in range(num_objects):
                states[step, obj_idx] = self._get_object_state(objects[obj_idx], properties[obj_idx])
                masks[step, obj_idx] = 1.0
        
        for obj_id in objects:
            try:
                p.removeBody(obj_id, physicsClientId=self.client)
            except:
                pass
        
        return states, masks


def generate_schema_hdf5(schema: str, num_episodes: int, output_dir: Path, batch_size: int = 1000):
    """Generate HDF5 file for a schema."""
    output_file = output_dir / f"{schema}.h5"
    
    print(f"\nGenerating {schema}: {num_episodes:,} episodes")
    print(f"  Output: {output_file}")
    
    generator = RoboticsGenerator()
    
    try:
        with h5py.File(output_file, 'w') as f:
            states_ds = f.create_dataset(
                'states', 
                shape=(num_episodes, MAX_SEQ_LENGTH, MAX_OBJECTS, STATE_DIM),
                dtype='float32',
                chunks=(min(50, num_episodes), MAX_SEQ_LENGTH, MAX_OBJECTS, STATE_DIM),
                compression='gzip',
                compression_opts=4
            )
            
            masks_ds = f.create_dataset(
                'masks',
                shape=(num_episodes, MAX_SEQ_LENGTH, MAX_OBJECTS),
                dtype='float32',
                chunks=(min(50, num_episodes), MAX_SEQ_LENGTH, MAX_OBJECTS),
                compression='gzip',
                compression_opts=4
            )
            
            next_states_ds = f.create_dataset(
                'next_states',
                shape=(num_episodes, MAX_SEQ_LENGTH - 1, MAX_OBJECTS, STATE_DIM),
                dtype='float32',
                chunks=(min(50, num_episodes), MAX_SEQ_LENGTH - 1, MAX_OBJECTS, STATE_DIM),
                compression='gzip',
                compression_opts=4
            )
            
            dt = h5py.special_dtype(vlen=str)
            schemas_ds = f.create_dataset('schemas', shape=(num_episodes,), dtype=dt)
            
            for batch_start in tqdm(range(0, num_episodes, batch_size), desc=f"  {schema}"):
                batch_end = min(batch_start + batch_size, num_episodes)
                batch_states = []
                batch_masks = []
                
                for ep_idx in range(batch_start, batch_end):
                    states, masks = generator.generate_episode(schema, ep_idx)
                    batch_states.append(states)
                    batch_masks.append(masks)
                    schemas_ds[ep_idx] = schema
                
                batch_states = np.array(batch_states)
                batch_masks = np.array(batch_masks)
                
                states_ds[batch_start:batch_end] = batch_states
                masks_ds[batch_start:batch_end] = batch_masks
                next_states_ds[batch_start:batch_end] = batch_states[:, 1:, :, :]
                
                gc.collect()
        
        print(f"  ✓ Saved {num_episodes:,} episodes")
        
    finally:
        generator.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Generate robotics HDF5 data")
    parser.add_argument("--episodes", type=int, default=1200, help="Episodes per schema")
    parser.add_argument("--output-dir", type=str, default="D:/physics_hdf5")
    parser.add_argument("--schema", type=str, default=None, help="Single schema to generate")
    parser.add_argument("--batch-size", type=int, default=100)
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    schemas = [args.schema] if args.schema else ROBOTICS_SCHEMAS
    
    print("=" * 60)
    print("ROBOTICS HDF5 GENERATION")
    print("=" * 60)
    print(f"Schemas: {schemas}")
    print(f"Episodes per schema: {args.episodes:,}")
    print(f"Output: {output_dir}")
    
    for schema in schemas:
        generate_schema_hdf5(schema, args.episodes, output_dir, args.batch_size)
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
