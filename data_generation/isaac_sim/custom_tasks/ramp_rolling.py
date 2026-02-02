# Copyright (c) 2026 Anonymous. All rights reserved.
# Author: Anonymous
#
# PROPRIETARY AND CONFIDENTIAL. This software is provided for academic review
# and research purposes only. Unauthorized copying, modification, distribution,
# or use of this software, via any medium, is strictly prohibited without prior
# written permission from Anonymous.
#
# Physics variety task for PhysicsFormer training
#
# RampRolling: Objects roll/slide down inclined ramp with varying friction
# Physics concepts: inclined plane, rolling vs sliding, friction, gravity

import math
import numpy as np
import torch
from omni.isaac.core.objects import DynamicSphere, DynamicCuboid, FixedCuboid
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch.maths import *
from omniisaacgymenvs.tasks.base.rl_task import RLTask


class RampRollingTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)
        
        self._num_observations = 13 * self._num_objects
        self._num_actions = 0
        
        RLTask.__init__(self, name, env)
        return

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._dt = self._task_cfg["sim"]["dt"]
        
        self._num_objects = self._task_cfg["env"].get("numObjects", 4)
        self._ramp_angle = self._task_cfg["env"].get("rampAngle", 30)
        self._ramp_length = self._task_cfg["env"].get("rampLength", 1.0)
        self._object_radius = self._task_cfg["env"].get("objectRadius", 0.05)
        self._max_episode_length = self._task_cfg["env"].get("maxEpisodeLength", 200)

    def set_up_scene(self, scene) -> None:
        self._create_ramp()
        self._create_rolling_objects()
        super().set_up_scene(scene, replicate_physics=False)
        
        self._ramp = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Ramp",
            name="ramp_view",
            reset_xform_properties=False
        )
        scene.add(self._ramp)
        
        self._rolling_objects = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Objects/*",
            name="rolling_objects_view",
            reset_xform_properties=False
        )
        scene.add(self._rolling_objects)
        return

    def _create_ramp(self):
        angle_rad = math.radians(self._ramp_angle)
        ramp_height = self._ramp_length * math.sin(angle_rad)
        ramp_horizontal = self._ramp_length * math.cos(angle_rad)
        
        from pxr import UsdGeom, Gf
        from omni.isaac.core.utils.stage import get_current_stage
        
        stage = get_current_stage()
        ramp_path = self.default_zero_env_path + "/Ramp"
        
        ramp = FixedCuboid(
            prim_path=ramp_path,
            translation=torch.tensor([0, 0, ramp_height/2]),
            scale=torch.tensor([self._ramp_length, 0.5, 0.02]),
            color=torch.tensor([0.5, 0.5, 0.6]),
        )
        
        ramp_prim = stage.GetPrimAtPath(ramp_path)
        xformable = UsdGeom.Xformable(ramp_prim)
        xformable.AddRotateYOp().Set(-self._ramp_angle)

    def _create_rolling_objects(self):
        objects_path = self.default_zero_env_path + "/Objects"
        angle_rad = math.radians(self._ramp_angle)
        
        colors = [
            [0.9, 0.2, 0.2], [0.2, 0.9, 0.2], [0.2, 0.2, 0.9], [0.9, 0.9, 0.2],
        ]
        
        frictions = [0.2, 0.4, 0.6, 0.8]
        
        for i in range(self._num_objects):
            offset_along_ramp = 0.8 - i * 0.15
            x_pos = offset_along_ramp * math.cos(angle_rad)
            z_pos = offset_along_ramp * math.sin(angle_rad) + self._object_radius + 0.01
            y_pos = (i - self._num_objects/2) * 0.1
            
            color = colors[i % len(colors)]
            
            if i % 2 == 0:
                obj = DynamicSphere(
                    prim_path=f"{objects_path}/obj_{i}",
                    translation=torch.tensor([x_pos, y_pos, z_pos]),
                    radius=self._object_radius,
                    color=torch.tensor(color),
                    mass=0.5,
                )
            else:
                obj = DynamicCuboid(
                    prim_path=f"{objects_path}/obj_{i}",
                    translation=torch.tensor([x_pos, y_pos, z_pos]),
                    scale=torch.tensor([self._object_radius*2, self._object_radius*2, self._object_radius*2]),
                    color=torch.tensor(color),
                    mass=0.5,
                )
            
            self._sim_config.apply_articulation_settings(
                "rolling_object", get_prim_at_path(obj.prim_path),
                self._sim_config.parse_actor_config("rolling_object")
            )

    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists("ramp_view"):
            scene.remove_object("ramp_view", registry_only=True)
        if scene.object_exists("rolling_objects_view"):
            scene.remove_object("rolling_objects_view", registry_only=True)
            
        self._ramp = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Ramp",
            name="ramp_view",
            reset_xform_properties=False
        )
        scene.add(self._ramp)
        
        self._rolling_objects = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Objects/*",
            name="rolling_objects_view",
            reset_xform_properties=False
        )
        scene.add(self._rolling_objects)

    def get_observations(self) -> dict:
        obj_pos, obj_rot = self._rolling_objects.get_world_poses()
        obj_vel = self._rolling_objects.get_velocities()
        
        self.obs_buf = torch.cat([
            obj_pos.view(self._num_envs, -1),
            obj_rot.view(self._num_envs, -1),
            obj_vel[:, :3].view(self._num_envs, -1),
        ], dim=-1)
        
        observations = {self._name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions) -> None:
        pass

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        
        angle_variation = self._ramp_angle + (torch.rand(1, device=self._device).item() - 0.5) * 10
        angle_rad = math.radians(angle_variation)
        
        obj_indices = self._rolling_objects.get_env_indices(env_ids)
        
        start_offset_variation = 0.6 + torch.rand(1, device=self._device).item() * 0.4
        spacing_variation = 0.1 + torch.rand(1, device=self._device).item() * 0.1
        
        new_positions = torch.zeros((num_resets * self._num_objects, 3), device=self._device)
        new_orientations = torch.zeros((num_resets * self._num_objects, 4), device=self._device)
        new_orientations[:, 3] = 1.0
        
        for i in range(self._num_objects):
            idx_start = i * num_resets
            idx_end = (i + 1) * num_resets
            
            offset_along_ramp = start_offset_variation - i * spacing_variation
            pos_noise_x = (torch.rand(num_resets, device=self._device) - 0.5) * 0.02
            pos_noise_y = (torch.rand(num_resets, device=self._device) - 0.5) * 0.02
            new_positions[idx_start:idx_end, 0] = offset_along_ramp * math.cos(angle_rad) + pos_noise_x
            new_positions[idx_start:idx_end, 1] = (i - self._num_objects/2) * 0.1 + pos_noise_y
            new_positions[idx_start:idx_end, 2] = offset_along_ramp * math.sin(angle_rad) + self._object_radius + 0.01
        
        initial_velocities = torch.zeros((num_resets * self._num_objects, 6), device=self._device)
        initial_velocities[:, 0] = (torch.rand(num_resets * self._num_objects, device=self._device) - 0.5) * 0.3
        initial_velocities[:, 1] = (torch.rand(num_resets * self._num_objects, device=self._device) - 0.5) * 0.3
        
        self._rolling_objects.set_world_poses(new_positions, new_orientations, indices=obj_indices)
        self._rolling_objects.set_velocities(initial_velocities, indices=obj_indices)
        
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        pass

    def calculate_metrics(self) -> None:
        self.rew_buf[:] = 0

    def is_done(self) -> None:
        self.reset_buf = torch.where(
            self.progress_buf >= self._max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf
        )
