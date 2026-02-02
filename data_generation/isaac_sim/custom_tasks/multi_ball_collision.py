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
# MultiBallCollision: Multiple balls colliding on a surface like billiards
# Physics concepts: 2D collision, momentum transfer, friction

import math
import numpy as np
import torch
from omni.isaac.core.objects import DynamicSphere, FixedCuboid
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch.maths import *
from omniisaacgymenvs.tasks.base.rl_task import RLTask


class MultiBallCollisionTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)
        
        self._num_observations = 13 * self._num_balls
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
        
        self._num_balls = self._task_cfg["env"].get("numBalls", 10)
        self._ball_radius = self._task_cfg["env"].get("ballRadius", 0.03)
        self._table_size = self._task_cfg["env"].get("tableSize", [1.0, 0.5])
        self._max_episode_length = self._task_cfg["env"].get("maxEpisodeLength", 300)

    def set_up_scene(self, scene) -> None:
        self._create_table()
        self._create_balls()
        super().set_up_scene(scene, replicate_physics=False)
        
        self._table = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Table/*",
            name="table_view",
            reset_xform_properties=False
        )
        scene.add(self._table)
        
        self._balls = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Balls/*",
            name="balls_view",
            reset_xform_properties=False
        )
        scene.add(self._balls)
        return

    def _create_table(self):
        table_path = self.default_zero_env_path + "/Table"
        tx, ty = self._table_size
        wall_height = 0.05
        wall_thickness = 0.02
        
        FixedCuboid(
            prim_path=f"{table_path}/surface",
            translation=torch.tensor([0, 0, -0.01]),
            scale=torch.tensor([tx, ty, 0.02]),
            color=torch.tensor([0.1, 0.5, 0.2]),
        )
        
        FixedCuboid(
            prim_path=f"{table_path}/wall_left",
            translation=torch.tensor([-tx/2, 0, wall_height/2]),
            scale=torch.tensor([wall_thickness, ty, wall_height]),
            color=torch.tensor([0.4, 0.2, 0.1]),
        )
        
        FixedCuboid(
            prim_path=f"{table_path}/wall_right",
            translation=torch.tensor([tx/2, 0, wall_height/2]),
            scale=torch.tensor([wall_thickness, ty, wall_height]),
            color=torch.tensor([0.4, 0.2, 0.1]),
        )
        
        FixedCuboid(
            prim_path=f"{table_path}/wall_front",
            translation=torch.tensor([0, ty/2, wall_height/2]),
            scale=torch.tensor([tx, wall_thickness, wall_height]),
            color=torch.tensor([0.4, 0.2, 0.1]),
        )
        
        FixedCuboid(
            prim_path=f"{table_path}/wall_back",
            translation=torch.tensor([0, -ty/2, wall_height/2]),
            scale=torch.tensor([tx, wall_thickness, wall_height]),
            color=torch.tensor([0.4, 0.2, 0.1]),
        )

    def _create_balls(self):
        balls_path = self.default_zero_env_path + "/Balls"
        tx, ty = self._table_size
        
        colors = [
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0],
            [0.5, 0.0, 0.5], [1.0, 0.5, 0.0], [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.7, 0.8],
        ]
        
        for i in range(self._num_balls):
            if i == 0:
                x_pos = -tx/3
                y_pos = 0
            else:
                row = int(math.sqrt(2 * (i - 1) + 0.25) - 0.5)
                col = i - 1 - row * (row + 1) // 2
                x_pos = tx/4 + row * self._ball_radius * 1.8
                y_pos = (col - row/2) * self._ball_radius * 2.1
            
            color = colors[i % len(colors)]
            
            ball = DynamicSphere(
                prim_path=f"{balls_path}/ball_{i}",
                translation=torch.tensor([x_pos, y_pos, self._ball_radius + 0.001]),
                radius=self._ball_radius,
                color=torch.tensor(color),
                mass=0.3,
            )
            
            self._sim_config.apply_articulation_settings(
                "ball", get_prim_at_path(ball.prim_path),
                self._sim_config.parse_actor_config("ball")
            )

    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists("table_view"):
            scene.remove_object("table_view", registry_only=True)
        if scene.object_exists("balls_view"):
            scene.remove_object("balls_view", registry_only=True)
            
        self._table = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Table/*",
            name="table_view",
            reset_xform_properties=False
        )
        scene.add(self._table)
        
        self._balls = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Balls/*",
            name="balls_view",
            reset_xform_properties=False
        )
        scene.add(self._balls)

    def get_observations(self) -> dict:
        ball_pos, ball_rot = self._balls.get_world_poses()
        ball_vel = self._balls.get_velocities()
        
        self.obs_buf = torch.cat([
            ball_pos.view(self._num_envs, -1),
            ball_rot.view(self._num_envs, -1),
            ball_vel[:, :3].view(self._num_envs, -1),
        ], dim=-1)
        
        observations = {self._name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions) -> None:
        if self.progress_buf[0] == 1:
            cue_ball_indices = torch.arange(
                0, self._num_envs * self._num_balls, self._num_balls,
                device=self._device
            )
            
            strike_velocity = torch.zeros((len(cue_ball_indices), 6), device=self._device)
            base_speed = 2.0 + torch.rand(1, device=self._device).item() * 1.5
            strike_velocity[:, 0] = base_speed + torch.rand(len(cue_ball_indices), device=self._device) * 1.0
            strike_velocity[:, 1] = (torch.rand(len(cue_ball_indices), device=self._device) - 0.5) * 0.3
            strike_velocity[:, 5] = (torch.rand(len(cue_ball_indices), device=self._device) - 0.5) * 3.0
            
            self._balls.set_velocities(strike_velocity, indices=cue_ball_indices)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        tx, ty = self._table_size
        
        ball_indices = self._balls.get_env_indices(env_ids)
        
        rack_offset_x = (torch.rand(1, device=self._device).item() - 0.5) * 0.1
        rack_offset_y = (torch.rand(1, device=self._device).item() - 0.5) * 0.1
        rack_spread = 0.9 + torch.rand(1, device=self._device).item() * 0.2
        
        new_positions = torch.zeros((num_resets * self._num_balls, 3), device=self._device)
        new_orientations = torch.zeros((num_resets * self._num_balls, 4), device=self._device)
        new_orientations[:, 3] = 1.0
        
        for i in range(self._num_balls):
            idx_start = i * num_resets
            idx_end = (i + 1) * num_resets
            
            if i == 0:
                cue_offset_y = (torch.rand(num_resets, device=self._device) - 0.5) * 0.15
                new_positions[idx_start:idx_end, 0] = -tx/3 + (torch.rand(num_resets, device=self._device) - 0.5) * 0.1
                new_positions[idx_start:idx_end, 1] = cue_offset_y
            else:
                row = int(math.sqrt(2 * (i - 1) + 0.25) - 0.5)
                col = i - 1 - row * (row + 1) // 2
                pos_noise_x = (torch.rand(num_resets, device=self._device) - 0.5) * 0.005
                pos_noise_y = (torch.rand(num_resets, device=self._device) - 0.5) * 0.005
                new_positions[idx_start:idx_end, 0] = tx/4 + rack_offset_x + row * self._ball_radius * 1.8 * rack_spread + pos_noise_x
                new_positions[idx_start:idx_end, 1] = rack_offset_y + (col - row/2) * self._ball_radius * 2.1 * rack_spread + pos_noise_y
            
            new_positions[idx_start:idx_end, 2] = self._ball_radius + 0.001
        
        self._balls.set_world_poses(new_positions, new_orientations, indices=ball_indices)
        self._balls.set_velocities(
            torch.zeros((num_resets * self._num_balls, 6), device=self._device),
            indices=ball_indices
        )
        
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
