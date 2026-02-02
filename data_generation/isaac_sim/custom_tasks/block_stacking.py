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
# BlockStacking: Blocks stacked and tested for stability under perturbation
# Physics concepts: stability, gravity, friction, center of mass

import gc
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch.maths import *
from omniisaacgymenvs.tasks.base.rl_task import RLTask


class BlockStackingTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)
        
        self._num_observations = 13 * self._num_blocks
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
        
        self._num_blocks = self._task_cfg["env"].get("numBlocks", 8)
        self._block_size = self._task_cfg["env"].get("blockSize", 0.05)
        self._stack_noise = self._task_cfg["env"].get("stackNoise", 0.01)
        self._max_episode_length = self._task_cfg["env"].get("maxEpisodeLength", 300)

    def set_up_scene(self, scene) -> None:
        self._create_ground()
        self._create_blocks()
        super().set_up_scene(scene, replicate_physics=False)
        
        self._ground = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Ground",
            name="ground_view",
            reset_xform_properties=False
        )
        scene.add(self._ground)
        
        self._blocks = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Blocks/*",
            name="blocks_view",
            reset_xform_properties=False
        )
        scene.add(self._blocks)
        return

    def _create_ground(self):
        FixedCuboid(
            prim_path=self.default_zero_env_path + "/Ground",
            translation=torch.tensor([0, 0, -0.01]),
            scale=torch.tensor([1.0, 1.0, 0.02]),
            color=torch.tensor([0.3, 0.3, 0.3]),
        )

    def _create_blocks(self):
        blocks_path = self.default_zero_env_path + "/Blocks"
        colors = [
            [0.8, 0.2, 0.2], [0.2, 0.8, 0.2], [0.2, 0.2, 0.8],
            [0.8, 0.8, 0.2], [0.8, 0.2, 0.8], [0.2, 0.8, 0.8],
            [0.9, 0.5, 0.2], [0.5, 0.2, 0.9],
        ]
        
        for i in range(self._num_blocks):
            x_noise = (np.random.rand() - 0.5) * self._stack_noise
            y_noise = (np.random.rand() - 0.5) * self._stack_noise
            z_pos = self._block_size / 2 + i * self._block_size + 0.001
            
            color = colors[i % len(colors)]
            
            block = DynamicCuboid(
                prim_path=f"{blocks_path}/block_{i}",
                translation=torch.tensor([x_noise, y_noise, z_pos]),
                scale=torch.tensor([self._block_size, self._block_size, self._block_size]),
                color=torch.tensor(color),
                mass=0.5,
            )
            
            self._sim_config.apply_articulation_settings(
                "block", get_prim_at_path(block.prim_path),
                self._sim_config.parse_actor_config("block")
            )

    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists("ground_view"):
            scene.remove_object("ground_view", registry_only=True)
        if scene.object_exists("blocks_view"):
            scene.remove_object("blocks_view", registry_only=True)
            
        self._ground = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Ground",
            name="ground_view",
            reset_xform_properties=False
        )
        scene.add(self._ground)
        
        self._blocks = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Blocks/*",
            name="blocks_view",
            reset_xform_properties=False
        )
        scene.add(self._blocks)

    def get_observations(self) -> dict:
        block_pos, block_rot = self._blocks.get_world_poses()
        block_vel = self._blocks.get_velocities()
        
        self.obs_buf = torch.cat([
            block_pos.view(self._num_envs, -1),
            block_rot.view(self._num_envs, -1),
            block_vel[:, :3].view(self._num_envs, -1),
        ], dim=-1)
        
        observations = {self._name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions) -> None:
        perturbation_step = 30 + int(torch.rand(1, device=self._device).item() * 40)
        if self.progress_buf[0] == perturbation_step:
            perturbation = torch.zeros((self._num_envs * self._num_blocks, 6), device=self._device)
            perturbation_strength = 0.3 + torch.rand(1, device=self._device).item() * 0.7
            perturbation[:, 0] = (torch.rand(self._num_envs * self._num_blocks, device=self._device) - 0.5) * perturbation_strength
            perturbation[:, 1] = (torch.rand(self._num_envs * self._num_blocks, device=self._device) - 0.5) * perturbation_strength
            perturbation[:, 2] = (torch.rand(self._num_envs * self._num_blocks, device=self._device) - 0.5) * 0.2
            
            try:
                current_vel = self._blocks.get_velocities()
                self._blocks.set_velocities(current_vel + perturbation)
            except Exception as e:
                logger.warning(f"Perturbation failed: {e}")

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        
        try:
            block_indices = self._blocks.get_env_indices(env_ids)
            
            stack_noise_variation = self._stack_noise * (0.5 + torch.rand(1, device=self._device).item() * 1.0)
            
            new_positions = torch.zeros((num_resets * self._num_blocks, 3), device=self._device)
            new_orientations = torch.zeros((num_resets * self._num_blocks, 4), device=self._device)
            new_orientations[:, 3] = 1.0
            
            for i in range(self._num_blocks):
                idx_start = i * num_resets
                idx_end = (i + 1) * num_resets
                new_positions[idx_start:idx_end, 0] = (torch.rand(num_resets, device=self._device) - 0.5) * stack_noise_variation
                new_positions[idx_start:idx_end, 1] = (torch.rand(num_resets, device=self._device) - 0.5) * stack_noise_variation
                new_positions[idx_start:idx_end, 2] = self._block_size / 2 + i * self._block_size + 0.001
                
                tilt_angle = (torch.rand(num_resets, device=self._device) - 0.5) * 0.05
                new_orientations[idx_start:idx_end, 0] = torch.sin(tilt_angle / 2)
                new_orientations[idx_start:idx_end, 3] = torch.cos(tilt_angle / 2)
            
            self._blocks.set_world_poses(new_positions, new_orientations, indices=block_indices)
            self._blocks.set_velocities(
                torch.zeros((num_resets * self._num_blocks, 6), device=self._device),
                indices=block_indices
            )
            
            self._check_memory_usage()
        except Exception as e:
            logger.warning(f"Reset failed: {e}")
        
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

    def _check_memory_usage(self):
        if not torch.cuda.is_available():
            return
        try:
            allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            if allocated_gb > 8.0:
                logger.warning(f"High GPU memory: {allocated_gb:.2f}GB")
                gc.collect()
                torch.cuda.empty_cache()
        except Exception:
            pass
