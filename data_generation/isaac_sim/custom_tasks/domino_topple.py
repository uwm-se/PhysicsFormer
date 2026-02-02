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
# DominoTopple: Chain reaction of falling dominoes transferring momentum
# Physics concepts: momentum transfer, chain reaction, angular momentum

import gc
import logging

import numpy as np
import torch
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch.maths import *
from omniisaacgymenvs.tasks.base.rl_task import RLTask

logger = logging.getLogger(__name__)


class DominoToppleTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self._initialization_failed = False
        self._error_count = 0
        self._max_errors = 10
        
        try:
            self.update_config(sim_config)
            self._num_observations = 13 * self._num_dominoes
            self._num_actions = 0
            RLTask.__init__(self, name, env)
        except Exception as e:
            logger.error(f"Failed to initialize DominoToppleTask: {e}")
            self._initialization_failed = True
            self._num_observations = 1
            self._num_actions = 0
            self._num_dominoes = 1
            RLTask.__init__(self, name, env)

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._dt = self._task_cfg["sim"]["dt"]
        
        self._num_dominoes = self._task_cfg["env"].get("numDominoes", 15)
        self._domino_size = self._task_cfg["env"].get("dominoSize", [0.01, 0.04, 0.08])
        self._domino_spacing = self._task_cfg["env"].get("dominoSpacing", 0.03)
        self._max_episode_length = self._task_cfg["env"].get("maxEpisodeLength", 400)

    def set_up_scene(self, scene) -> None:
        if self._initialization_failed:
            super().set_up_scene(scene)
            return
            
        try:
            self._create_ground()
            self._create_dominoes()
            super().set_up_scene(scene, replicate_physics=False)
            
            self._dominoes = RigidPrimView(
                prim_paths_expr="/World/envs/.*/Dominoes/*",
                name="dominoes_view",
                reset_xform_properties=False
            )
            scene.add(self._dominoes)
        except Exception as e:
            logger.error(f"Failed to set up scene: {e}")
            self._initialization_failed = True
            super().set_up_scene(scene)

    def _create_ground(self):
        FixedCuboid(
            prim_path=self.default_zero_env_path + "/Ground",
            translation=torch.tensor([0, 0, -0.01]),
            scale=torch.tensor([2.0, 0.5, 0.02]),
            color=torch.tensor([0.3, 0.3, 0.3]),
        )

    def _create_dominoes(self):
        dominoes_path = self.default_zero_env_path + "/Dominoes"
        dx, dy, dz = self._domino_size
        
        total_length = (self._num_dominoes - 1) * self._domino_spacing
        start_x = -total_length / 2
        
        for i in range(self._num_dominoes):
            x_pos = start_x + i * self._domino_spacing
            
            initial_tilt = 0.0
            if i == 0:
                initial_tilt = 0.3
            
            domino = DynamicCuboid(
                prim_path=f"{dominoes_path}/domino_{i}",
                translation=torch.tensor([x_pos, 0, dz / 2 + 0.001]),
                scale=torch.tensor([dx, dy, dz]),
                color=torch.tensor([0.1, 0.1, 0.1]),
                mass=0.2,
            )
            
            self._sim_config.apply_articulation_settings(
                "domino", get_prim_at_path(domino.prim_path),
                self._sim_config.parse_actor_config("domino")
            )

    def initialize_views(self, scene):
        if self._initialization_failed:
            return
            
        try:
            super().initialize_views(scene)
            if scene.object_exists("dominoes_view"):
                scene.remove_object("dominoes_view", registry_only=True)
                
            self._dominoes = RigidPrimView(
                prim_paths_expr="/World/envs/.*/Dominoes/*",
                name="dominoes_view",
                reset_xform_properties=False
            )
            scene.add(self._dominoes)
        except Exception as e:
            logger.error(f"Failed to initialize views: {e}")
            self._initialization_failed = True

    def get_observations(self) -> dict:
        if self._initialization_failed:
            self.obs_buf = torch.zeros((self._num_envs, self._num_observations), device=self._device)
            return {self._name: {"obs_buf": self.obs_buf}}
            
        try:
            domino_pos, domino_rot = self._dominoes.get_world_poses()
            domino_vel = self._dominoes.get_velocities()
            
            if domino_pos is None or domino_rot is None or domino_vel is None:
                raise RuntimeError("Failed to get domino state")
            
            self.obs_buf = torch.cat([
                domino_pos.view(self._num_envs, -1),
                domino_rot.view(self._num_envs, -1),
                domino_vel[:, :3].view(self._num_envs, -1),
            ], dim=-1)
        except Exception as e:
            self._handle_error("get_observations", e)
            self.obs_buf = torch.zeros((self._num_envs, self._num_observations), device=self._device)
        
        return {self._name: {"obs_buf": self.obs_buf}}

    def pre_physics_step(self, actions) -> None:
        if self._initialization_failed:
            return
            
        try:
            if self.progress_buf[0] == 1:
                first_domino_indices = torch.arange(
                    0, self._num_envs * self._num_dominoes, self._num_dominoes,
                    device=self._device
                )
                
                push_force = torch.zeros((len(first_domino_indices), 6), device=self._device)
                push_strength = 4.0 + torch.rand(len(first_domino_indices), device=self._device) * 2.0
                push_force[:, 0] = push_strength
                push_force[:, 1] = (torch.rand(len(first_domino_indices), device=self._device) - 0.5) * 0.2
                
                self._dominoes.set_velocities(push_force, indices=first_domino_indices)
        except Exception as e:
            self._handle_error("pre_physics_step", e)

    def reset_idx(self, env_ids):
        if self._initialization_failed:
            self.reset_buf[env_ids] = 0
            self.progress_buf[env_ids] = 0
            return
            
        try:
            num_resets = len(env_ids)
            dx, dy, dz = self._domino_size
            
            domino_indices = self._dominoes.get_env_indices(env_ids)
            
            spacing_variation = self._domino_spacing * (0.8 + torch.rand(1, device=self._device).item() * 0.4)
            total_length = (self._num_dominoes - 1) * spacing_variation
            start_x = -total_length / 2
            
            new_positions = torch.zeros((num_resets * self._num_dominoes, 3), device=self._device)
            new_orientations = torch.zeros((num_resets * self._num_dominoes, 4), device=self._device)
            new_orientations[:, 3] = 1.0
            
            for i in range(self._num_dominoes):
                idx_start = i * num_resets
                idx_end = (i + 1) * num_resets
                pos_noise_x = (torch.rand(num_resets, device=self._device) - 0.5) * 0.002
                pos_noise_y = (torch.rand(num_resets, device=self._device) - 0.5) * 0.005
                new_positions[idx_start:idx_end, 0] = start_x + i * spacing_variation + pos_noise_x
                new_positions[idx_start:idx_end, 1] = pos_noise_y
                new_positions[idx_start:idx_end, 2] = dz / 2 + 0.001
                
                tilt_angle = (torch.rand(num_resets, device=self._device) - 0.5) * 0.02
                new_orientations[idx_start:idx_end, 1] = torch.sin(tilt_angle / 2)
                new_orientations[idx_start:idx_end, 3] = torch.cos(tilt_angle / 2)
            
            self._dominoes.set_world_poses(new_positions, new_orientations, indices=domino_indices)
            self._dominoes.set_velocities(
                torch.zeros((num_resets * self._num_dominoes, 6), device=self._device),
                indices=domino_indices
            )
            
            self._check_memory_usage()
        except Exception as e:
            self._handle_error("reset_idx", e)
        
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

    def _handle_error(self, method_name: str, error: Exception):
        self._error_count += 1
        logger.warning(f"Error in {method_name}: {error}")
        if self._error_count >= self._max_errors:
            logger.error(f"Too many errors ({self._error_count}), forcing memory cleanup")
            self._force_memory_cleanup()
            self._error_count = 0

    def _check_memory_usage(self):
        if not torch.cuda.is_available():
            return
        try:
            allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            if allocated_gb > 8.0:
                logger.warning(f"High GPU memory: {allocated_gb:.2f}GB")
                self._force_memory_cleanup()
        except Exception:
            pass

    def _force_memory_cleanup(self):
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
