# Copyright (c) 2024
# Physics variety task for PhysicsFormer training
#
# ContainerFillSpill: Objects fall into a container, pile up, and spill over edges
# Physics concepts: gravity, stacking, collision, overflow dynamics

import gc
import logging

import numpy as np
import torch
from omni.isaac.core.objects import DynamicSphere, FixedCuboid
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch.maths import *
from omniisaacgymenvs.tasks.base.rl_task import RLTask

logger = logging.getLogger(__name__)


class ContainerFillSpillTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self._initialization_failed = False
        self._error_count = 0
        self._max_errors = 10
        
        try:
            self.update_config(sim_config)
            self._num_observations = 24 * self._num_balls
            self._num_actions = 0
            RLTask.__init__(self, name, env)
        except Exception as e:
            logger.error(f"Failed to initialize ContainerFillSpillTask: {e}")
            self._initialization_failed = True
            self._num_observations = 1
            self._num_actions = 0
            self._num_balls = 1
            RLTask.__init__(self, name, env)

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._dt = self._task_cfg["sim"]["dt"]
        
        self._num_balls_base = self._task_cfg["env"].get("numBalls", 15)
        self._num_balls_min = max(8, self._num_balls_base - 5)
        self._num_balls_max = min(20, self._num_balls_base + 5)
        self._num_balls = self._num_balls_base
        self._container_size = self._task_cfg["env"].get("containerSize", [0.4, 0.4, 0.2])
        self._ball_radius = self._task_cfg["env"].get("ballRadius", 0.03)
        self._drop_height = self._task_cfg["env"].get("dropHeight", 0.8)
        self._max_episode_length = self._task_cfg["env"].get("maxEpisodeLength", 300)

    def set_up_scene(self, scene) -> None:
        if self._initialization_failed:
            super().set_up_scene(scene)
            return
            
        try:
            self._create_container()
            self._create_falling_balls()
            super().set_up_scene(scene, replicate_physics=False)
            
            self._container = RigidPrimView(
                prim_paths_expr="/World/envs/.*/Container/*",
                name="container_view",
                reset_xform_properties=False
            )
            scene.add(self._container)
            
            self._falling_objects = RigidPrimView(
                prim_paths_expr="/World/envs/.*/Balls/*",
                name="balls_view",
                reset_xform_properties=False
            )
            scene.add(self._falling_objects)
        except Exception as e:
            logger.error(f"Failed to set up scene: {e}")
            self._initialization_failed = True
            super().set_up_scene(scene)

    def _create_container(self):
        container_path = self.default_zero_env_path + "/Container"
        wall_thickness = 0.02
        sx, sy, sz = self._container_size
        
        FixedCuboid(
            prim_path=f"{container_path}/bottom",
            translation=torch.tensor([0, 0, 0]),
            scale=torch.tensor([sx, sy, wall_thickness]),
            color=torch.tensor([0.4, 0.4, 0.4]),
        )
        
        FixedCuboid(
            prim_path=f"{container_path}/wall_front",
            translation=torch.tensor([0, sy/2, sz/2]),
            scale=torch.tensor([sx, wall_thickness, sz]),
            color=torch.tensor([0.5, 0.5, 0.5]),
        )
        
        FixedCuboid(
            prim_path=f"{container_path}/wall_back",
            translation=torch.tensor([0, -sy/2, sz/2]),
            scale=torch.tensor([sx, wall_thickness, sz]),
            color=torch.tensor([0.5, 0.5, 0.5]),
        )
        
        FixedCuboid(
            prim_path=f"{container_path}/wall_left",
            translation=torch.tensor([-sx/2, 0, sz/2]),
            scale=torch.tensor([wall_thickness, sy, sz]),
            color=torch.tensor([0.5, 0.5, 0.5]),
        )
        
        FixedCuboid(
            prim_path=f"{container_path}/wall_right",
            translation=torch.tensor([sx/2, 0, sz/2]),
            scale=torch.tensor([wall_thickness, sy, sz]),
            color=torch.tensor([0.5, 0.5, 0.5]),
        )

    def _create_falling_balls(self):
        balls_path = self.default_zero_env_path + "/Balls"
        colors = [
            [0.9, 0.3, 0.3], [0.3, 0.9, 0.3], [0.3, 0.3, 0.9],
            [0.9, 0.9, 0.3], [0.9, 0.3, 0.9], [0.3, 0.9, 0.9],
        ]
        
        for i in range(self._num_balls):
            x_offset = (np.random.rand() - 0.5) * 0.2
            y_offset = (np.random.rand() - 0.5) * 0.2
            z_offset = self._drop_height + i * 0.1
            
            color = colors[i % len(colors)]
            
            ball = DynamicSphere(
                prim_path=f"{balls_path}/ball_{i}",
                translation=torch.tensor([x_offset, y_offset, z_offset]),
                radius=self._ball_radius,
                color=torch.tensor(color),
                mass=0.1,
            )
            self._sim_config.apply_articulation_settings(
                "ball", get_prim_at_path(ball.prim_path),
                self._sim_config.parse_actor_config("ball")
            )

    def initialize_views(self, scene):
        if self._initialization_failed:
            return
            
        try:
            super().initialize_views(scene)
            if scene.object_exists("container_view"):
                scene.remove_object("container_view", registry_only=True)
            if scene.object_exists("balls_view"):
                scene.remove_object("balls_view", registry_only=True)
                
            self._container = RigidPrimView(
                prim_paths_expr="/World/envs/.*/Container/*",
                name="container_view",
                reset_xform_properties=False
            )
            scene.add(self._container)
            
            self._falling_objects = RigidPrimView(
                prim_paths_expr="/World/envs/.*/Balls/*",
                name="balls_view",
                reset_xform_properties=False
            )
            scene.add(self._falling_objects)
        except Exception as e:
            logger.error(f"Failed to initialize views: {e}")
            self._initialization_failed = True

    def get_observations(self) -> dict:
        if self._initialization_failed:
            self.obs_buf = torch.zeros((self._num_envs, self._num_observations), device=self._device)
            return {self._name: {"obs_buf": self.obs_buf}}
            
        try:
            ball_pos, ball_rot = self._falling_objects.get_world_poses()
            ball_vel = self._falling_objects.get_velocities()
            
            if ball_pos is None or ball_rot is None or ball_vel is None:
                raise RuntimeError("Failed to get ball state")
            
            self.obs_buf = torch.cat([
                ball_pos.view(self._num_envs, -1),
                ball_rot.view(self._num_envs, -1),
                ball_vel.view(self._num_envs, -1),
            ], dim=-1)
        except Exception as e:
            self._handle_error("get_observations", e)
            self.obs_buf = torch.zeros((self._num_envs, self._num_observations), device=self._device)
        
        return {self._name: {"obs_buf": self.obs_buf}}

    def pre_physics_step(self, actions) -> None:
        pass

    def reset_idx(self, env_ids):
        if self._initialization_failed:
            self.reset_buf[env_ids] = 0
            self.progress_buf[env_ids] = 0
            return
            
        try:
            num_resets = len(env_ids)
            ball_indices = self._falling_objects.get_env_indices(env_ids)
            
            drop_height_variation = 0.5 + torch.rand(1, device=self._device).item() * 0.6
            spread_variation = 0.1 + torch.rand(1, device=self._device).item() * 0.2
            height_spacing = 0.05 + torch.rand(1, device=self._device).item() * 0.1
            
            new_positions = torch.zeros((num_resets * self._num_balls, 3), device=self._device)
            for i in range(self._num_balls):
                idx_start = i * num_resets
                idx_end = (i + 1) * num_resets
                new_positions[idx_start:idx_end, 0] = (torch.rand(num_resets, device=self._device) - 0.5) * spread_variation
                new_positions[idx_start:idx_end, 1] = (torch.rand(num_resets, device=self._device) - 0.5) * spread_variation
                new_positions[idx_start:idx_end, 2] = drop_height_variation + i * height_spacing
            
            new_orientations = torch.zeros((num_resets * self._num_balls, 4), device=self._device)
            new_orientations[:, 3] = 1.0
            
            initial_velocities = torch.zeros((num_resets * self._num_balls, 6), device=self._device)
            initial_velocities[:, 0] = (torch.rand(num_resets * self._num_balls, device=self._device) - 0.5) * 0.5
            initial_velocities[:, 1] = (torch.rand(num_resets * self._num_balls, device=self._device) - 0.5) * 0.5
            
            self._falling_objects.set_world_poses(new_positions, new_orientations, indices=ball_indices)
            self._falling_objects.set_velocities(initial_velocities, indices=ball_indices)
            
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
