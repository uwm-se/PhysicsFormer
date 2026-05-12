# Copyright (c) 2024
# Physics variety task for PhysicsFormer training
#
# WallCollision: High-velocity ball strikes massive stationary cube horizontally
# Physics concepts: momentum transfer, elastic/inelastic collision, mass ratio effects

import gc
import logging

import numpy as np
import torch
from omni.isaac.core.objects import DynamicSphere, DynamicCuboid, FixedCuboid
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch.maths import *
from omniisaacgymenvs.tasks.base.rl_task import RLTask

logger = logging.getLogger(__name__)


class WallCollisionTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self._initialization_failed = False
        self._error_count = 0
        self._max_errors = 10
        
        try:
            self.update_config(sim_config)
            self._num_observations = 26
            self._num_actions = 0
            RLTask.__init__(self, name, env)
        except Exception as e:
            logger.error(f"Failed to initialize WallCollisionTask: {e}")
            self._initialization_failed = True
            self._num_observations = 1
            self._num_actions = 0
            RLTask.__init__(self, name, env)

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._dt = self._task_cfg["sim"]["dt"]
        
        self._ball_radius = self._task_cfg["env"].get("ballRadius", 0.05)
        self._ball_mass = self._task_cfg["env"].get("ballMass", 1.0)
        self._ball_velocity = self._task_cfg["env"].get("ballVelocity", 20.0)
        self._cube_size = self._task_cfg["env"].get("cubeSize", 0.3)
        self._cube_mass = self._task_cfg["env"].get("cubeMass", 10000.0)
        self._ball_start_distance = self._task_cfg["env"].get("ballStartDistance", 2.0)
        self._max_episode_length = self._task_cfg["env"].get("maxEpisodeLength", 200)

    def set_up_scene(self, scene) -> None:
        if self._initialization_failed:
            super().set_up_scene(scene)
            return
            
        try:
            self._create_ground()
            self._create_massive_cube()
            self._create_projectile_ball()
            super().set_up_scene(scene, replicate_physics=False)
            
            self._cube = RigidPrimView(
                prim_paths_expr="/World/envs/.*/MassiveCube",
                name="cube_view",
                reset_xform_properties=False
            )
            scene.add(self._cube)
            
            self._ball = RigidPrimView(
                prim_paths_expr="/World/envs/.*/ProjectileBall",
                name="ball_view",
                reset_xform_properties=False
            )
            scene.add(self._ball)
        except Exception as e:
            logger.error(f"Failed to set up scene: {e}")
            self._initialization_failed = True
            super().set_up_scene(scene)

    def _create_ground(self):
        FixedCuboid(
            prim_path=self.default_zero_env_path + "/Ground",
            translation=torch.tensor([0, 0, -0.01]),
            scale=torch.tensor([5.0, 5.0, 0.02]),
            color=torch.tensor([0.3, 0.3, 0.3]),
        )

    def _create_massive_cube(self):
        cube = DynamicCuboid(
            prim_path=self.default_zero_env_path + "/MassiveCube",
            translation=torch.tensor([0, 0, self._cube_size / 2 + 0.001]),
            scale=torch.tensor([self._cube_size, self._cube_size, self._cube_size]),
            color=torch.tensor([0.2, 0.2, 0.2]),
            mass=self._cube_mass,
        )
        
        self._sim_config.apply_articulation_settings(
            "massive_cube", get_prim_at_path(cube.prim_path),
            self._sim_config.parse_actor_config("massive_cube")
        )

    def _create_projectile_ball(self):
        ball = DynamicSphere(
            prim_path=self.default_zero_env_path + "/ProjectileBall",
            translation=torch.tensor([-self._ball_start_distance, 0, self._cube_size / 2]),
            radius=self._ball_radius,
            color=torch.tensor([1.0, 0.2, 0.2]),
            mass=self._ball_mass,
        )
        
        self._sim_config.apply_articulation_settings(
            "projectile_ball", get_prim_at_path(ball.prim_path),
            self._sim_config.parse_actor_config("projectile_ball")
        )

    def initialize_views(self, scene):
        if self._initialization_failed:
            return
            
        try:
            super().initialize_views(scene)
            if scene.object_exists("cube_view"):
                scene.remove_object("cube_view", registry_only=True)
            if scene.object_exists("ball_view"):
                scene.remove_object("ball_view", registry_only=True)
                
            self._cube = RigidPrimView(
                prim_paths_expr="/World/envs/.*/MassiveCube",
                name="cube_view",
                reset_xform_properties=False
            )
            scene.add(self._cube)
            
            self._ball = RigidPrimView(
                prim_paths_expr="/World/envs/.*/ProjectileBall",
                name="ball_view",
                reset_xform_properties=False
            )
            scene.add(self._ball)
        except Exception as e:
            logger.error(f"Failed to initialize views: {e}")
            self._initialization_failed = True

    def get_observations(self) -> dict:
        if self._initialization_failed:
            self.obs_buf = torch.zeros((self._num_envs, self._num_observations), device=self._device)
            return {self._name: {"obs_buf": self.obs_buf}}
            
        try:
            cube_pos, cube_rot = self._cube.get_world_poses()
            cube_vel = self._cube.get_velocities()
            ball_pos, ball_rot = self._ball.get_world_poses()
            ball_vel = self._ball.get_velocities()
            
            if any(x is None for x in [cube_pos, cube_rot, cube_vel, ball_pos, ball_rot, ball_vel]):
                raise RuntimeError("Failed to get object state")
            
            self.obs_buf = torch.cat([
                cube_pos.view(self._num_envs, -1),
                cube_rot.view(self._num_envs, -1),
                cube_vel[:, :3].view(self._num_envs, -1),
                ball_pos.view(self._num_envs, -1),
                ball_rot.view(self._num_envs, -1),
                ball_vel[:, :3].view(self._num_envs, -1),
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
            
            cube_indices = self._cube.get_env_indices(env_ids)
            ball_indices = self._ball.get_env_indices(env_ids)
            
            cube_positions = torch.zeros((num_resets, 3), device=self._device)
            cube_positions[:, 0] = 0
            cube_positions[:, 1] = 0
            cube_positions[:, 2] = self._cube_size / 2 + 0.001
            
            cube_orientations = torch.zeros((num_resets, 4), device=self._device)
            cube_orientations[:, 3] = 1.0
            
            self._cube.set_world_poses(cube_positions, cube_orientations, indices=cube_indices)
            self._cube.set_velocities(
                torch.zeros((num_resets, 6), device=self._device),
                indices=cube_indices
            )
            
            ball_positions = torch.zeros((num_resets, 3), device=self._device)
            ball_positions[:, 0] = -self._ball_start_distance
            ball_positions[:, 1] = (torch.rand(num_resets, device=self._device) - 0.5) * 0.1
            ball_positions[:, 2] = self._cube_size / 2
            
            ball_orientations = torch.zeros((num_resets, 4), device=self._device)
            ball_orientations[:, 3] = 1.0
            
            self._ball.set_world_poses(ball_positions, ball_orientations, indices=ball_indices)
            
            ball_velocities = torch.zeros((num_resets, 6), device=self._device)
            velocity_variation = 0.8 + torch.rand(num_resets, device=self._device) * 0.4
            ball_velocities[:, 0] = self._ball_velocity * velocity_variation
            ball_velocities[:, 1] = (torch.rand(num_resets, device=self._device) - 0.5) * 0.5
            ball_velocities[:, 2] = (torch.rand(num_resets, device=self._device) - 0.5) * 0.3
            ball_velocities[:, 3] = (torch.rand(num_resets, device=self._device) - 0.5) * 2.0
            ball_velocities[:, 4] = (torch.rand(num_resets, device=self._device) - 0.5) * 2.0
            ball_velocities[:, 5] = (torch.rand(num_resets, device=self._device) - 0.5) * 2.0
            
            self._ball.set_velocities(ball_velocities, indices=ball_indices)
            
            self._check_memory_usage()
        except Exception as e:
            self._handle_error("reset_idx", e)
        
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        if self._initialization_failed:
            return
        try:
            ball_velocities = torch.zeros((self._num_envs, 6), device=self._device)
            ball_velocities[:, 0] = self._ball_velocity
            self._ball.set_velocities(ball_velocities)
        except Exception as e:
            self._handle_error("post_reset", e)

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
