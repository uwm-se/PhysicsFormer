# Copyright (c) 2026 Anonymous. All rights reserved.
# Author: Anonymous
#
# PROPRIETARY AND CONFIDENTIAL. This software is provided for academic review
# and research purposes only. Unauthorized copying, modification, distribution,
# or use of this software, via any medium, is strictly prohibited without prior
# written permission from Anonymous.
#
# Physics variety task base class for PhysicsFormer training
#
# Provides common error handling, memory management, and graceful failure support

import gc
import logging
import traceback
from typing import Optional

import numpy as np
import torch
from omni.isaac.core.prims import RigidPrimView
from omniisaacgymenvs.tasks.base.rl_task import RLTask

logger = logging.getLogger(__name__)


class PhysicsVarietyBaseTask(RLTask):
    """Base class for physics variety tasks with error handling and memory management."""

    def __init__(self, name, sim_config, env, offset=None) -> None:
        self._initialization_failed = False
        self._error_count = 0
        self._max_errors_before_reset = 10
        self._memory_warning_threshold_gb = 8.0
        
        try:
            self.update_config(sim_config)
            RLTask.__init__(self, name, env)
        except Exception as e:
            logger.error(f"Failed to initialize {name}: {e}")
            logger.error(traceback.format_exc())
            self._initialization_failed = True
            self._num_observations = 1
            self._num_actions = 0
            RLTask.__init__(self, name, env)

    def update_config(self, sim_config):
        """Override in subclass to parse task-specific config."""
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._dt = self._task_cfg["sim"]["dt"]
        self._max_episode_length = self._task_cfg["env"].get("maxEpisodeLength", 300)

    def set_up_scene(self, scene) -> None:
        """Set up scene with error handling."""
        if self._initialization_failed:
            logger.warning("Skipping scene setup due to initialization failure")
            super().set_up_scene(scene)
            return

        try:
            self._set_up_scene_impl(scene)
            super().set_up_scene(scene, replicate_physics=False)
            self._set_up_views(scene)
        except Exception as e:
            logger.error(f"Failed to set up scene: {e}")
            logger.error(traceback.format_exc())
            self._initialization_failed = True
            super().set_up_scene(scene)

    def _set_up_scene_impl(self, scene) -> None:
        """Override in subclass to create scene objects."""
        pass

    def _set_up_views(self, scene) -> None:
        """Override in subclass to set up RigidPrimViews."""
        pass

    def initialize_views(self, scene):
        """Initialize views with error handling."""
        try:
            super().initialize_views(scene)
            self._initialize_views_impl(scene)
        except Exception as e:
            logger.error(f"Failed to initialize views: {e}")
            logger.error(traceback.format_exc())

    def _initialize_views_impl(self, scene):
        """Override in subclass to initialize views."""
        pass

    def get_observations(self) -> dict:
        """Get observations with error handling."""
        if self._initialization_failed:
            self.obs_buf = torch.zeros(
                (self._num_envs, self._num_observations),
                device=self._device
            )
            return {self._name: {"obs_buf": self.obs_buf}}

        try:
            return self._get_observations_impl()
        except Exception as e:
            self._handle_error("get_observations", e)
            self.obs_buf = torch.zeros(
                (self._num_envs, self._num_observations),
                device=self._device
            )
            return {self._name: {"obs_buf": self.obs_buf}}

    def _get_observations_impl(self) -> dict:
        """Override in subclass to compute observations."""
        self.obs_buf = torch.zeros(
            (self._num_envs, self._num_observations),
            device=self._device
        )
        return {self._name: {"obs_buf": self.obs_buf}}

    def pre_physics_step(self, actions) -> None:
        """Pre-physics step with error handling."""
        if self._initialization_failed:
            return

        try:
            self._pre_physics_step_impl(actions)
        except Exception as e:
            self._handle_error("pre_physics_step", e)

    def _pre_physics_step_impl(self, actions) -> None:
        """Override in subclass to apply actions."""
        pass

    def reset_idx(self, env_ids):
        """Reset environments with error handling and memory cleanup."""
        if self._initialization_failed:
            self.reset_buf[env_ids] = 0
            self.progress_buf[env_ids] = 0
            return

        try:
            self._reset_idx_impl(env_ids)
            self._check_memory_usage()
        except Exception as e:
            self._handle_error("reset_idx", e)
            self.reset_buf[env_ids] = 0
            self.progress_buf[env_ids] = 0

    def _reset_idx_impl(self, env_ids):
        """Override in subclass to reset environments."""
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        """Post-reset with error handling."""
        if self._initialization_failed:
            return

        try:
            self._post_reset_impl()
        except Exception as e:
            self._handle_error("post_reset", e)

    def _post_reset_impl(self):
        """Override in subclass for post-reset logic."""
        pass

    def calculate_metrics(self) -> None:
        """Calculate metrics with error handling."""
        try:
            self._calculate_metrics_impl()
        except Exception as e:
            self._handle_error("calculate_metrics", e)
            self.rew_buf[:] = 0

    def _calculate_metrics_impl(self) -> None:
        """Override in subclass to compute rewards."""
        self.rew_buf[:] = 0

    def is_done(self) -> None:
        """Check done condition with error handling."""
        try:
            self._is_done_impl()
        except Exception as e:
            self._handle_error("is_done", e)
            self.reset_buf = torch.where(
                self.progress_buf >= self._max_episode_length - 1,
                torch.ones_like(self.reset_buf),
                self.reset_buf
            )

    def _is_done_impl(self) -> None:
        """Override in subclass to check termination."""
        self.reset_buf = torch.where(
            self.progress_buf >= self._max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf
        )

    def _handle_error(self, method_name: str, error: Exception):
        """Handle errors gracefully with logging and recovery."""
        self._error_count += 1
        logger.warning(f"Error in {method_name}: {error}")
        
        if self._error_count >= self._max_errors_before_reset:
            logger.error(f"Too many errors ({self._error_count}), forcing memory cleanup")
            self._force_memory_cleanup()
            self._error_count = 0

    def _check_memory_usage(self):
        """Check GPU memory usage and trigger cleanup if needed."""
        if not torch.cuda.is_available():
            return

        try:
            allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
            
            if allocated_gb > self._memory_warning_threshold_gb:
                logger.warning(
                    f"High GPU memory usage: {allocated_gb:.2f}GB allocated, "
                    f"{reserved_gb:.2f}GB reserved"
                )
                self._force_memory_cleanup()
        except Exception as e:
            logger.debug(f"Could not check memory usage: {e}")

    def _force_memory_cleanup(self):
        """Force garbage collection and clear GPU cache."""
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.info("Memory cleanup completed")
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")

    def cleanup(self):
        """Clean up resources when task is destroyed."""
        try:
            self._force_memory_cleanup()
            logger.info(f"Task {self._name} cleaned up successfully")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

    @staticmethod
    def safe_get_world_poses(view: RigidPrimView, indices=None):
        """Safely get world poses with error handling."""
        try:
            if indices is not None:
                return view.get_world_poses(indices=indices)
            return view.get_world_poses()
        except Exception as e:
            logger.warning(f"Failed to get world poses: {e}")
            return None, None

    @staticmethod
    def safe_get_velocities(view: RigidPrimView, indices=None):
        """Safely get velocities with error handling."""
        try:
            if indices is not None:
                return view.get_velocities(indices=indices)
            return view.get_velocities()
        except Exception as e:
            logger.warning(f"Failed to get velocities: {e}")
            return None

    @staticmethod
    def safe_set_world_poses(view: RigidPrimView, positions, orientations, indices=None):
        """Safely set world poses with error handling."""
        try:
            if indices is not None:
                view.set_world_poses(positions, orientations, indices=indices)
            else:
                view.set_world_poses(positions, orientations)
            return True
        except Exception as e:
            logger.warning(f"Failed to set world poses: {e}")
            return False

    @staticmethod
    def safe_set_velocities(view: RigidPrimView, velocities, indices=None):
        """Safely set velocities with error handling."""
        try:
            if indices is not None:
                view.set_velocities(velocities, indices=indices)
            else:
                view.set_velocities(velocities)
            return True
        except Exception as e:
            logger.warning(f"Failed to set velocities: {e}")
            return False
