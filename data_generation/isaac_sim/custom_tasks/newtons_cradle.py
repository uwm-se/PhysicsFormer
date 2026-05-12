# Copyright (c) 2024
# Physics variety task for PhysicsFormer training
#
# NewtonsCradle: Classic momentum and energy conservation demonstration
# Physics concepts: elastic collision, momentum conservation, energy transfer

import numpy as np
import torch
from omni.isaac.core.objects import DynamicSphere, FixedCuboid
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.maths import *
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from pxr import UsdPhysics, Gf


class NewtonsCradleTask(RLTask):
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
        
        self._num_balls = self._task_cfg["env"].get("numBalls", 5)
        self._ball_radius = self._task_cfg["env"].get("ballRadius", 0.05)
        self._string_length = self._task_cfg["env"].get("stringLength", 0.3)
        self._max_episode_length = self._task_cfg["env"].get("maxEpisodeLength", 400)

    def set_up_scene(self, scene) -> None:
        self._create_frame()
        self._create_pendulum_balls()
        super().set_up_scene(scene, replicate_physics=False)
        self._setup_joints()
        
        self._frame = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Frame/*",
            name="frame_view",
            reset_xform_properties=False
        )
        scene.add(self._frame)
        
        self._cradle_balls = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Balls/*",
            name="cradle_balls_view",
            reset_xform_properties=False
        )
        scene.add(self._cradle_balls)
        return

    def _create_frame(self):
        frame_path = self.default_zero_env_path + "/Frame"
        frame_height = self._string_length + 0.1
        frame_width = (self._num_balls + 1) * self._ball_radius * 2
        
        FixedCuboid(
            prim_path=f"{frame_path}/top_bar",
            translation=torch.tensor([0, 0, frame_height]),
            scale=torch.tensor([frame_width, 0.02, 0.02]),
            color=torch.tensor([0.1, 0.1, 0.1]),
        )
        
        FixedCuboid(
            prim_path=f"{frame_path}/left_leg",
            translation=torch.tensor([-frame_width/2, 0, frame_height/2]),
            scale=torch.tensor([0.02, 0.02, frame_height]),
            color=torch.tensor([0.1, 0.1, 0.1]),
        )
        
        FixedCuboid(
            prim_path=f"{frame_path}/right_leg",
            translation=torch.tensor([frame_width/2, 0, frame_height/2]),
            scale=torch.tensor([0.02, 0.02, frame_height]),
            color=torch.tensor([0.1, 0.1, 0.1]),
        )

    def _create_pendulum_balls(self):
        balls_path = self.default_zero_env_path + "/Balls"
        
        start_x = -(self._num_balls - 1) * self._ball_radius
        
        for i in range(self._num_balls):
            x_pos = start_x + i * self._ball_radius * 2
            z_pos = self._string_length + 0.1 - self._string_length
            
            ball = DynamicSphere(
                prim_path=f"{balls_path}/ball_{i}",
                translation=torch.tensor([x_pos, 0, z_pos]),
                radius=self._ball_radius,
                color=torch.tensor([0.8, 0.8, 0.8]),
                mass=1.0,
            )
            
            self._sim_config.apply_articulation_settings(
                "cradle_ball", get_prim_at_path(ball.prim_path),
                self._sim_config.parse_actor_config("cradle_ball")
            )

    def _setup_joints(self):
        stage = get_current_stage()
        frame_height = self._string_length + 0.1
        start_x = -(self._num_balls - 1) * self._ball_radius
        
        for env_idx in range(self._num_envs):
            env_path = f"{self.default_base_env_path}/env_{env_idx}"
            
            for ball_idx in range(self._num_balls):
                ball_path = f"{env_path}/Balls/ball_{ball_idx}"
                joint_path = f"{ball_path}_joint"
                
                x_pos = start_x + ball_idx * self._ball_radius * 2
                anchor_pos = Gf.Vec3f(x_pos, 0, frame_height)
                
                joint = UsdPhysics.SphericalJoint.Define(stage, joint_path)
                joint.CreateBody0Rel().SetTargets([f"{env_path}/Frame/top_bar"])
                joint.CreateBody1Rel().SetTargets([ball_path])
                joint.CreateLocalPos0Attr().Set(Gf.Vec3f(x_pos, 0, 0))
                joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, self._string_length))

    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists("frame_view"):
            scene.remove_object("frame_view", registry_only=True)
        if scene.object_exists("cradle_balls_view"):
            scene.remove_object("cradle_balls_view", registry_only=True)
            
        self._frame = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Frame/*",
            name="frame_view",
            reset_xform_properties=False
        )
        scene.add(self._frame)
        
        self._cradle_balls = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Balls/*",
            name="cradle_balls_view",
            reset_xform_properties=False
        )
        scene.add(self._cradle_balls)

    def get_observations(self) -> dict:
        ball_pos, ball_rot = self._cradle_balls.get_world_poses()
        ball_vel = self._cradle_balls.get_velocities()
        
        self.obs_buf = torch.cat([
            ball_pos.view(self._num_envs, -1),
            ball_rot.view(self._num_envs, -1),
            ball_vel[:, :3].view(self._num_envs, -1),
        ], dim=-1)
        
        observations = {self._name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions) -> None:
        if self.progress_buf[0] == 1:
            num_balls_to_swing = 1 + int(torch.rand(1, device=self._device).item() * 2)
            
            swing_indices = []
            for b in range(num_balls_to_swing):
                ball_idx = torch.arange(
                    b, self._num_envs * self._num_balls, self._num_balls,
                    device=self._device
                )
                swing_indices.append(ball_idx)
            swing_indices = torch.cat(swing_indices)
            
            swing_velocity = torch.zeros((len(swing_indices), 6), device=self._device)
            swing_speed = -2.5 - torch.rand(1, device=self._device).item() * 2.0
            swing_velocity[:, 0] = swing_speed + (torch.rand(len(swing_indices), device=self._device) - 0.5) * 0.2
            
            self._cradle_balls.set_velocities(swing_velocity, indices=swing_indices)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        
        ball_indices = self._cradle_balls.get_env_indices(env_ids)
        
        spacing_variation = 1.9 + torch.rand(1, device=self._device).item() * 0.2
        start_x = -(self._num_balls - 1) * self._ball_radius * spacing_variation / 2
        z_pos = self._string_length + 0.1 - self._string_length
        
        new_positions = torch.zeros((num_resets * self._num_balls, 3), device=self._device)
        new_orientations = torch.zeros((num_resets * self._num_balls, 4), device=self._device)
        new_orientations[:, 3] = 1.0
        
        for i in range(self._num_balls):
            idx_start = i * num_resets
            idx_end = (i + 1) * num_resets
            pos_noise = (torch.rand(num_resets, device=self._device) - 0.5) * 0.002
            new_positions[idx_start:idx_end, 0] = start_x + i * self._ball_radius * spacing_variation + pos_noise
            new_positions[idx_start:idx_end, 1] = (torch.rand(num_resets, device=self._device) - 0.5) * 0.002
            new_positions[idx_start:idx_end, 2] = z_pos
        
        self._cradle_balls.set_world_poses(new_positions, new_orientations, indices=ball_indices)
        self._cradle_balls.set_velocities(
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
