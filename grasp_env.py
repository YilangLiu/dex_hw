import torch
import math
from typing import Literal
import genesis as gs
from genesis.utils.geom import xyz_to_quat, transform_quat_by_quat, transform_by_quat
from dexmate_urdf import robots

class GraspEnv:
    def __init__(
        self,
        num_envs,
        env_cfg,
        reward_cfg,
        robot_cfg,
        show_viewer=False,
    ):
        self.num_envs = num_envs
        self.num_obs = env_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.device = gs.device

        self.ctrl_dt = env_cfg["ctrl_dt"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.ctrl_dt)

        # configs
        self.env_cfg = env_cfg
        self.reward_scales = reward_cfg
        self.action_scales = torch.tensor(env_cfg["action_scales"], device=self.device)

        # == setup scene ==
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.ctrl_dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.ctrl_dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))), # here 
            rigid_options=gs.options.RigidOptions(
                dt=self.ctrl_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            show_viewer=show_viewer,
        )

        # == add ground ==
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # == add robot ==
        self.robot = Vega(
            num_envs=self.num_envs,
            scene=self.scene,
            args=robot_cfg,
            device=gs.device,
        )

        # == add object ==
        self.object = self.scene.add_entity(
            gs.morphs.Box(
                size=env_cfg["box_size"],
                fixed=env_cfg["box_fixed"],
                collision=env_cfg["box_collision"],
            ),
            # material=gs.materials.Rigid(gravity_compensation=1),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(1.0, 0.0, 0.0),
                ),
            ),
        )

        # == add a table ==
        self.table = self.scene.add_entity(
            gs.morphs.Box(
                size=[0.2, 0.2, 0.01],
                fixed=True,
                collision=True,
            ),
            # material=gs.materials.Rigid(gravity_compensation=1),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(1.0, 1.0, 1.0),
                ),
            ),
        )

        # == add camera ==
        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(1280, 720),
                pos=(1.5, 0.0, 0.2),
                lookat=(0, 0, 0.2),
                fov=50,
                GUI=False,
            )

        # build
        self.scene.build(n_envs=num_envs)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.ctrl_dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        self.keypoints_offset = self.get_keypoint_offsets(batch_size=self.num_envs, device=self.device, unit_length=0.03)
        # == init buffers ==
        self._init_buffers()

    def _init_buffers(self):
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=gs.device)
        self.goal_pose = torch.zeros(self.num_envs, 7, device=gs.device)
        self.extras = dict()
        self.extras["observations"] = dict()

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        self.episode_length_buf[envs_idx] = 0

        # reset robot
        self.robot.reset(envs_idx)

        # reset object
        num_reset = len(envs_idx)
        random_x = torch.rand(num_reset, device=self.device) * 0.4 + 0.3  # 0.3 ~ 0.6
        random_y = torch.rand(num_reset, device=self.device) * 0.1 - 0.2   # -0.3 ~ -0.2
        random_z = torch.rand(num_reset, device=self.device) * 0.0 + 0.1   # 0.1 ~ 0.4
        
        random_pos = torch.stack([random_x, random_y, random_z], dim=-1)

        # randomly yaw the object
        q_downward = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).repeat(num_reset, 1)
        PI = 3.1415926
        random_yaw = (torch.rand(num_reset, device=self.device) * 2 * PI - PI) * 0.25
        q_yaw = torch.stack(
            [
                torch.cos(random_yaw / 2),
                torch.zeros(num_reset, device=self.device),
                torch.zeros(num_reset, device=self.device),
                torch.sin(random_yaw / 2),
            ],
            dim=-1,
        )
        goal_yaw = transform_quat_by_quat(q_yaw, q_downward)

        self.goal_pose[envs_idx] = torch.cat([random_pos, goal_yaw], dim=-1)
        self.object.set_pos(random_pos, envs_idx=envs_idx)
        self.object.set_quat(goal_yaw, envs_idx=envs_idx)
        
        # reset table 
        table_quat = torch.tensor(
                    [1.0, 0.0, 0.0, 0.0],
                    device=self.device,
                    dtype=gs.tc_float,
                ).repeat(num_reset, 1)
        
        table_pos = torch.stack([random_x, random_y, random_z - self.env_cfg["box_size"][0]/2], dim=-1)
        self.table.set_pos(table_pos, envs_idx=envs_idx)
        self.table.set_quat(table_quat, envs_idx=envs_idx)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))

        obs, self.extras = self.get_observations()
        return obs, None

    def step(self, actions):
        # update time
        self.episode_length_buf += 1

        # apply action based on task
        actions = self.rescale_action(actions)

        self.robot.apply_action(actions, open_gripper=True)
        self.scene.step()

        # check termination
        env_reset_idx = self.is_episode_complete()
        if len(env_reset_idx) > 0:
            self.reset_idx(env_reset_idx)

        # compute reward based on task
        reward = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew

        # get observations and fill extras
        obs, self.extras = self.get_observations()
        return obs, reward, self.reset_buf, self.extras

    def is_episode_complete(self):
        time_out_buf = self.episode_length_buf > self.max_episode_length

        # check if the ee is in the valid position
        self.reset_buf = time_out_buf

        # fill time out buffer for reward/value bootstrapping
        time_out_idx = (time_out_buf).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0
        return self.reset_buf.nonzero(as_tuple=True)[0]

    def get_observations(self):
        palm_pos, palm_quat = (
            self.robot.ee_pose[:, :3],
            self.robot.ee_pose[:, 3:7],
        )
        
        obj_pos, obj_quat = self.object.get_pos(), self.object.get_quat()
        obs_components = [
            palm_pos - obj_pos,  # 3D position difference
            palm_quat,  # current orientation (4D quaternion)
            obj_pos,  # goal pose (7D: pos + quat)
            obj_quat,  # goal pose (7D: pos + quat)
        ]
        obs_tensor = torch.cat(obs_components, dim=-1)
        self.extras["observations"]["critic"] = obs_tensor
        return obs_tensor, self.extras

    def rescale_action(self, action: torch.Tensor) -> torch.Tensor:
        rescaled_action = action * self.action_scales
        return rescaled_action

    # ------------ begin reward functions----------------
    def _reward_keypoints(self):
        keypoints_offset = self.keypoints_offset
        # there is a offset between the finger tip and the finger base frame
        finger_tip_z_offset = torch.tensor(
            [0.0, 0.0, -0.06],
            device=self.device,
            dtype=gs.tc_float,
        ).repeat(self.num_envs, 1)

        finger_pos_keypoints = self._to_world_frame(
            self.robot.ee_pose[:, :3] + finger_tip_z_offset,
            self.robot.ee_pose[:, 3:7],
            keypoints_offset,
        )
        object_pos_keypoints = self._to_world_frame(self.object.get_pos(), self.object.get_quat(), keypoints_offset)
        dist = torch.norm(finger_pos_keypoints - object_pos_keypoints, p=2, dim=-1).sum(-1)
        self.finger_pos_keypoints = finger_pos_keypoints
        self.object_pos_keypoints = object_pos_keypoints
        return torch.exp(-dist)

    def _reward_table_contact(self):
        # Get current gripper DOF positions
        hand_dofs = self.robot._robot_entity.get_qpos()[:, self.robot.right_hand_dofs_idx]
        expected_open_pos = self.robot.right_hand_open.repeat(self.num_envs, 1)
        dof_error = torch.norm(hand_dofs - expected_open_pos, dim=-1)

        # Apply penalty when error is above threshold (indicating contact)
        contact_threshold = 0.01  # 1cm tolerance
        contact_penalty = torch.where(dof_error > contact_threshold, -dof_error, torch.zeros_like(dof_error))

        return contact_penalty

    # ------------ end reward functions----------------

    def _to_world_frame(
        self,
        position: torch.Tensor,  # [B, 3]
        quaternion: torch.Tensor,  # [B, 4]
        keypoints_offset: torch.Tensor,  # [B, 7, 3]
    ) -> torch.Tensor:
        world = torch.zeros_like(keypoints_offset)
        for k in range(keypoints_offset.shape[1]):
            world[:, k] = position + transform_by_quat(keypoints_offset[:, k], quaternion)
        return world

    @staticmethod
    def get_keypoint_offsets(batch_size, device, unit_length=0.5):
        """
        Get uniformly-spaced keypoints along a line of unit length, centered at body center.
        """
        keypoint_offsets = (
            torch.tensor(
                [
                    [0, 0, 0],  # origin
                    [-1.0, 0, 0],  # x-negative
                    [1.0, 0, 0],  # x-positive
                    [0, -1.0, 0],  # y-negative
                    [0, 1.0, 0],  # y-positive
                    [0, 0, -1.0],  # z-negative
                    [0, 0, 1.0],  # z-positive
                ],
                device=device,
                dtype=torch.float32,
            )
            * unit_length
        )
        return keypoint_offsets.unsqueeze(0).repeat(batch_size, 1, 1)

    def grasp_and_lift_demo(self, render: bool = False):
        total_steps = 200
        goal_pose = self.robot.ee_link_pose.clone()

        down_pose = goal_pose.clone()
        down_pose[:, 1] -= 0.01
        down_pose[:, 2] -= 0.06

        lift_height = 0.1
        lift_pose = goal_pose.clone()
        lift_pose[:, 2] += lift_height

        final_pose = goal_pose.clone()
        final_pose[:, 0] = 0.3
        final_pose[:, 1] = 0.0
        final_pose[:, 2] = 0.4
        reset_pose = torch.tensor([0.2, 0.0, 0.4, 0.0, 1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        for i in range(total_steps):
            if render:
                self.cam.render()
            if i < total_steps / 4:
                self.robot.go_to_goal(down_pose, open_gripper=True)
            elif i < total_steps / 4 * 2:  # grasping
                self.robot.go_to_goal(down_pose, open_gripper=False)
            else:  # lifting
                self.robot.go_to_goal(lift_pose, open_gripper=False)
            self.scene.step()


## ------------ robot ----------------
class Vega:
    def __init__(self, num_envs: int, scene: gs.Scene, args: dict, device: str = "cpu"):
        # == set members ==
        self._device = device
        self._scene = scene
        self._num_envs = num_envs
        self._args = args
        self._asset_path = robots.humanoid.vega_1.vega_upper_body.urdf
        # == Genesis configurations ==
        material: gs.materials.Rigid = gs.materials.Rigid()
        morph: gs.morphs.URDF = gs.morphs.URDF(
            file= self._asset_path,
            pos=(0.0, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
            fixed=True
        )
        self._robot_entity: gs.Entity = scene.add_entity(material=material, morph=morph)

        self._ik_method: Literal["rel_pose", "dls"] = args["ik_method"]

        # == some buffer initialization ==
        self._init()

    def _init(self):
        self.head_dofs_idx = [self._robot_entity.get_joint(name).dof_idx_local 
                         for name in self._args["head_names"]]
        self.left_arm_dofs_idx = [self._robot_entity.get_joint(name).dof_idx_local 
                             for name in self._args["left_arm_names"]]
        self.right_arm_dofs_idx = [self._robot_entity.get_joint(name).dof_idx_local 
                              for name in self._args["right_arm_names"]]
        self.left_hand_dofs_idx = [self._robot_entity.get_joint(name).dof_idx_local 
                              for name in self._args["left_hand_names"]]
        self.right_hand_dofs_idx = [self._robot_entity.get_joint(name).dof_idx_local 
                               for name in self._args["right_hand_names"]]

        
        self.head_default_dof = torch.tensor(self._args["head_default_dof"], dtype=torch.float32, device=self._device)
        self.left_arm_default_dof = torch.tensor(self._args["left_arm_default_dof"], dtype=torch.float32, device=self._device)
        self.left_hand_default_dof = torch.tensor(self._args["left_hand_default_dof"], dtype=torch.float32, device=self._device)
        self.right_arm_default_dof = torch.tensor(self._args["right_arm_default_dof"],  dtype=torch.float32, device=self._device)
        self.right_hand_default_dof = torch.tensor(self._args["right_hand_default_dof"], dtype=torch.float32, device=self._device)
        
        self.right_hand_open = torch.tensor(self._args["right_hand_open"], dtype=torch.float32, device=self._device)
        self.right_hand_close = torch.tensor(self._args["right_hand_close"], dtype=torch.float32, device=self._device)

        self._ee_link = self._robot_entity.get_link(self._args["ee_link_name"])
        self._ee_frame_offset = torch.tensor([0.2006,  0.0186, -0.0266], device=self._device, dtype=torch.float32).repeat(self._num_envs, 1)
        
        self._default_joint_angles =  self._args["head_default_dof"]+ self._args["left_arm_default_dof"] + \
                            self._args["left_hand_default_dof"] + self._args["right_arm_default_dof"] + \
                            self._args["right_hand_open"]
        
        self._default_joint_idxs = self.head_dofs_idx + self.left_arm_dofs_idx + self.left_hand_dofs_idx + \
                            self.right_arm_dofs_idx + self.right_hand_dofs_idx

    def reset(self, envs_idx: torch.IntTensor):
        if len(envs_idx) == 0:
            return
        self.reset_home(envs_idx)

    def reset_home(self, envs_idx: torch.IntTensor | None = None):
        if envs_idx is None:
            envs_idx = torch.arange(self._num_envs, device=self._device)
        default_joint_angles = torch.tensor(
            self._default_joint_angles, dtype=torch.float32, device=self._device
        ).repeat(len(envs_idx), 1)

        self._robot_entity.set_dofs_position(position=default_joint_angles,
                                            dofs_idx_local=self._default_joint_idxs, 
                                            envs_idx=envs_idx)
        
        self._robot_entity.zero_all_dofs_velocity(envs_idx)
        

    def apply_action(self, action: torch.Tensor, open_gripper: bool) -> None:
        """
        Apply the action to the robot.
        """
        q_pos = self._robot_entity.get_qpos()
        if self._ik_method == "gs_ik":
            q_pos = self._gs_ik(action)
        elif self._ik_method == "dls_ik": # TODO: add later
            q_pos = self._dls_ik(action)
        else:
            raise ValueError(f"Invalid control mode: {self._ik_method}")
        # set gripper to open
        if open_gripper:
            q_pos[:, self.right_hand_dofs_idx] = self.right_hand_open
        else:
            q_pos[:, self.right_hand_dofs_idx] = self.right_hand_close

        # set head, left arm and left hand to default 
        q_pos[:, self.head_dofs_idx] = self.head_default_dof
        q_pos[:, self.left_arm_dofs_idx] = self.left_arm_default_dof
        q_pos[:, self.left_hand_dofs_idx] = self.left_hand_default_dof

        self._robot_entity.control_dofs_position(position=q_pos)

    def _gs_ik(self, action: torch.Tensor) -> torch.Tensor:
        """
        Genesis inverse kinematics
        """
        delta_position = action[:, :3]
        delta_orientation = action[:, 3:6]

        # compute target pose
        
        target_position = delta_position + self._ee_link.get_pos()
        quat_rel = xyz_to_quat(delta_orientation, rpy=True, degrees=False)
        target_orientation = transform_quat_by_quat(quat_rel, self._ee_link.get_quat())
        q_pos = self._robot_entity.inverse_kinematics(
            link=self._ee_link,
            pos=target_position,
            quat=target_orientation,
            dofs_idx_local=self.right_arm_dofs_idx,
        )
        return q_pos

    def _dls_ik(self, action: torch.Tensor) -> torch.Tensor:
        """
        Damped least squares inverse kinematics
        """
        delta_pose = action[:, :6]
        lambda_val = 0.01
        jacobian = self._robot_entity.get_jacobian(link=self._ee_link)
        jacobian_T = jacobian.transpose(1, 2)
        lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[1], device=self._device)
        delta_joint_pos = (
            jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
        ).squeeze(-1)
        return self._robot_entity.get_qpos() + delta_joint_pos

    def go_to_goal(self, goal_pose: torch.Tensor, open_gripper: bool = True):
        q_pos = self._robot_entity.inverse_kinematics(
            link=self._ee_link,
            pos=goal_pose[:, :3],
            quat=goal_pose[:, 3:7],
            dofs_idx_local=self.right_arm_dofs_idx,
        )
        if open_gripper:
            q_pos[:, self.right_hand_dofs_idx] = self.right_hand_open
        else:
            q_pos[:, self.right_hand_dofs_idx] = self.right_hand_close

        # set head, left arm and left hand to default 
        q_pos[:, self.head_dofs_idx] = self.head_default_dof
        q_pos[:, self.left_arm_dofs_idx] = self.left_arm_default_dof
        q_pos[:, self.left_hand_dofs_idx] = self.left_hand_default_dof
        self._robot_entity.control_dofs_position(position=q_pos)

    @property
    def base_pos(self):
        return self._robot_entity.get_pos()

    @property
    def ee_pose(self) -> torch.Tensor:
        """
        The end-effector pose (the hand pose)
        """
        pos, quat = self._ee_link.get_pos(), self._ee_link.get_quat()
        ee_pos = pos + transform_by_quat(self._ee_frame_offset, quat)
        # ee_pos = pos + self._ee_frame_offset 
        return torch.cat([ee_pos, quat], dim=-1)
    
    @property
    def ee_link_pose(self) -> torch.Tensor:
        pos, quat = self._ee_link.get_pos(), self._ee_link.get_quat()
        return torch.cat([pos, quat], dim=-1)