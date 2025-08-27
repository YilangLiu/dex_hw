import os
import argparse
from typing import List, Tuple, Optional

import numpy as np
import torch
import genesis as gs

from dexmate_urdf import robots


class VegaManipulationDemo:
    """A demonstration of robot manipulation using the Vega humanoid robot.
    
    This class handles the setup and execution of a robotic manipulation task
    where the robot reaches for and grasps a cube using inverse kinematics
    and path planning.
    
    Attributes:
        device: The device to run computations on (CUDA).
        scene: The Genesis simulation scene.
        robot: The Vega humanoid robot entity.
        object: The target cube object.
        camera: The scene camera for recording.
        args: Command line arguments.
    """
    
    def __init__(self, args: argparse.Namespace):
        """Initialize the Vega manipulation demo.
        
        Args:
            args: Parsed command line arguments containing visualization,
                  debug, and recording options.
        """
        self.args = args
        self.device = "cuda"
        self.scene = None
        self.robot = None
        self.object = None
        self.camera = None
        
        # Robot joint configurations
        self._setup_joint_configurations()
        
    def _setup_joint_configurations(self):
        """Set up robot joint names and default configurations."""
        # Joint group names
        self.head_names = ["head_j1", "head_j2", "head_j3"]
        self.left_arm_names = [
            "L_arm_j1", "L_arm_j2", "L_arm_j3", "L_arm_j4",
            "L_arm_j5", "L_arm_j6", "L_arm_j7"
        ]
        self.right_arm_names = [
            "R_arm_j1", "R_arm_j2", "R_arm_j3", "R_arm_j4",
            "R_arm_j5", "R_arm_j6", "R_arm_j7"
        ]
        self.left_hand_names = [
            "L_th_j0", "L_th_j1", "L_th_j2", "L_ff_j1", "L_ff_j2",
            "L_mf_j1", "L_mf_j2", "L_rf_j1", "L_rf_j2", "L_lf_j1", "L_lf_j2"
        ]
        self.right_hand_names = [
            "R_th_j0", "R_th_j1", "R_th_j2", "R_ff_j1", "R_ff_j2",
            "R_mf_j1", "R_mf_j2", "R_rf_j1", "R_rf_j2", "R_lf_j1", "R_lf_j2"
        ]
        
        # Default joint positions
        self.head_default = torch.tensor([0.0, 0.0, 0.0], 
                                       device=self.device, dtype=torch.float32)
        self.left_arm_default = torch.tensor(
            [1.5, 0.0, 0.0, -1.756, -1.271, 0.0, 0.0],
            device=self.device, dtype=torch.float32
        )
        self.right_arm_default = torch.tensor(
            [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            device=self.device, dtype=torch.float32
        )
        self.left_hand_default = torch.tensor(
            [1.6, -0.2668, 0.0, -1.0946, 0.0, -1.0844, 0.0,
             -1.0154, 0.0, -1.0118, 0.0],
            device=self.device, dtype=torch.float32
        )
        self.right_hand_default = torch.tensor(
            [1.6, -0.2668, 0.0, -1.0946, 0.0, -1.0844, 0.0,
             -1.0154, 0.0, -1.0118, 0.0],
            device=self.device, dtype=torch.float32
        )
        
        # Hand grasping configurations
        self.right_hand_close = torch.tensor(
            [1.5, 0.0, -0.8, -1.0946, -0.5, -1.0844, -0.5,
             -1.0154, -0.5, -1.0118, -0.5],
            device=self.device, dtype=torch.float32
        )
        self.left_hand_close = torch.tensor(
            [0.4, -0.2668, -0.4, -1.0946, 0.0, -1.0844, 0.0,
             -1.0154, 0.0, -1.0118, 0.0],
            device=self.device, dtype=torch.float32
        )
        self.right_hand_open = torch.tensor(
            [0.0, 0.0, 0.0, -0.5, 0.0, -0.5, 0.0, -0.5, 0.0, -0.5, 0.0],
            device=self.device, dtype=torch.float32
        )
        
        # Joint limits for right hand
        self.right_hand_low = [
            -0.0158, -0.3468, -0.4298, -1.0946, -1.2101, -1.0844, -1.2026,
            -1.0154, -1.1156, -1.0118, -1.1073
        ]
        self.right_hand_high = [
            1.605, 0.1834, 0.2731, 0.2891, 0.3681, 0.2801, 0.3533,
            0.2840, 0.3599, 0.2811, 0.4014
        ]
    
    def initialize_genesis(self):
        """Initialize the Genesis physics engine."""
        gs.init(seed=0, precision="32", backend=gs.gpu)
    
    def create_scene(self):
        """Create and configure the simulation scene."""
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0, -3.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(dt=0.01),
            show_viewer=self.args.vis,
            rigid_options=gs.options.RigidOptions(
                dt=0.01,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                enable_self_collision=False
            ),
        )
    
    def add_camera(self):
        """Add a camera to the scene for recording."""
        self.camera = self.scene.add_camera(
            res=(1280, 960),
            pos=(3.5, 0.0, 2.5),
            lookat=(0, 0, 0.5),
            fov=30,
            GUI=False,
        )
    
    def add_ground(self):
        """Add a ground plane to the scene."""
        self.scene.add_entity(
            gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True)
        )
    
    def add_robot(self):
        """Add the Vega humanoid robot to the scene."""
        vega_urdf_path = robots.humanoid.vega_1.vega_upper_body.urdf
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=os.path.join(vega_urdf_path),
                pos=np.array([0, 0, 0.0]),
                quat=np.array([1, 0, 0, 0]),
                fixed=True
            ),
        )
    
    def add_target_object(self):
        """Add a red cube as the manipulation target."""
        self.object = self.scene.add_entity(
            gs.morphs.Box(
                size=[0.06, 0.06, 0.06],
                fixed=True,
                collision=True,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(1.0, 0.0, 0.0),
                ),
            ),
        )
    
    def place_object_randomly(self):
        """Place the target object at a random position."""
        random_x = torch.rand(1, device=self.device) * 0.3 + 0.3  # 0.3 ~ 0.6
        random_y = torch.rand(1, device=self.device) * 0.1 - 0.3   # -0.3 ~ -0.2
        random_z = torch.rand(1, device=self.device) * 0.3 + 0.1   # 0.1 ~ 0.4
        
        random_pos = torch.tensor([random_x, random_y, random_z])
        random_quat = torch.tensor([1, 0, 0, 0], device=self.device)
        
        self.object.set_pos(random_pos)
        self.object.set_quat(random_quat)
        
        return random_pos
    
    def get_joint_indices(self) -> Tuple[List[int], ...]:
        """Get the local DOF indices for each joint group.
        
        Returns:
            A tuple containing lists of joint indices for head, left arm,
            right arm, left hand, and right hand.
        """
        head_dofs_idx = [self.robot.get_joint(name).dof_idx_local 
                         for name in self.head_names]
        left_arm_dofs_idx = [self.robot.get_joint(name).dof_idx_local 
                             for name in self.left_arm_names]
        right_arm_dofs_idx = [self.robot.get_joint(name).dof_idx_local 
                              for name in self.right_arm_names]
        left_hand_dofs_idx = [self.robot.get_joint(name).dof_idx_local 
                              for name in self.left_hand_names]
        right_hand_dofs_idx = [self.robot.get_joint(name).dof_idx_local 
                               for name in self.right_hand_names]
        
        return (head_dofs_idx, left_arm_dofs_idx, right_arm_dofs_idx,
                left_hand_dofs_idx, right_hand_dofs_idx)
    
    def plan_manipulation_path(self, object_pos: torch.Tensor) -> np.ndarray:
        """Plan a path for the robot to reach the target object.
        
        Args:
            object_pos: The position of the target object.
            
        Returns:
            A path of joint configurations to reach the target.
        """
        end_effector = self.robot.get_link('R_arm_l7')
        target_offset = np.array([-0.2, -0.05, 0.06])
        target_pos = object_pos.cpu().numpy() + target_offset
        
        # Calculate inverse kinematics solution
        qpos_goal = self.robot.inverse_kinematics(
            link=end_effector,
            pos=target_pos,
            quat=np.array([1, 0, 0, 0])
        )
        
        # Plan path to the goal
        path = self.robot.plan_path(qpos_goal=qpos_goal, num_waypoints=400)
        
        return path
    
    def execute_reaching_motion(self, path: torch.Tensor, 
                               joint_indices: Tuple[List[int], ...]):
        """Execute the reaching motion along the planned path.
        
        Args:
            path: The planned joint configuration path.
            joint_indices: Tuple of joint indices for different body parts.
        """
        (head_dofs_idx, left_arm_dofs_idx, right_arm_dofs_idx,
         left_hand_dofs_idx, right_hand_dofs_idx) = joint_indices
        
        for waypoint in path:
            # Set joint positions for each body part
            waypoint[head_dofs_idx] = self.head_default
            waypoint[left_arm_dofs_idx] = self.left_arm_default
            waypoint[left_hand_dofs_idx] = self.left_hand_default
            waypoint[right_hand_dofs_idx] = self.right_hand_open
            
            # Control robot to the waypoint
            self.robot.control_dofs_position(position=waypoint)
            self.scene.step()
            
            if self.args.record:
                self.camera.render()
    
    def execute_grasping_motion(self, final_waypoint: torch.Tensor,
                               right_hand_dofs_idx: List[int]):
        """Execute the grasping motion to close the hand around the object.
        
        Args:
            final_waypoint: The final waypoint from the reaching path.
            right_hand_dofs_idx: Indices of the right hand joints.
        """
        # Close the right hand
        grasp_pose = final_waypoint
        grasp_pose[right_hand_dofs_idx] = self.right_hand_close
        self.robot.control_dofs_position(grasp_pose)
        
        # Simulate the grasping motion
        for _ in range(100):
            self.scene.step()
            if self.args.record:
                self.camera.render()
    
    def run_demo(self):
        """Run the complete manipulation demonstration."""
        # Initialize and setup
        self.initialize_genesis()
        self.create_scene()
        self.add_camera()
        self.add_ground()
        self.add_robot()
        self.add_target_object()
        
        # Build the scene
        self.scene.build()
        
        if self.args.record:
            self.camera.start_recording()
        
        # Place object and plan manipulation
        object_pos = self.place_object_randomly()
        path = self.plan_manipulation_path(object_pos)
        
        # Debug visualization if requested
        path_debug = None
        if self.args.debug:
            path_debug = self.scene.draw_debug_path(path, self.robot)
        
        # Execute manipulation
        joint_indices = self.get_joint_indices()
        self.execute_reaching_motion(path, joint_indices)
        self.execute_grasping_motion(path[-1], joint_indices[3])  # right hand
        
        # Cleanup
        if self.args.debug and path_debug:
            self.scene.clear_debug_object(path_debug)
        
        if self.args.record:
            self.camera.stop_recording(save_to_filename="video.mp4", fps=60)


def main():
    """Main function to run the Vega manipulation demo."""
    parser = argparse.ArgumentParser(
        description="Vega humanoid robot manipulation demonstration"
    )
    parser.add_argument(
        "-v", "--vis", 
        action="store_true", 
        default=False,
        help="Show the viewer during simulation"
    )
    parser.add_argument(
        "-d", "--debug", 
        action="store_true", 
        default=False,
        help="Enable debug visualization"
    )
    parser.add_argument(
        "-r", "--record", 
        action="store_true", 
        default=False,
        help="Record the simulation to video"
    )
    
    args = parser.parse_args()
    
    # Create and run the demo
    demo = VegaManipulationDemo(args)
    demo.run_demo()


if __name__ == "__main__":
    main()
