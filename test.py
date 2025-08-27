import os
import genesis as gs
import numpy as np 
from dexmate_urdf import robots
import argparse
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument("-r", "--record", action="store_true", default=False)
    args = parser.parse_args()
    device = "cuda"

    # == Genesis Initialization ==
    gs.init(seed=0, precision="32", backend=gs.gpu)

    # == Create Scene ==
    scene = gs.Scene(
            viewer_options = gs.options.ViewerOptions(
        camera_pos    = (0, -3.5, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 30,
        max_FPS       = 60,
        ),
        sim_options = gs.options.SimOptions(
            dt = 0.01,
        ),
        show_viewer = args.vis,
        rigid_options=gs.options.RigidOptions(
                    dt=0.01,
                    constraint_solver=gs.constraint_solver.Newton,
                    enable_collision=True,
                    enable_joint_limit=True,
                    enable_self_collision=False
                ),
        )

    # == add camera ==
    cam = scene.add_camera(
        res=(1280, 960),
        pos=(3.5, 0.0, 2.5),
        lookat=(0, 0, 0.5),
        fov=30,
        GUI=False,
    )

    # == add ground ==
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    
    # == add Robot ==
    vega_upper_body_only_urdf_path = robots.humanoid.vega_1.vega_upper_body.urdf
    robot = scene.add_entity(
            gs.morphs.URDF(
                file=os.path.join(vega_upper_body_only_urdf_path),
                pos=np.array([0, 0, 0.0]),
                quat=np.array([1, 0, 0, 0]),
                fixed = True
            ),
        )
    
    # == add Cube ==
    object = scene.add_entity(
            gs.morphs.Box(
                size=[0.06, 0.06, 0.06],
                fixed= False,
                collision=True,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(1.0, 0.0, 0.0),
                ),
            ),
        )

    # == Build Scene ==
    scene.build()
    if args.record:
        cam.start_recording()

    # == Place Cube at random Position with fixed Orientation ==
    random_x = torch.rand(1, device=device) * 0.3 + 0.3  # 0.3 ~ 0.6
    random_y = torch.rand(1, device=device) * 0.1 - 0.3   # -0.3 ~ -0.2
    random_z = 0.03 # torch.rand(1, device=device) * 0.3 +  0.1 # 0.3 ~ 0.4
    random_pos = torch.tensor([random_x, random_y, random_z])
    random_quat = torch.tensor([1, 0, 0, 0], device=device)
    object.set_pos(random_pos)
    object.set_quat(random_quat)
    
    # == Define Robot Default Position and Orientation ==
    end_effector = robot.get_link('R_arm_l7')
    head_names = ["head_j1", "head_j2", "head_j3"] # 3
    left_arm_names = ["L_arm_j1", "L_arm_j2", "L_arm_j3", "L_arm_j4", "L_arm_j5", "L_arm_j6", "L_arm_j7"] # 7
    right_arm_names = ["R_arm_j1", "R_arm_j2", "R_arm_j3", "R_arm_j4", "R_arm_j5", "R_arm_j6", "R_arm_j7"] # 7
    left_hand_names = ["L_th_j0", "L_th_j1", "L_th_j2", "L_ff_j1", "L_ff_j2", "L_mf_j1", "L_mf_j2", "L_rf_j1", "L_rf_j2", "L_lf_j1", "L_lf_j2"] # 11
    right_hand_names = ["R_th_j0", "R_th_j1", "R_th_j2", "R_ff_j1", "R_ff_j2", "R_mf_j1", "R_mf_j2", "R_rf_j1", "R_rf_j2", "R_lf_j1", "R_lf_j2"] # 11

    head_dofs_idx = [robot.get_joint(name).dof_idx_local for name in head_names]
    left_arm_dofs_idx = [robot.get_joint(name).dof_idx_local for name in left_arm_names]
    right_arm_dofs_idx = [robot.get_joint(name).dof_idx_local for name in right_arm_names]
    left_hand_dofs_idx = [robot.get_joint(name).dof_idx_local for name in left_hand_names]
    right_hand_dofs_idx = [robot.get_joint(name).dof_idx_local for name in right_hand_names]

    head_default = torch.tensor([0.0, 0.0, 0.0], device= device, dtype=torch.float32)
    left_arm_default = torch.tensor([1.5, 0.0, 0.0, -1.756, -1.271, 0.0, 0.0], device= device, dtype=torch.float32)
    right_arm_default = torch.tensor([-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device= device, dtype=torch.float32)
    left_hand_default  = torch.tensor([1.6, -0.2668, 0.0, -1.0946, 0.0, -1.0844, 0.0, -1.0154, 0.0, -1.0118, 0.0], device= device, dtype=torch.float32)
    right_hand_default  = torch.tensor([1.6, -0.2668, 0.0, -1.0946, 0.0, -1.0844, 0.0, -1.0154, 0.0, -1.0118, 0.0], device= device, dtype=torch.float32)

    right_hand_close = torch.tensor([0.4, -0.2668, -0.4, -1.0946, -0.5, -1.0844, -0.5, -1.0154, -0.5, -1.0118, -0.5], device= device, dtype=torch.float32)
    left_hand_close = torch.tensor([0.4, -0.2668, -0.4, -1.0946, 0.0, -1.0844, 0.0, -1.0154, 0.0, -1.0118, 0.0], device= device, dtype=torch.float32)
    
    right_hand_open = torch.tensor([0.0, 0.0, 0.0, -0.5, 0.0, -0.5, 0.0, -0.5, 0.0, -0.5, 0.0])

    # import pdb; pdb.set_trace()    
    # == Define Robot End Effector Target Position and Orientation ==
    target_offeset =  np.array([-0.2, -0.05, 0.06]) #  np.array([-0.2, -0.01, -0.06])
    target_pos =  random_pos.cpu().numpy() + target_offeset

    qpos_goal = robot.inverse_kinematics(link = end_effector, pos  = target_pos, quat = np.array([1, 0, 0, 0]),) # np.array([0.7071068, -0.7071068, 0, 0])
    path = robot.plan_path(qpos_goal = qpos_goal, num_waypoints=400)

    if args.debug:
        path_debug = scene.draw_debug_path(path, robot) # Optional: Draw future waypoints

    # Reach to the goal with right hand open 
    for waypoint in path:
        waypoint[head_dofs_idx] = head_default
        waypoint[left_arm_dofs_idx] = left_arm_default
        waypoint[left_hand_dofs_idx] = left_hand_default
        waypoint[right_hand_dofs_idx] = right_hand_open
        robot.control_dofs_position(position = waypoint)
        scene.step()
        if args.record:
            cam.render()

    # Grasp
    last_pose = path[-1]
    last_pose[right_hand_dofs_idx] = right_hand_close
    robot.control_dofs_position(last_pose) 
    for _ in range(100):
        scene.step()
        if args.record:
            cam.render()

    # Lift
    target_pos = target_pos + np.array([0.0, 0.0, 0.4])
    qpos_goal = robot.inverse_kinematics(link = end_effector, pos  = target_pos, quat = np.array([1, 0, 0, 0]),)
    path = robot.plan_path(qpos_goal = qpos_goal, num_waypoints=200)
    for waypoint in path:
        waypoint[head_dofs_idx] = head_default
        waypoint[left_arm_dofs_idx] = left_arm_default
        waypoint[left_hand_dofs_idx] = left_hand_default
        waypoint[right_hand_dofs_idx] = right_hand_close
        robot.control_dofs_position(waypoint)
        scene.step()
        if args.record:
            cam.render()

    if args.debug:
        scene.clear_debug_object(path_debug)

    if args.record:
        cam.stop_recording(save_to_filename="video.mp4", fps=60)

    return 

if __name__ == "__main__":
    main()
