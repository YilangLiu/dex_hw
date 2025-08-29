import argparse
import os
import pickle
import shutil
from importlib import metadata

import datetime

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner


import genesis as gs

from grasp_env import GraspEnv


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.00,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0003,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "relu",
            "actor_hidden_dims": [256, 256, 128],
            "critic_hidden_dims": [256, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_obs": 14,
        "num_actions": 6,
        "action_scales": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        "episode_length_s": 3.0,
        "ctrl_dt": 0.01,
        "box_size": [0.06, 0.06, 0.06],
        "box_collision": False,
        "box_fixed": True,
        "visualize_camera": False,
    }
    reward_scales = {
        "keypoints": 1.0,
        "table_contact": 0.0 # -1.0,  # Negative scale for penalty
    }
    # vega robot specific
    robot_cfg = {
        "ee_link_name": "R_arm_l7",
        "ik_method": "gs_ik",
        "left_arm_names": [
            "L_arm_j1", "L_arm_j2", "L_arm_j3", "L_arm_j4",
            "L_arm_j5", "L_arm_j6", "L_arm_j7"
        ],
        "right_arm_names": [
            "R_arm_j1", "R_arm_j2", "R_arm_j3", "R_arm_j4",
            "R_arm_j5", "R_arm_j6", "R_arm_j7"
        ],
        "left_hand_names": [
            "L_th_j0", "L_th_j1", "L_th_j2", "L_ff_j1", "L_ff_j2",
            "L_mf_j1", "L_mf_j2", "L_rf_j1", "L_rf_j2", "L_lf_j1", "L_lf_j2"
        ],
        "right_hand_names": [
            "R_th_j0", "R_th_j1", "R_th_j2", "R_ff_j1", "R_ff_j2",
            "R_mf_j1", "R_mf_j2", "R_rf_j1", "R_rf_j2", "R_lf_j1", "R_lf_j2"
        ],
        "head_names": [
            "head_j1", "head_j2", "head_j3"
        ],
        "head_default_dof": [0.0, 0.0, 0.0],
        "left_arm_default_dof": [1.5, 0.0, 0.0, -1.756, -1.271, 0.0, 0.0],
        "right_arm_default_dof": [-1.5, 0.0, 0.0, -1.756, -1.271, 0.0, 0.0],
        "left_hand_default_dof": [1.6, -0.2668, 0.0, -1.0946, 0.0, -1.0844, 0.0,
             -1.0154, 0.0, -1.0118, 0.0],
        "right_hand_default_dof": [1.6, -0.2668, 0.0, -1.0946, 0.0, -1.0844, 0.0,
             -1.0154, 0.0, -1.0118, 0.0],
        
        
        "right_hand_close": [1.5, 0.0, -0.8, -1.0946, -0.5, -1.0844, -0.5,
             -1.0154, -0.5, -1.0118, -0.5],
        "left_hand_close_dof": [0.4, -0.2668, -0.4, -1.0946, 0.0, -1.0844, 0.0,
             -1.0154, 0.0, -1.0118, 0.0],
        "right_hand_open" : [0.0, 0.0, 0.0, -0.5, 0.0, -0.5, 0.0, -0.5, 0.0, -0.5, 0.0],
    }
    return env_cfg, reward_scales, robot_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="grasp")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=200)
    parser.add_argument("--max_iterations", type=int, default=500)
    args = parser.parse_args()

    gs.init(logging_level="warning", precision="32")

    log_dir = f"logs/{args.exp_name}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    env_cfg, reward_scales, robot_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    if args.vis:
        env_cfg["visualize_target"] = True

    pickle.dump(
        [env_cfg, reward_scales, robot_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = GraspEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        reward_cfg=reward_scales,
        robot_cfg=robot_cfg,
        show_viewer=args.vis,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python examples/manipulation/grasp_train.py
"""
