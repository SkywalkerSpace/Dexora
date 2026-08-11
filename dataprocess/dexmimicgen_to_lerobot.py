#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python dexmimicgen_to_lerobot.py --inspect /home/ubuntu/myh/experiment/dexmimicgen/datasets/two_arm_can_sort_random.hdf5

python dexmimicgen_to_lerobot.py --hdf5_files /home/ubuntu/myh/experiment/dexmimicgen/datasets/two_arm_can_sort_random.hdf5 --repo_id two_arm_can_sort_random --output_root ../lerobot_data --overwrite

dexmimicgen (robomimic HDF5) -> LeRobot v2.1 转换脚本
用于 Dexora 双臂灵巧手 VLA 数据管线（对应步骤1）

使用前必读
----------
本脚本无法凭空猜出你的 HDF5 内部具体键名（obs / action 的字段命名因 robot 组合、
controller_configs 版本而异）。第一次接入新数据时，请先跑：

    python dexmimicgen_to_lerobot.py --inspect /path/to/xxx.hdf5

把打印出来的 env_args（robots / controller_configs）、action shape、obs 各键的
shape 对照下面 CONFIG 区域逐项核对，尤其是：
  - ACTION_LAYOUT：确认 24 维 action 里各段的真实切片顺序
  - STATE_OBS_KEYS：确认 obs 里各字段的真实键名
  - CAMERA_KEY_MAP：确认相机键名，缺哪一路就去 env 里补相机，不要在这里塞占位图
不核对直接跑，大概率得到"维度对得上但语义错位"的静默 bug（比如把左手数据当成了右手）。

依赖: h5py, numpy, opencv-python, lerobot>=0.3.4 (v2.1 dataset format)
"""

import re
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import cv2
import h5py
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ============================================================
# CONFIG —— 每接入一个新任务 / 新 HDF5 之前，务必先用 --inspect 核对这里
# ============================================================

# --- 1. 24 维 (M) 语义表：顺序与步骤0/1 定好的一致，不要改动顺序 ---
STATE_ACTION_NAMES = [
    # 右臂 IK 目标 (world frame, xyz + axis-angle)
    "right_arm_x", "right_arm_y", "right_arm_z",
    "right_arm_rx", "right_arm_ry", "right_arm_rz",
    # 左臂 IK 目标
    "left_arm_x", "left_arm_y", "left_arm_z",
    "left_arm_rx", "left_arm_ry", "left_arm_rz",
    # 右手 6 维原始驱动量（环境内部经 indices=[0,0,1,1,2,2,3,3,4,4,5] 展开成 11 个关节）
    "right_hand_thumb_yaw", "right_hand_thumb_pitch_index_prox",
    "right_hand_index_mid_middle_prox", "right_hand_middle_mid_ring_prox",
    "right_hand_ring_mid_pinky_prox", "right_hand_pinky_mid",
    # 左手 6 维（同上，L_ 前缀对应的语义）
    "left_hand_thumb_yaw", "left_hand_thumb_pitch_index_prox",
    "left_hand_index_mid_middle_prox", "left_hand_middle_mid_ring_prox",
    "left_hand_ring_mid_pinky_prox", "left_hand_pinky_mid",
]
assert len(STATE_ACTION_NAMES) == 24

# --- 2. action 在 hdf5 `data/{demo}/actions` 里各段的切片方式 ---
# !! 已按 dataset_statistics.json 的实测统计特征（percentile_99 精确卡在 pi/2 的
# 维度落在"手"上而不是"臂"上）修正：真实原始顺序是 右臂|左臂|右手|左手（跟
# STATE_ACTION_NAMES 的目标顺序本来就一致），不是最初假设的 右臂|右手|左臂|左手。
# 如果你换了新的 env / 新的 controller 配置，务必重新用 --inspect + wiggle 测试核实一遍，
# 不要直接照抄这份。
ACTION_LAYOUT = {
    "right_arm": slice(0, 6),
    "left_arm": slice(6, 12),
    "right_hand": slice(12, 18),
    "left_hand": slice(18, 24),
}

# --- 3. state（observation）对应的 hdf5 obs 键名 ---
# 同样需要用 --inspect 核对；不同 robosuite/robomimic 版本命名可能是
# robot0_eef_pos / robot0_right_eef_pos / gr1_right_eef_pos 等，不保证一致。
# rot_format 填 "axisangle" 或 "quat"：若 obs 里只有四元数，脚本会自动转成轴角。
STATE_OBS_KEYS = {
    "right_arm_pos": "robot0_right_eef_pos",       # 3
    "right_arm_rot": ("robot0_right_eef_quat", "quat"),   # 4 (quat) 或 3 (axisangle)
    "left_arm_pos": "robot0_left_eef_pos",
    "left_arm_rot": ("robot0_left_eef_quat", "quat"),
    "right_hand_qpos": "robot0_right_gripper_qpos",  # 11 维实际关节角，用 HAND_INDICES 逆映射回 6 维
    "left_hand_qpos": "robot0_left_gripper_qpos",
}

# 手部 6->11 的正向映射（来自 fourier_hands.py 源码 indices 数组），
# 用于把 obs 里的 11 维实际 qpos 逆映射回 6 维近似 state，跟 action 对齐。
HAND_INDICES = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5])

# --- 4. 相机键名映射：dexmimicgen 原始相机名 -> 目标 4 路相机名 ---
# 目标名固定为 top / wrist_left / wrist_right / front。若某一路在环境里还没配置，
# 先在 env 里加自定义相机位姿重新采数据，不要在转换脚本里用黑图占位（会给模型引入噪声先验）。
CAMERA_KEY_MAP = {
    # "top": 没有对应键，见下方说明
    "wrist_left": "robot0_eye_in_left_hand_image",
    "wrist_right": "robot0_eye_in_right_hand_image",
    "front": "frontview_image",   # 这个不用改
}
# 直接对齐 siglip-so400m-patch14-384 的输入尺寸（384x384）
IMAGE_SIZE = (384, 384)  # (H, W)

# --- 5. 每个任务的 5 条语言指令 phrasing，key 用任务名（需能从 hdf5 文件名里匹配到）---
TASK_INSTRUCTIONS = {
    "Two_Arm_Pouring": [
        "Pour the contents from the cup into the bowl using both hands.",
        "Use both arms to pour from the cup into the bowl.",
        "Pick up the cup and pour it into the bowl.",
        "Carefully pour the liquid from the cup into the bowl with both hands.",
        "Grasp the cup and empty it into the bowl.",
    ],
    "Two_Arm_Coffee": [
        "Make coffee by placing the pod and closing the machine lid.",
        "Insert the coffee pod and close the lid to start brewing.",
        "Use both hands to load the pod and shut the coffee machine.",
        "Place the capsule inside the machine and close it.",
        "Prepare the coffee machine by inserting the pod and closing the lid.",
    ],
    "Two_Arm_Can_Sort_Random": [
        "Sort the blue can into the correct bin using both arms.",
        "Pick up the blue can and place it in the designated bin.",
        "Use both hands to move the blue can to its sorting bin.",
        "Grasp the blue can and sort it into the correct container.",
        "Move the blue can into the blue sorting bin.",
    ],
}

DEFAULT_FPS = 20  # 跟 dexmimicgen 采集/回放帧率一致


# ============================================================
# Inspect 工具：跑一遍就能看到需要填进上面 CONFIG 的所有信息
# ============================================================

def inspect_hdf5(path: str, n_demo_preview: int = 1):
    with h5py.File(path, "r") as f:
        print("=" * 60)
        print(f"文件: {path}")
        if "data" in f and "env_args" in f["data"].attrs:
            env_args = json.loads(f["data"].attrs["env_args"])
            print("env_args.env_name:", env_args.get("env_name"))
            print("env_args.env_kwargs:", env_args.get("env_kwargs", {}))
            print("env_args.env_kwargs.robots:", env_args.get("env_kwargs", {}).get("robots"))
            cc = env_args.get("env_kwargs", {}).get("controller_configs")
            print("controller_configs:")
            print(json.dumps(cc, indent=2, ensure_ascii=False)[:3000])
        else:
            print("!! 没找到 data.attrs['env_args']，请手动确认 action/obs 语义来源")

        demos = list(f["data"].keys())
        print(f"\n共 {len(demos)} 条 demo，示例: {demos[:5]}")

        for demo in demos[:n_demo_preview]:
            g = f[f"data/{demo}"]
            print(f"\n--- {demo} ---")
            print("actions shape:", g["actions"].shape)
            print("obs keys:")
            for k in g["obs"].keys():
                print(f"  {k}: {g['obs'][k].shape} {g['obs'][k].dtype}")
        print("=" * 60)
        print("对照上面的输出核对 CONFIG 区域的 ACTION_LAYOUT / STATE_OBS_KEYS / CAMERA_KEY_MAP")


# ============================================================
# 核心转换逻辑
# ============================================================

def quat_to_axisangle(quat_xyzw: np.ndarray) -> np.ndarray:
    """四元数 (x,y,z,w) 转轴角向量 (rx,ry,rz)，模长=旋转角(弧度)。"""
    q = quat_xyzw / (np.linalg.norm(quat_xyzw) + 1e-8)
    xyz, w = q[:3], q[3]
    sin_half = np.linalg.norm(xyz)
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(sin_half, w)
    axis = xyz / sin_half
    return (axis * angle).astype(np.float32)


def hand_qpos11_to_action6(qpos11: np.ndarray) -> np.ndarray:
    """把 11 维实际关节角 qpos 逆映射回 6 维近似 action/state。
    HAND_INDICES=[0,0,1,1,2,2,3,3,4,4,5] 表示 qpos 里哪些位置共享同一个驱动量，
    这里对每组取均值回归出 6 维（action[5] 只对应 qpos[10] 一个位置，等价于直接取值）。
    """
    out = np.zeros(6, dtype=np.float32)
    for a_idx in range(6):
        positions = np.where(HAND_INDICES == a_idx)[0]
        out[a_idx] = qpos11[positions].mean()
    return out


def _read_rot(obs_group: h5py.Group, key_cfg, frame_idx: int) -> np.ndarray:
    key, fmt = key_cfg if isinstance(key_cfg, tuple) else (key_cfg, "axisangle")
    val = obs_group[key][frame_idx]
    if fmt == "quat":
        return quat_to_axisangle(val)
    return val.astype(np.float32)


def build_state_vector(obs_group: h5py.Group, frame_idx: int) -> np.ndarray:
    right_pos = obs_group[STATE_OBS_KEYS["right_arm_pos"]][frame_idx].astype(np.float32)
    right_rot = _read_rot(obs_group, STATE_OBS_KEYS["right_arm_rot"], frame_idx)
    left_pos = obs_group[STATE_OBS_KEYS["left_arm_pos"]][frame_idx].astype(np.float32)
    left_rot = _read_rot(obs_group, STATE_OBS_KEYS["left_arm_rot"], frame_idx)

    right_hand_qpos = obs_group[STATE_OBS_KEYS["right_hand_qpos"]][frame_idx]
    left_hand_qpos = obs_group[STATE_OBS_KEYS["left_hand_qpos"]][frame_idx]
    right_hand6 = (
        hand_qpos11_to_action6(right_hand_qpos) if right_hand_qpos.shape[-1] == 11
        else right_hand_qpos.astype(np.float32)
    )
    left_hand6 = (
        hand_qpos11_to_action6(left_hand_qpos) if left_hand_qpos.shape[-1] == 11
        else left_hand_qpos.astype(np.float32)
    )

    state = np.concatenate([right_pos, right_rot, left_pos, left_rot, right_hand6, left_hand6])
    assert state.shape[0] == 24, f"state 维度不是 24，而是 {state.shape[0]}，检查 STATE_OBS_KEYS 配置"
    return state.astype(np.float32)


def build_action_vector(actions: np.ndarray, frame_idx: int) -> np.ndarray:
    a = actions[frame_idx]
    assert a.shape[0] == 24, f"action 维度不是 24，而是 {a.shape[0]}，检查 ACTION_LAYOUT / 原始 action_spec"
    # 按 ACTION_LAYOUT 重新拼接成 STATE_ACTION_NAMES 的顺序。
    # 如果原始顺序本来就一致，这一步是恒等操作；如果不一致，只需要改 ACTION_LAYOUT。
    return np.concatenate([
        a[ACTION_LAYOUT["right_arm"]],
        a[ACTION_LAYOUT["left_arm"]],
        a[ACTION_LAYOUT["right_hand"]],
        a[ACTION_LAYOUT["left_hand"]],
    ]).astype(np.float32)


def load_camera_frame(obs_group: h5py.Group, cam_hdf5_key: str, frame_idx: int) -> np.ndarray:
    img = obs_group[cam_hdf5_key][frame_idx]
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    if img.shape[:2] != IMAGE_SIZE:
        img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_CUBIC)
    return img


def make_features() -> Dict:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (24,),
            "names": STATE_ACTION_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (24,),
            "names": STATE_ACTION_NAMES,
        },
    }
    for cam_name in CAMERA_KEY_MAP.keys():
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": (IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
            "names": ["height", "width", "channels"],
        }
    return features


def resolve_task_name(hdf5_path: str) -> str:
    stem = Path(hdf5_path).stem
    for task_name in TASK_INSTRUCTIONS:
        if task_name.lower() in stem.lower():
            return task_name
    raise ValueError(
        f"无法从文件名 {hdf5_path} 推断任务名，请在 TASK_INSTRUCTIONS 里补充对应任务，"
        f"或在文件名里包含任务关键字（如 TwoArmPouring）"
    )


def get_can_target_color(model_file_xml: str) -> str:
    cube_rgba = re.search(r'<geom name="cube_g0_vis"[^>]*rgba="([^"]+)"', model_file_xml).group(1)
    red_rgba  = re.search(r'<geom name="red_box_base_vis"[^>]*rgba="([^"]+)"', model_file_xml).group(1)
    blue_rgba = re.search(r'<geom name="blue_box_base_vis"[^>]*rgba="([^"]+)"', model_file_xml).group(1)
    if cube_rgba == red_rgba:
        return "red"
    elif cube_rgba == blue_rgba:
        return "blue"
    else:
        return "blue"


def convert_one_hdf5(hdf5_path: str, dataset: LeRobotDataset, fps: int, demo_filter: List[str] = None):
    task_name = resolve_task_name(hdf5_path)
    instructions = TASK_INSTRUCTIONS[task_name]

    with h5py.File(hdf5_path, "r") as f:
        demos = demo_filter or list(f["data"].keys())
        for demo in demos:
            g = f[f"data/{demo}"]
            actions = g["actions"][:]
            obs = g["obs"]
            num_frames = actions.shape[0]

            # 每个 episode 固定用同一条 phrasing（保证一个 episode 内指令不跳变），
            # 5 条 phrasing 在不同 episode 间随机分布，最终整体覆盖到全部 5 条。
            instruction = random.choice(instructions)

            '''
            model_xml = g.attrs["model_file"] if "model_file" in g.attrs else f[f"data/{demo}"].attrs.get("model_file")
            color = get_can_target_color(model_xml)
            instruction = random.choice([
                f"Sort the {color} can into the correct bin using both arms.",
                f"Pick up the {color} can and place it in the designated bin.",
                f"Use both hands to move the {color} can to its sorting bin.",
                f"Grasp the {color} can and sort it into the correct container.",
                f"Move the {color} can into the {color} sorting bin.",
            ])
            '''

            for t in range(num_frames):
                frame = {
                    "observation.state": build_state_vector(obs, t),
                    "action": build_action_vector(actions, t),
                }
                for cam_name, hdf5_key in CAMERA_KEY_MAP.items():
                    if hdf5_key not in obs:
                        raise KeyError(
                            f"{demo} 的 obs 里没有 {hdf5_key}（目标相机 {cam_name}）。"
                            f"请先在环境里补上这路相机再重新采数据，不要用占位图代替。"
                        )
                    frame[f"observation.images.{cam_name}"] = load_camera_frame(obs, hdf5_key, t)

                dataset.add_frame(frame, task=instruction, timestamp=t / fps)

            dataset.save_episode()
            print(f"[{task_name}] {demo}: {num_frames} 帧, instruction=\"{instruction}\"")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", type=str, default=None, help="传入一个 hdf5 路径，只打印结构不做转换")
    parser.add_argument("--hdf5_dir", type=str, default=None, help="包含多个任务 hdf5 的目录，按文件名自动匹配任务")
    parser.add_argument("--hdf5_files", type=str, nargs="+", default=None, help="也可以直接给一串 hdf5 文件路径")
    parser.add_argument("--repo_id", type=str, default="dexora_dexmimicgen")
    parser.add_argument("--output_root", type=str, default="./lerobot_data")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.inspect:
        inspect_hdf5(args.inspect)
        return

    hdf5_files = []
    if args.hdf5_dir:
        hdf5_files += [str(p) for p in Path(args.hdf5_dir).glob("*.hdf5")]
    if args.hdf5_files:
        hdf5_files += args.hdf5_files
    if not hdf5_files:
        raise ValueError("请用 --hdf5_dir 或 --hdf5_files 指定输入数据，或先用 --inspect 看结构")

    dataset_root = Path(args.output_root) / args.repo_id
    if dataset_root.exists() and args.overwrite:
        import shutil
        shutil.rmtree(dataset_root)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=make_features(),
        root=str(dataset_root),
        robot_type="dexora",
        use_videos=True,
    )

    for hdf5_path in hdf5_files:
        convert_one_hdf5(hdf5_path, dataset, fps=args.fps)

    print(f"\n转换完成，共 {dataset.num_episodes} episodes, {len(dataset)} 帧")
    print(f"数据集路径: {dataset_root}")
    print("下一步: python -m data.lerobot_vla_dataset --stat 重新生成 dataset_statistics.json（不要沿用官方统计量）")


if __name__ == "__main__":
    main()
