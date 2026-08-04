#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python dexmimicgen_to_lerobot.py --inspect /home/ubuntu/myh/expirement/dexmimicgen/datasets/two_arm_can_sort_random.hdf5

python dexmimicgen_to_lerobot.py --hdf5_files /home/ubuntu/myh/expirement/dexmimicgen/datasets/two_arm_can_sort_random.hdf5 --repo_id dexora_v1 --output_root ../lerobot_data --overwrite

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

import argparse
import json
import multiprocessing as mp
import os
import random
from functools import partial
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
# 直接对齐 siglip-so400m-patch14-384 的输入尺寸（384x384），避免训练时先resize到256
# 再被 SigLIP processor 二次resize到384，两次插值会比一次插值更糊。
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
DEFAULT_NUM_WORKERS = os.cpu_count() or 4


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
# 向量化 state/action 构建：一次性处理整段 (T, ...) 数组，
# 避免逐帧对 hdf5 做随机读取（这是转换脚本最大的性能瓶颈）。
# ============================================================

def quat_to_axisangle_batch(quat_xyzw: np.ndarray) -> np.ndarray:
    """(T,4) -> (T,3) 四元数转轴角，向量化版本。"""
    q = quat_xyzw / (np.linalg.norm(quat_xyzw, axis=1, keepdims=True) + 1e-8)
    xyz, w = q[:, :3], q[:, 3]
    sin_half = np.linalg.norm(xyz, axis=1, keepdims=True)
    angle = 2.0 * np.arctan2(sin_half, w[:, None])
    axis = np.where(sin_half < 1e-8, 0.0, xyz / np.clip(sin_half, 1e-8, None))
    return (axis * angle).astype(np.float32)


def hand_qpos11_to_action6_batch(qpos11: np.ndarray) -> np.ndarray:
    """(T,11) -> (T,6)，用矩阵乘法代替逐帧 python 循环做耦合关节求均值。"""
    weight = np.zeros((11, 6), dtype=np.float32)
    for i, slot in enumerate(HAND_INDICES):
        weight[i, slot] = 1.0
    counts = weight.sum(axis=0, keepdims=True)
    return (qpos11.astype(np.float32) @ weight) / counts


def build_state_array(obs_group: h5py.Group) -> np.ndarray:
    """一次性读整段 hdf5 数组、向量化转换，返回 (T, 24)。

    替代旧版逐帧调用 build_state_vector(obs, t) 的写法——旧写法每帧都单独
    对 hdf5 做一次切片读取，1020 个 demo x 每个几百帧下来是几十万次零散小
    读取，是最大的性能瓶颈。这里改成每个 demo 只读 6 次（每个 obs key 一次）。
    """
    right_pos = obs_group[STATE_OBS_KEYS["right_arm_pos"]][:].astype(np.float32)
    right_quat = obs_group[STATE_OBS_KEYS["right_arm_rot"][0]][:].astype(np.float32)
    left_pos = obs_group[STATE_OBS_KEYS["left_arm_pos"]][:].astype(np.float32)
    left_quat = obs_group[STATE_OBS_KEYS["left_arm_rot"][0]][:].astype(np.float32)
    right_hand_qpos = obs_group[STATE_OBS_KEYS["right_hand_qpos"]][:]
    left_hand_qpos = obs_group[STATE_OBS_KEYS["left_hand_qpos"]][:]

    right_rot = quat_to_axisangle_batch(right_quat)
    left_rot = quat_to_axisangle_batch(left_quat)
    right_hand6 = (
        hand_qpos11_to_action6_batch(right_hand_qpos) if right_hand_qpos.shape[-1] == 11
        else right_hand_qpos.astype(np.float32)
    )
    left_hand6 = (
        hand_qpos11_to_action6_batch(left_hand_qpos) if left_hand_qpos.shape[-1] == 11
        else left_hand_qpos.astype(np.float32)
    )

    state = np.concatenate(
        [right_pos, right_rot, left_pos, left_rot, right_hand6, left_hand6], axis=1
    )
    assert state.shape[1] == 24, f"state 维度不是 24，而是 {state.shape[1]}，检查 STATE_OBS_KEYS 配置"
    return state.astype(np.float32)


def build_action_array(actions: np.ndarray) -> np.ndarray:
    """(T,24) -> (T,24)，按 ACTION_LAYOUT 一次性重排整段 action。

    替代旧版逐帧调用 build_action_vector(actions, t)。
    """
    assert actions.shape[1] == 24, f"action 维度不是 24，而是 {actions.shape[1]}，检查 ACTION_LAYOUT / 原始 action_spec"
    return np.concatenate([
        actions[:, ACTION_LAYOUT["right_arm"]],
        actions[:, ACTION_LAYOUT["left_arm"]],
        actions[:, ACTION_LAYOUT["right_hand"]],
        actions[:, ACTION_LAYOUT["left_hand"]],
    ], axis=1).astype(np.float32)


def resize_frame(img: np.ndarray) -> np.ndarray:
    """只做类型转换 + resize，不做 hdf5 读取（读取已经在 worker 进程里一次性做完）。
    放大倍数较大（84x84 -> 384x384），用 INTER_CUBIC 比默认的 INTER_LINEAR 更清晰。
    """
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


# ============================================================
# 多进程 worker：只负责"读 hdf5 + 算 state/action"，不碰 LeRobotDataset。
# h5py.File 不能跨进程传递，所以 worker 里用 hdf5_path 自己重新打开文件；
# 只传回纯 numpy 数据（state/action/原始分辨率图像），LeRobotDataset 的写入
# （add_frame/save_episode，涉及视频编码等有状态资源）必须留在主进程里串行做。
# ============================================================

def process_demo(hdf5_path: str, demo_name: str) -> dict:
    with h5py.File(hdf5_path, "r") as f:
        g = f[f"data/{demo_name}"]
        actions = g["actions"][:]
        obs = g["obs"]
        num_frames = actions.shape[0]

        state_all = build_state_array(obs)
        action_all = build_action_array(actions)

        # 故意不在 worker 里 resize：保持原始(比如84x84)分辨率传回主进程，
        # 减少进程间 pickle/IPC 的数据量（384x384 传回会大好几倍）。
        cam_raw = {}
        for cam_name, hdf5_key in CAMERA_KEY_MAP.items():
            if hdf5_key not in obs:
                raise KeyError(
                    f"{demo_name} 的 obs 里没有 {hdf5_key}（目标相机 {cam_name}）。"
                    f"请先在环境里补上这路相机再重新采数据，不要用占位图代替。"
                )
            cam_raw[cam_name] = obs[hdf5_key][:]

    task_name = resolve_task_name(hdf5_path)
    instruction = random.choice(TASK_INSTRUCTIONS[task_name])

    return {
        "demo_name": demo_name,
        "state": state_all,
        "action": action_all,
        "cams": cam_raw,
        "num_frames": num_frames,
        "instruction": instruction,
    }


# ============================================================
# 核心转换逻辑
# ============================================================

def convert_one_hdf5(
    hdf5_path: str,
    dataset: LeRobotDataset,
    fps: int,
    demo_filter: List[str] = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
):
    task_name = resolve_task_name(hdf5_path)  # 提前算一次，仅用于打印

    with h5py.File(hdf5_path, "r") as f:
        demos = demo_filter or list(f["data"].keys())

    worker_fn = partial(process_demo, hdf5_path)

    with mp.Pool(processes=num_workers) as pool:
        # imap 按 demos 原顺序返回结果，方便调试对照；不在意顺序的话换成
        # imap_unordered 吞吐会更高一点。
        for result in pool.imap(worker_fn, demos, chunksize=1):
            num_frames = result["num_frames"]
            for t in range(num_frames):
                frame = {
                    "observation.state": result["state"][t],
                    "action": result["action"][t],
                }
                for cam_name in CAMERA_KEY_MAP:
                    frame[f"observation.images.{cam_name}"] = resize_frame(result["cams"][cam_name][t])

                dataset.add_frame(frame, task=result["instruction"], timestamp=t / fps)

            dataset.save_episode()
            print(f"[{task_name}] {result['demo_name']}: {num_frames} 帧, instruction=\"{result['instruction']}\"")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", type=str, default=None, help="传入一个 hdf5 路径，只打印结构不做转换")
    parser.add_argument("--hdf5_dir", type=str, default=None, help="包含多个任务 hdf5 的目录，按文件名自动匹配任务")
    parser.add_argument("--hdf5_files", type=str, nargs="+", default=None, help="也可以直接给一串 hdf5 文件路径")
    parser.add_argument("--repo_id", type=str, default="dexora_dexmimicgen")
    parser.add_argument("--output_root", type=str, default="./lerobot_data")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--num_workers", type=int, default=DEFAULT_NUM_WORKERS,
        help="并行读取/预处理 hdf5 demo 的进程数，默认用 CPU 核数",
    )
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
        convert_one_hdf5(hdf5_path, dataset, fps=args.fps, num_workers=args.num_workers)

    print(f"\n转换完成，共 {dataset.num_episodes} episodes, {len(dataset)} 帧")
    print(f"数据集路径: {dataset_root}")
    print("下一步: python -m data.lerobot_vla_dataset --stat 重新生成 dataset_statistics.json（不要沿用官方统计量）")


if __name__ == "__main__":
    main()
