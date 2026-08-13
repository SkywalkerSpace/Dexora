#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sim_eval_dexora_dexmg.py

在 dexmimicgen 仿真环境里评估 Dexora 策略，进程内直接调用
DexoraPolicy.get_action，不走 ZMQ server/client（和 sim_eval_dexora.py
的调用方式一致——sim 阶段没有真实时性冲突，直接调用更简单也更少出错点）。

和旧版 sim_eval_dexora.py 的区别：旧版是给单一 GR1 humanoid embodiment
写死 24 维语义表的版本；这版支持全部 6 个 dexmimicgen 数据集（panda 组 +
humanoid 组），state/action 走 dexmg_schema.py 定义的统一 42 维 schema
（M = state_dim = action_dim，对齐 Dexora 训练代码里
RDTRunner(action_dim=config["common"]["state_dim"]) 的约束），
camera/state/action 的原始 key 名和分组全部复用 dexmg_config.py 这个唯一真源。

env / 训练参数怎么"从 hdf5 里导入"：
    --dataset_hdf5   某一个 dexmimicgen hdf5，读它的 data.attrs["env_args"]
                     拿到采集时用的 env_name + controller_configs（尤其是
                     WHOLE_BODY_MINK_IK 这类必须复用的配置，见
                     load_recorded_env_config），并且通过文件名在
                     dexmg_config.DATASET_CONFIGS 里查到相机/状态/动作 key。
    --dataset_root   所有 6 个 hdf5 所在目录，给 dexmg_schema.build_schema
                     探测/加载统一 schema（gripper 宽度等）用，和
                     compute_dexmg_stats.py 用的是同一个目录、同一份缓存。
    --model_path / --model_config_path
                     Dexora checkpoint + configs/base_400m.yaml，和
                     dexora_policy.py 的用法完全一致，训练时的模型结构/
                     tokenizer_max_length/num_cameras 等参数都从这个 yaml 读。
    --stats_file     compute_dexmg_stats.py 生成的 dataset_statistics.json，
                     归一化/反归一化统计量按 dataset_name 查表。

用法示例：
    # 先看看某个 env 的 obs 里到底有哪些 key（顺带确认 dexmg_config.py 里
    # 记的 low_dim_keys / image_keys 和真实环境一致）
    python sim_eval_dexora_dexmg.py \
        --dataset_hdf5 /data/dexmg/two_arm_can_sort_random.hdf5 \
        --dataset_root /data/dexmg \
        --inspect_obs

    # 跑评估
    python sim_eval_dexora_dexmg.py \
        --dataset_hdf5 /data/dexmg/two_arm_box_cleanup.hdf5 \
        --dataset_root /data/dexmg \
        --model_path /path/to/dexora_ckpt_dir \
        --model_config_path configs/base_400m.yaml \
        --stats_file configs/dataset_statistics.json \
        --schema_cache_dir configs/ \
        --n_rollouts 5 --horizon 400 \
        --video_dir ./eval_videos

依赖：robosuite, dexmimicgen, imageio, opencv-python, h5py, numpy，以及
dexora_policy.py（需要在 PYTHONPATH 里能 import 到）和本目录下的
dexmg_config.py / dexmg_schema.py / dexmg_convert.py / dexmg_camera.py /
dexmg_rotation.py。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Optional

import cv2
import h5py
import imageio
import numpy as np
import robosuite
from robosuite import load_composite_controller_config
from robosuite.utils.transform_utils import quat2axisangle  # noqa: F401  (未直接用到，但保留以防其它 env 需要)

import dexmimicgen  # noqa: F401  必须 import 才能把自定义环境注册到 robosuite 里

from dexmg.dexmg_camera import build_camera_key_map
from dexmg.dexmg_config import DatasetConfig, get_dataset_config
from dexmg.dexmg_convert import build_unified_action, build_unified_state
from dexmg.dexmg_rotation import rot6d_to_axis_angle
from dexmg.dexmg_schema import Schema, build_schema

import sys
sys.path.append("/home/ubuntu/myh/expirement")  # 按你实际路径改，或用 PYTHONPATH 代替
from Dexora.deploy.dexora_policy import DexoraPolicy, DexoraPolicyConfig  # noqa: E402


# =============================================================================
# env_name -> robots 参数，抄自 dexmimicgen_demo_random_action.py 的 ENV_ROBOTS，
# 和 sim_eval_dexora.py 里那份保持一致（全部 6 个数据集对应的 env 都要覆盖到）。
# =============================================================================
ENV_ROBOTS = {
    "TwoArmThreading": ["Panda", "Panda"],
    "TwoArmThreePieceAssembly": ["Panda", "Panda"],
    "TwoArmTransport": ["Panda", "Panda"],
    "TwoArmLiftTray": ["PandaDexRH", "PandaDexLH"],
    "TwoArmBoxCleanup": ["PandaDexRH", "PandaDexLH"],
    "TwoArmDrawerCleanup": ["PandaDexRH", "PandaDexLH"],
    "TwoArmCoffee": ["GR1FixedLowerBody"],
    "TwoArmPouring": ["GR1FixedLowerBody"],
    "TwoArmCanSortBlue": ["GR1ArmsOnly"],
    "TwoArmCanSortRandom": ["GR1ArmsOnly"],
}

# dexmg_camera.py 用 "cam_high"，dexora_policy.py 的 DEXORA_CAMERA_ORDER
# 用的是 "cam_head" —— 同一个东西，两个文件命名不一致，这里做一次改名。
DEXMG_SLOT_TO_POLICY_CAM = {
    "cam_high": "cam_head",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
    "cam_third_view": "cam_third_view",
}

IMAGE_SIZE = (384, 384)  # (H, W)，对齐 siglip-so400m-patch14-384 的原生输入尺寸


# =============================================================================
# env / obs 相关工具函数（env 创建 + hdf5 读 env_args，和 sim_eval_dexora.py 一致）
# =============================================================================

def load_recorded_env_config(dataset_hdf5: str):
    with h5py.File(dataset_hdf5, "r") as f:
        if "data" not in f or "env_args" not in f["data"].attrs:
            raise KeyError(f"{dataset_hdf5} 缺少 data.attrs['env_args']")
        env_args_raw = f["data"].attrs["env_args"]
    if isinstance(env_args_raw, bytes):
        env_args_raw = env_args_raw.decode("utf-8")
    env_args = json.loads(env_args_raw)
    env_name = env_args.get("env_name")
    controller_configs = env_args.get("env_kwargs", {}).get("controller_configs")
    if not env_name or controller_configs is None:
        raise KeyError("env_args 中缺少 env_name 或 env_kwargs.controller_configs，无法复用录制控制器")
    return env_name, controller_configs


def make_env(env_name, camera_names, camera_height=384, camera_width=384,
             has_renderer=False, controller_configs=None):
    if env_name not in ENV_ROBOTS:
        raise ValueError(f"未知 env: {env_name}，请检查 ENV_ROBOTS 里有没有这个 key")
    robots = ENV_ROBOTS[env_name]
    env_kwargs = dict(
        env_name=env_name,
        robots=robots,
        controller_configs=(
            controller_configs if controller_configs is not None
            else load_composite_controller_config(robot=robots[0])
        ),
        has_renderer=has_renderer,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_names=camera_names,
        camera_heights=camera_height,
        camera_widths=camera_width,
        control_freq=20,
    )
    return robosuite.make(**env_kwargs)


def inspect_obs(env_name, camera_names, camera_height=384, camera_width=384, controller_configs=None):
    env = make_env(env_name, camera_names, camera_height, camera_width,
                    has_renderer=False, controller_configs=controller_configs)
    obs = env.reset()
    print(f"=== obs keys for env={env_name} (robots={ENV_ROBOTS[env_name]}) ===")
    for k in sorted(obs.keys()):
        v = obs[k]
        shape = getattr(v, "shape", None)
        dtype = getattr(v, "dtype", type(v))
        print(f"  {k:35s} shape={shape} dtype={dtype}")
    env.close()


# =============================================================================
# state / image 构造：直接复用训练时那套 dexmg_convert / dexmg_camera 模块，
# 保证 sim 评估时的 state/action 语义和训练时完全一致，不用再单独维护一份。
# =============================================================================

def build_state_from_obs(obs: dict, cfg: DatasetConfig, schema: Schema):
    """从 robosuite 实时 obs 里挑出 cfg['low_dim_keys']，走 build_unified_state。

    robosuite live obs 的 key 命名和采集 demo 时写进 hdf5 的 obs key 是同一套
    （low_dim_keys 直接来自 dexmg_config.py），所以能直接复用训练时的转换函数，
    不需要另外写一份映射。
    """
    obs_subset = {k: np.asarray(obs[k]) for k in cfg["low_dim_keys"]}
    unified_state, state_mask = build_unified_state(obs_subset, cfg, schema)
    return unified_state.astype(np.float32), state_mask


def build_images_for_policy(obs: dict, cfg: DatasetConfig) -> Dict[str, np.ndarray]:
    """dexmg 槽位名(cam_high/...) -> policy 期望的槽位名(cam_head/...)，
    缺失的相机（cam_map 值为 None）不放进 images dict，
    DexoraPolicy._encode_images 会自动用 SigLIP 均值色填充占位，
    和训练时 zero-pad 的处理方式保持一致（不要在这里手动伪造一个假机位）。
    """
    cam_map = build_camera_key_map(cfg)  # {dexmg_slot: raw_obs_key or None}
    images: Dict[str, np.ndarray] = {}
    for dexmg_slot, raw_key in cam_map.items():
        if raw_key is None:
            continue
        if raw_key not in obs:
            continue
        img = obs[raw_key][::-1]  # robosuite 图像是上下翻转的（OpenGL 惯例）
        if img.shape[:2] != IMAGE_SIZE:
            img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_CUBIC)
        policy_slot = DEXMG_SLOT_TO_POLICY_CAM[dexmg_slot]
        images[policy_slot] = img
    return images


# =============================================================================
# 归一化 / 反归一化 —— 和 compute_dexmg_stats.py 写出来的 dataset_statistics.json
# 格式对齐（mean/std/min/max/q01/q99），按 dataset_name 查表。
# =============================================================================

def normalize(data: np.ndarray, stats_entry: dict, mode: str) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    if mode == "mean_std":
        mean = np.array(stats_entry["mean"])
        std = np.array(stats_entry["std"])
        std = np.where(std == 0, 1, std)
        out = (data - mean) / std
    elif mode == "min_max":
        min_val = np.array(stats_entry["q01"])
        max_val = np.array(stats_entry["q99"])
        rng = max_val - min_val
        rng = np.where(rng == 0, 1, rng)
        out = (data - min_val) / rng
    elif mode == "rms":
        # 只缩放不减均值：state_norm = sqrt(mean(x^2))。
        # 之所以不减均值，是因为 masked/padding 维度恒为0，
        # 减均值(默认0)不影响，但如果哪天默认值改了，只缩放能保证
        # padding 维度归一化后依然是0，不破坏 mask 的"0=无效"约定。
        norm = np.array(stats_entry["norm"])
        norm = np.where(norm == 0, 1, norm)
        out = data / norm
    else:
        raise ValueError(f"未知 normalize_mode: {mode}")
    return out.astype(np.float32)


def denormalize(data: np.ndarray, stats_entry: dict, mode: str) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    if mode == "mean_std":
        mean = np.array(stats_entry["mean"])
        std = np.array(stats_entry["std"])
        out = data * std + mean
    elif mode == "min_max":
        min_val = np.array(stats_entry["q01"])
        max_val = np.array(stats_entry["q99"])
        rng = max_val - min_val
        out = data * rng + min_val
    elif mode == "rms":
        norm = np.array(stats_entry["norm"])
        out = data * norm
    else:
        raise ValueError(f"未知 normalize_mode: {mode}")
    return out.astype(np.float32)


# =============================================================================
# 统一 42 维模型输出 -> env.step() 需要的原生 flat action。
# 按 cfg["action_keys"] 的顺序拼（这个顺序直接抄自 dexmimicgen 官方
# generate_training_config.py，就是它自己训练时用的顺序，理应和
# env.action_spec 对齐 —— 不是我们自己猜的重排）。
# =============================================================================

def unified_action_to_env(unified_M: np.ndarray, cfg: DatasetConfig, schema: Schema) -> np.ndarray:
    group = cfg["embodiment_group"]

    def _slot(name: str) -> np.ndarray:
        s = schema.slots[name]
        return unified_M[..., s.offset: s.offset + s.dim]

    right_pos = _slot("right_arm_pos")
    right_rot6d = _slot("right_arm_rot6d")
    left_pos = _slot("left_arm_pos")
    left_rot6d = _slot("left_arm_rot6d")

    right_gripper_w = schema.action_real_width[group]["right_gripper"]
    left_gripper_w = schema.action_real_width[group]["left_gripper"]
    right_gripper = _slot("right_gripper")[..., :right_gripper_w]
    left_gripper = _slot("left_gripper")[..., :left_gripper_w]

    if group == "panda":
        right_rot = rot6d_to_axis_angle(right_rot6d)
        left_rot = rot6d_to_axis_angle(left_rot6d)
        component_map = {
            "right_rel_pos": right_pos, "right_rel_rot_axis_angle": right_rot,
            "right_gripper": right_gripper,
            "left_rel_pos": left_pos, "left_rel_rot_axis_angle": left_rot,
            "left_gripper": left_gripper,
        }
    elif group == "humanoid":
        component_map = {
            "right_abs_pos": right_pos, "right_abs_rot_6d": right_rot6d,
            "left_abs_pos": left_pos, "left_abs_rot_6d": left_rot6d,
            "right_gripper": right_gripper, "left_gripper": left_gripper,
        }
    else:
        raise ValueError(f"未知 embodiment_group: {group}")

    flat = np.concatenate([component_map[k] for k in cfg["action_keys"]], axis=-1)
    return flat.astype(np.float32)


# =============================================================================
# action chunk 队列
# =============================================================================

class ChunkActionQueue:
    def __init__(self):
        self._queue = []

    def empty(self):
        return len(self._queue) == 0

    def push_chunk(self, chunk):
        self._queue = list(np.asarray(chunk))

    def pop(self):
        return self._queue.pop(0)

    def clear(self):
        self._queue = []


def rollout_episode(
    env, policy, cfg: DatasetConfig, schema: Schema, instruction: str,
    horizon: int, stats: dict, normalize_mode: str, replan_interval: int,
    ctrl_freq: float = 20.0, viz_camera: str = "agentview",
    video_writer=None, live_render: bool = False,
):
    obs = env.reset()
    if live_render:
        env.render()

    ds_name = cfg["dataset_name"]
    action_queue = ChunkActionQueue()
    success = False
    t = 0
    for t in range(horizon):
        if action_queue.empty() or t % replan_interval == 0:
            state_raw, _state_mask = build_state_from_obs(obs, cfg, schema)
            state_norm = normalize(state_raw, stats[ds_name]["state"], normalize_mode)
            images = build_images_for_policy(obs, cfg)
            policy_obs = {
                "state": state_norm,
                "images": images,
                "instruction": instruction or cfg["lang"],
                "ctrl_freq": ctrl_freq,
            }
            action_chunk = policy.get_action(policy_obs)  # [chunk_size, M]，仍是归一化+统一schema空间
            action_chunk = denormalize(action_chunk, stats[ds_name]["action"], normalize_mode)
            env_action_chunk = np.stack(
                [unified_action_to_env(a, cfg, schema) for a in action_chunk], axis=0
            )
            action_queue.push_chunk(env_action_chunk)

        action = action_queue.pop()
        obs, reward, done, info = env.step(action)

        if live_render:
            env.render()
        if video_writer is not None:
            frame_key = f"{viz_camera}_image"
            if frame_key in obs:
                video_writer.append_data(obs[frame_key][::-1])

        if (t + 1) % replan_interval == 0:
            action_queue.clear()

        if hasattr(env, "_check_success") and env._check_success():
            success = True
            break
        if done:
            break

    return success, t + 1


# =============================================================================
# main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_hdf5", type=str, required=True,
                         help="要评估哪个数据集/env，从这个 hdf5 的 env_args 读取 env_name + controller_configs")
    parser.add_argument("--dataset_root", type=str, required=True,
                         help="6个hdf5所在目录，给 dexmg_schema.build_schema 用（须和 compute_dexmg_stats.py 用的是同一个目录）")
    parser.add_argument("--schema_cache_dir", type=str, default=None,
                         help="dexmg_unified_schema_cache.json 所在目录，默认等于 --dataset_root")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_config_path", type=str, default="configs/base_400m.yaml")
    parser.add_argument("--stats_file", type=str, default=None,
                         help="compute_dexmg_stats.py 生成的 dataset_statistics.json")
    parser.add_argument("--normalize_mode", type=str, default="min_max", choices=["min_max", "mean_std", "rms"])
    parser.add_argument("--instruction", type=str, default="", help="不填则用 dexmg_config.py 里该数据集的默认 lang")
    parser.add_argument("--n_rollouts", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--replan_interval", type=int, default=6)
    parser.add_argument("--video_dir", type=str, default="./eval_videos")
    parser.add_argument("--viz_camera", type=str, default="agentview")
    parser.add_argument("--camera_height", type=int, default=384)
    parser.add_argument("--camera_width", type=int, default=384)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inspect_obs", action="store_true",
                         help="只打印一次 obs keys/shape 就退出，不加载 policy")
    args = parser.parse_args()

    cfg = get_dataset_config(args.dataset_hdf5)
    schema_cache_dir = args.schema_cache_dir or args.dataset_root
    schema = build_schema(dataset_root=args.dataset_root, cache_dir=schema_cache_dir)

    env_name, recorded_controller_configs = load_recorded_env_config(args.dataset_hdf5)

    # 相机名：cfg["image_keys"] 已经是完整 obs key（含 "_image" 后缀），
    # robosuite camera_names 参数要去掉这个后缀。
    camera_names = sorted(
        {k[: -len("_image")] for k in cfg["image_keys"]} | {args.viz_camera}
    )

    if args.inspect_obs:
        inspect_obs(env_name, camera_names, args.camera_height, args.camera_width,
                    controller_configs=recorded_controller_configs)
        return

    if args.model_path is None:
        raise ValueError("--model_path 必须提供（除非只是 --inspect_obs）")
    if args.stats_file is None:
        raise ValueError("--stats_file 必须提供（dataset_statistics.json）")

    with open(args.stats_file, "r") as f:
        stats = json.load(f)
    if cfg["dataset_name"] not in stats:
        raise KeyError(
            f"{args.stats_file} 里没有 {cfg['dataset_name']} 的统计量，"
            f"先跑一遍 compute_dexmg_stats.py"
        )

    np.random.seed(args.seed)

    policy_cfg = DexoraPolicyConfig(
        model_config_path=args.model_config_path,
        state_dim=schema.dim,  # 关键：不是写死的 24，是这次 6 数据集统一 schema 的 M
        chunk_size=args.chunk_size,
    )
    policy = DexoraPolicy(model_path=args.model_path, cfg=policy_cfg)
    # DexoraPolicy.__init__ 默认把 _action_mask 设成全 1（单一 embodiment 场景下够用），
    # 这里按当前数据集所属的 group 换成真实 mask（padding 维度不应该参与采样/被当作有效动作）。
    action_mask = schema.action_group_mask(cfg["embodiment_group"])
    policy._action_mask = (
        __import__("torch").from_numpy(action_mask)[None, None, :]
        .to(policy.device, dtype=policy.cfg.dtype)
    )

    env = make_env(env_name, camera_names, args.camera_height, args.camera_width,
                    has_renderer=args.render, controller_configs=recorded_controller_configs)

    if not args.render:
        os.makedirs(args.video_dir, exist_ok=True)

    n_success = 0
    for ep in range(args.n_rollouts):
        writer = None
        video_path = None
        if not args.render:
            video_path = os.path.join(args.video_dir, f"{env_name}_ep{ep}.mp4")
            writer = imageio.get_writer(video_path, fps=20)

        t0 = time.time()
        success, n_steps = rollout_episode(
            env, policy, cfg, schema, args.instruction, args.horizon,
            stats, args.normalize_mode, args.replan_interval,
            viz_camera=args.viz_camera, video_writer=writer, live_render=args.render,
        )
        if writer is not None:
            writer.close()
        dt = time.time() - t0
        n_success += int(success)

        msg = f"[ep {ep}] success={success} steps={n_steps} time={dt:.1f}s"
        if video_path is not None:
            msg += f" video={video_path}"
        print(msg)

    print(f"=== {cfg['dataset_name']} success rate: {n_success}/{args.n_rollouts} ===")
    env.close()


if __name__ == "__main__":
    main()
