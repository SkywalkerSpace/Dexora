#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Teacher-forcing offline evaluation for Dexora on dexmimicgen HDF5 demos.

Purpose:
    1) pick one demo from a dexmimicgen HDF5 file;
    2) read the ground-truth state sequence from the recorded obs;
    3) feed each frame into `DexoraPolicy.get_action` without any sim/env loop;
    4) compare the predicted hand/finger action against the ground-truth hand/finger
       action in the same demo;
    5) save a plot for quick diagnosis of whether the issue is in model learning or
       in sim rollout / closed-loop deployment.

This is intentionally "teacher-forcing" and open-loop: no env.reset(), no env.step(),
no controller, no dynamics. It isolates the pure prediction problem given correct state
and instruction.

Usage example:
    python Dexora/dexmg/teacher_forcing_dexora_dexmg.py \
        --hdf5 /data/dexmg/two_arm_box_cleanup.hdf5 \
        --dataset_root /data/dexmg \
        --model_path /path/to/dexora_ckpt_dir \
        --model_config_path /path/to/Dexora/configs/base_400m.yaml \
        --stats_file /path/to/Dexora/dexmg/configs/dataset_statistics.json \
        --demo_idx 0 \
        --output_dir ./offline_eval

The script automatically picks the dataset config by filename, builds the unified schema,
normalizes states to the same distribution as training, and denormalizes the model output
before comparing finger trajectories.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

DEXORA_ROOT = Path(__file__).resolve().parent.parent
if str(DEXORA_ROOT) not in sys.path:
    sys.path.insert(0, str(DEXORA_ROOT))

from dexmg.dexmg_camera import build_camera_key_map
from dexmg.dexmg_config import DatasetConfig, get_dataset_config
from dexmg.dexmg_convert import build_unified_action, build_unified_state
from dexmg.dexmg_schema import Schema, build_schema

IMAGE_SIZE = (384, 384)
DEXMG_SLOT_TO_POLICY_CAM = {
    "cam_high": "cam_head",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
    "cam_third_view": "cam_third_view",
}


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


def _get_action_head_fc2(policy: object) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return the final action-head fc2 weight and bias from the loaded policy."""
    if hasattr(policy, "policy"):
        policy_obj = policy.policy
    else:
        policy_obj = policy

    model = getattr(policy_obj, "model", None)
    if model is None:
        raise RuntimeError("无法定位 policy.model；当前 checkpoint 结构不符合 Dexora RDTRunner 预期")

    final_layer = getattr(model, "final_layer", None)
    if final_layer is None:
        raise RuntimeError("无法定位 policy.model.final_layer；当前 checkpoint 结构不符合 Dexora RDTRunner 预期")

    ffn_final = getattr(final_layer, "ffn_final", None)
    if ffn_final is None:
        raise RuntimeError("无法定位 policy.model.final_layer.ffn_final；当前 checkpoint 结构不符合 Dexora RDTRunner 预期")

    fc2 = getattr(ffn_final, "fc2", None)
    if fc2 is None:
        raise RuntimeError("无法定位 policy.model.final_layer.ffn_final.fc2；当前 checkpoint 结构不符合 Dexora RDTRunner 预期")

    return fc2.weight.detach().float().cpu(), fc2.bias.detach().float().cpu() if fc2.bias is not None else None


def _summarize_action_head_rows(weight: torch.Tensor, bias: Optional[torch.Tensor]) -> Dict[str, Dict[str, float]]:
    """Compute L2/std stats for the major action rows used in the DexMG schema."""
    weight_np = weight.detach().cpu().numpy()
    bias_np = bias.detach().cpu().numpy() if bias is not None else np.zeros(weight_np.shape[0], dtype=np.float32)
    groups = {
        "right_arm_pos": (0, 3),
        "right_gripper": (9, 15),
        "left_arm_pos": (15, 18),
        "left_gripper": (30, 36),
    }
    summary: Dict[str, Dict[str, float]] = {}
    for name, (start, end) in groups.items():
        if start >= weight_np.shape[0]:
            continue
        end = min(end, weight_np.shape[0])
        rows = weight_np[start:end]
        row_l2 = np.linalg.norm(rows, axis=1)
        row_std = rows.std(axis=1)
        row_bias_abs = np.abs(bias_np[start:end])
        summary[name] = {
            "start": int(start),
            "end": int(end),
            "l2_mean": float(row_l2.mean()),
            "l2_median": float(np.median(row_l2)),
            "l2_max": float(row_l2.max()),
            "l2_min": float(row_l2.min()),
            "std_mean": float(row_std.mean()),
            "std_median": float(np.median(row_std)),
            "std_max": float(row_std.max()),
            "bias_abs_mean": float(row_bias_abs.mean()),
            "bias_abs_median": float(np.median(row_bias_abs)),
            "bias_abs_max": float(row_bias_abs.max()),
            "bias_std": float(bias_np[start:end].std()),
        }
    return summary


def _print_action_head_diagnostics(policy: object) -> None:
    """Print a direct checkpoint-level sanity check for output rows tied to gripper DoFs."""
    try:
        weight, bias = _get_action_head_fc2(policy)
    except Exception as e:
        print(f"[head-check] could not inspect action head: {e}")
        return

    if weight.ndim != 2:
        print(f"[head-check] unexpected fc2 weight shape={tuple(weight.shape)}; skip detailed diagnostic")
        return

    summary = _summarize_action_head_rows(weight, bias)
    print("[head-check] action head row statistics (fc2):")
    for name in ["right_arm_pos", "right_gripper", "left_arm_pos", "left_gripper"]:
        if name not in summary:
            continue
        s = summary[name]
        print(
            f"  {name}[{s['start']}:{s['end']}] "
            f"weight_l2_mean={s['l2_mean']:.6e} median={s['l2_median']:.6e} max={s['l2_max']:.6e} "
            f"weight_std_mean={s['std_mean']:.6e} median={s['std_median']:.6e} "
            f"bias_abs_mean={s['bias_abs_mean']:.6e} median={s['bias_abs_median']:.6e} bias_std={s['bias_std']:.6e}"
        )

    if "right_gripper" in summary and "right_arm_pos" in summary:
        right_gripper_l2 = summary["right_gripper"]["l2_median"]
        right_arm_l2 = summary["right_arm_pos"]["l2_median"]
        ratio = right_gripper_l2 / max(right_arm_l2, 1e-12)
        print(f"[head-check] right_gripper_vs_right_arm_pos median_L2_ratio={ratio:.3e}")
        if ratio > 10.0:
            print("[head-check] WARNING: right_gripper row norms are >10x larger than normal arm rows; this strongly suggests a row-wise divergence in the action head.")

    if "left_gripper" in summary and "left_arm_pos" in summary:
        left_gripper_l2 = summary["left_gripper"]["l2_median"]
        left_arm_l2 = summary["left_arm_pos"]["l2_median"]
        ratio = left_gripper_l2 / max(left_arm_l2, 1e-12)
        print(f"[head-check] left_gripper_vs_left_arm_pos median_L2_ratio={ratio:.3e}")
        if ratio > 10.0:
            print("[head-check] WARNING: left_gripper row norms are >10x larger than normal arm rows; this strongly suggests a row-wise divergence in the action head.")


def build_images_for_policy(obs: Dict[str, np.ndarray], cfg: DatasetConfig) -> Dict[str, np.ndarray]:
    cam_map = build_camera_key_map(cfg)
    images: Dict[str, np.ndarray] = {}
    for dexmg_slot, raw_key in cam_map.items():
        if raw_key is None or raw_key not in obs:
            continue
        img = np.asarray(obs[raw_key])
        if img.ndim == 3 and img.shape[:2] != IMAGE_SIZE:
            img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_CUBIC)
        elif img.ndim == 4:
            if img.shape[0] == 0:
                continue
            img = img[0]
            if img.shape[:2] != IMAGE_SIZE:
                img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_CUBIC)
        policy_slot = DEXMG_SLOT_TO_POLICY_CAM[dexmg_slot]
        images[policy_slot] = img[::-1] if img.ndim == 3 else img
    return images


def _slot_slice(unified_action: np.ndarray, schema: Schema, slot_name: str) -> np.ndarray:
    slot = schema.slots[slot_name]
    return unified_action[..., slot.offset:slot.offset + slot.dim]


def _gripper_slice_for_group(unified_action: np.ndarray, schema: Schema, group: str) -> Tuple[np.ndarray, np.ndarray]:
    right_w = int(schema.action_real_width[group]["right_gripper"])
    left_w = int(schema.action_real_width[group]["left_gripper"])
    right_slot = schema.slots["right_gripper"]
    left_slot = schema.slots["left_gripper"]
    right = unified_action[..., right_slot.offset:right_slot.offset + right_w]
    left = unified_action[..., left_slot.offset:left_slot.offset + left_w]
    return right, left


def _load_demo(hdf5_path: str, demo_id: str, cfg: DatasetConfig, schema: Schema) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    with h5py.File(hdf5_path, "r") as f:
        obs_group = f[f"data/{demo_id}/obs"]
        action_group = f[f"data/{demo_id}/action_dict"]
        obs_arrays = {k: np.asarray(obs_group[k][()]) for k in cfg["low_dim_keys"]}
        action_arrays = {k: np.asarray(action_group[k][()]) for k in cfg["action_keys"]}
        image_arrays = {k: np.asarray(obs_group[k][()]) for k in cfg["image_keys"] if k in obs_group}

    if not obs_arrays:
        raise ValueError(f"{demo_id} 中没有 low_dim obs 可读取")
    T = next(iter(obs_arrays.values())).shape[0]
    if T <= 0:
        raise ValueError(f"{demo_id} 长度为 0")

    state_seq: List[np.ndarray] = []
    action_seq: List[np.ndarray] = []
    frame_obs: List[Dict[str, np.ndarray]] = []
    instruction = cfg["lang"]

    for t in range(T):
        obs_t = {k: np.asarray(obs_arrays[k][t]) for k in cfg["low_dim_keys"]}
        for cam_key in cfg["image_keys"]:
            if cam_key in image_arrays:
                obs_t[cam_key] = np.asarray(image_arrays[cam_key][t])
        state_raw, _ = build_unified_state(obs_t, cfg, schema)
        action_t = {k: np.asarray(action_arrays[k][t]) for k in cfg["action_keys"]}
        action_raw, _ = build_unified_action(action_t, cfg, schema)
        state_seq.append(state_raw.reshape(-1))
        action_seq.append(action_raw.reshape(-1))
        frame_obs.append(obs_t)

    return {
        "state": np.stack(state_seq, axis=0),
        "action": np.stack(action_seq, axis=0),
        "frame_obs": frame_obs,
        "instruction": instruction,
    }, image_arrays, [demo_id]


def _load_hdf5_demos(hdf5_path: str) -> List[str]:
    with h5py.File(hdf5_path, "r") as f:
        return sorted(f["data"].keys())


def run_teacher_forcing(args: argparse.Namespace) -> None:
    try:
        from deploy.dexora_policy import DexoraPolicy, DexoraPolicyConfig
    except Exception as e:
        raise RuntimeError(
            "无法导入 DexoraPolicy；请确认当前环境已安装 Dexora 运行依赖，" 
            "尤其是 diffusers / T5 / SigLIP 相关依赖。"
        ) from e

    if not os.path.exists(args.hdf5):
        raise FileNotFoundError(f"找不到 hdf5: {args.hdf5}")
    if args.stats_file is None:
        raise ValueError("--stats_file 必须提供 dataset_statistics.json")
    if args.model_path is None:
        raise ValueError("--model_path 必须提供")

    dataset_root = args.dataset_root or str(Path(args.hdf5).parent)
    schema_cache_dir = args.schema_cache_dir or dataset_root
    schema = build_schema(dataset_root=dataset_root, cache_dir=schema_cache_dir)
    cfg = get_dataset_config(args.hdf5)

    stats = json.load(open(args.stats_file, "r"))
    if cfg["dataset_name"] not in stats:
        raise KeyError(f"stats_file 中缺少 dataset_name={cfg['dataset_name']}，请先跑 compute_dexmg_stats.py")
    stats_entry = stats[cfg["dataset_name"]]

    demos = _load_hdf5_demos(args.hdf5)
    if not demos:
        raise ValueError(f"{args.hdf5} 中没有任何 demo")
    if args.demo_name is not None:
        if args.demo_name not in demos:
            raise KeyError(f"未在 {args.hdf5} 中找到 demo_name={args.demo_name}, 可用 demo={demos[:10]}")
        demo_ids = [args.demo_name]
    else:
        idx = int(args.demo_idx)
        if idx < 0 or idx >= len(demos):
            raise IndexError(f"demo_idx={idx} 超出范围 [0, {len(demos)})，可用 demo={demos[:10]}")
        demo_ids = [demos[idx]]

    os.makedirs(args.output_dir, exist_ok=True)

    policy_cfg = DexoraPolicyConfig(
        model_config_path=args.model_config_path,
        state_dim=schema.dim,
        chunk_size=args.chunk_size,
        device=args.device,
    )
    policy = DexoraPolicy(model_path=args.model_path, cfg=policy_cfg)
    _print_action_head_diagnostics(policy)

    demo_id = demo_ids[0]
    with h5py.File(args.hdf5, "r") as f:
        obs_group = f[f"data/{demo_id}/obs"]
        action_group = f[f"data/{demo_id}/action_dict"]
        low_dim = {k: np.asarray(obs_group[k][()]) for k in cfg["low_dim_keys"]}
        action_dict = {k: np.asarray(action_group[k][()]) for k in cfg["action_keys"]}

    T = next(iter(low_dim.values())).shape[0]
    if args.max_steps is not None:
        T = min(T, int(args.max_steps))

    pred_traj = []
    gt_traj = []
    gt_arm_right_pos = []
    gt_arm_right_rot6d = []
    gt_arm_left_pos = []
    gt_arm_left_rot6d = []
    gt_finger_right = []
    gt_finger_left = []
    pred_arm_right_pos = []
    pred_arm_right_rot6d = []
    pred_arm_left_pos = []
    pred_arm_left_rot6d = []
    pred_finger_right = []
    pred_finger_left = []
    pred_arm_right_pos_norm = []
    pred_arm_right_rot6d_norm = []
    pred_arm_left_pos_norm = []
    pred_arm_left_rot6d_norm = []
    pred_finger_right_norm = []
    pred_finger_left_norm = []

    for t in range(T):
        obs_t = {k: np.asarray(low_dim[k][t]) for k in cfg["low_dim_keys"]}
        for cam_key in cfg["image_keys"]:
            if cam_key in obs_group:
                obs_t[cam_key] = np.asarray(obs_group[cam_key][t])

        state_raw, _ = build_unified_state(obs_t, cfg, schema)
        state_norm = normalize(state_raw.reshape(-1), stats_entry["state"], args.normalize_mode)

        images = build_images_for_policy(obs_t, cfg)
        policy_obs = {
            "state": state_norm,
            "images": images,
            "instruction": args.instruction or cfg["lang"],
            "ctrl_freq": float(args.ctrl_freq),
        }

        action_t = {k: np.asarray(action_dict[k][t]) for k in cfg["action_keys"]}
        gt_action = build_unified_action(action_t, cfg, schema)[0].reshape(-1)
        gt_traj.append(gt_action)

        warm_start_action = None
        if args.warm_start_t is not None:
            gt_action_norm = normalize(gt_action, stats_entry["action"], args.normalize_mode)
            gt_action_norm = np.asarray(gt_action_norm, dtype=np.float32)
            gt_action_torch = torch.as_tensor(gt_action_norm, device=policy.device, dtype=policy.cfg.dtype)
            gt_action_chunk = gt_action_torch[None, None, :].expand(
                1, policy.policy.pred_horizon, -1
            )
            noise = torch.randn_like(gt_action_chunk)
            warm_start_t = torch.full(
                (1,), int(args.warm_start_t), dtype=torch.long, device=policy.device
            )
            warm_start_action = policy.policy.noise_scheduler_sample.add_noise(
                gt_action_chunk,
                noise,
                warm_start_t,
            ).detach().cpu().numpy()

        action_chunk = policy.get_action(
            policy_obs,
            warm_start_t=args.warm_start_t,
            warm_start_action=warm_start_action,
        )
        pred_action_norm = np.asarray(action_chunk[0], dtype=np.float32)
        pred_action = denormalize(pred_action_norm, stats_entry["action"], args.normalize_mode)
        pred_action = pred_action.astype(np.float32)
        pred_traj.append(pred_action)

        gt_arm_right_pos.append(_slot_slice(gt_action, schema, "right_arm_pos").reshape(-1))
        gt_arm_right_rot6d.append(_slot_slice(gt_action, schema, "right_arm_rot6d").reshape(-1))
        gt_arm_left_pos.append(_slot_slice(gt_action, schema, "left_arm_pos").reshape(-1))
        gt_arm_left_rot6d.append(_slot_slice(gt_action, schema, "left_arm_rot6d").reshape(-1))
        g_right, g_left = _gripper_slice_for_group(gt_action, schema, cfg["embodiment_group"])
        gt_finger_right.append(g_right.reshape(-1))
        gt_finger_left.append(g_left.reshape(-1))

        pred_arm_right_pos.append(_slot_slice(pred_action, schema, "right_arm_pos").reshape(-1))
        pred_arm_right_rot6d.append(_slot_slice(pred_action, schema, "right_arm_rot6d").reshape(-1))
        pred_arm_left_pos.append(_slot_slice(pred_action, schema, "left_arm_pos").reshape(-1))
        pred_arm_left_rot6d.append(_slot_slice(pred_action, schema, "left_arm_rot6d").reshape(-1))
        p_right, p_left = _gripper_slice_for_group(pred_action, schema, cfg["embodiment_group"])
        pred_finger_right.append(p_right.reshape(-1))
        pred_finger_left.append(p_left.reshape(-1))

        pred_arm_right_pos_norm.append(_slot_slice(pred_action_norm, schema, "right_arm_pos").reshape(-1))
        pred_arm_right_rot6d_norm.append(_slot_slice(pred_action_norm, schema, "right_arm_rot6d").reshape(-1))
        pred_arm_left_pos_norm.append(_slot_slice(pred_action_norm, schema, "left_arm_pos").reshape(-1))
        pred_arm_left_rot6d_norm.append(_slot_slice(pred_action_norm, schema, "left_arm_rot6d").reshape(-1))
        p_right_norm, p_left_norm = _gripper_slice_for_group(pred_action_norm, schema, cfg["embodiment_group"])
        pred_finger_right_norm.append(p_right_norm.reshape(-1))
        pred_finger_left_norm.append(p_left_norm.reshape(-1))

    gt_arm_right_pos = np.stack(gt_arm_right_pos, axis=0)
    gt_arm_right_rot6d = np.stack(gt_arm_right_rot6d, axis=0)
    gt_arm_left_pos = np.stack(gt_arm_left_pos, axis=0)
    gt_arm_left_rot6d = np.stack(gt_arm_left_rot6d, axis=0)
    gt_finger_right = np.stack(gt_finger_right, axis=0)
    gt_finger_left = np.stack(gt_finger_left, axis=0)
    pred_arm_right_pos = np.stack(pred_arm_right_pos, axis=0)
    pred_arm_right_rot6d = np.stack(pred_arm_right_rot6d, axis=0)
    pred_arm_left_pos = np.stack(pred_arm_left_pos, axis=0)
    pred_arm_left_rot6d = np.stack(pred_arm_left_rot6d, axis=0)
    pred_finger_right = np.stack(pred_finger_right, axis=0)
    pred_finger_left = np.stack(pred_finger_left, axis=0)
    pred_arm_right_pos_norm = np.stack(pred_arm_right_pos_norm, axis=0)
    pred_arm_right_rot6d_norm = np.stack(pred_arm_right_rot6d_norm, axis=0)
    pred_arm_left_pos_norm = np.stack(pred_arm_left_pos_norm, axis=0)
    pred_arm_left_rot6d_norm = np.stack(pred_arm_left_rot6d_norm, axis=0)
    pred_finger_right_norm = np.stack(pred_finger_right_norm, axis=0)
    pred_finger_left_norm = np.stack(pred_finger_left_norm, axis=0)

    arm_norm_groups = [
        ("right_arm_pos", gt_arm_right_pos, pred_arm_right_pos, pred_arm_right_pos_norm),
        ("right_arm_rot6d", gt_arm_right_rot6d, pred_arm_right_rot6d, pred_arm_right_rot6d_norm),
        ("left_arm_pos", gt_arm_left_pos, pred_arm_left_pos, pred_arm_left_pos_norm),
        ("left_arm_rot6d", gt_arm_left_rot6d, pred_arm_left_rot6d, pred_arm_left_rot6d_norm),
    ]

    print(f"[teacher_forcing] demo={demo_id} T={T} dataset={cfg['dataset_name']} group={cfg['embodiment_group']}")
    print(f"  state_dims={schema.dim} right_gripper_dim={gt_finger_right.shape[-1]} left_gripper_dim={gt_finger_left.shape[-1]}")
    print(f"  overall right MAE={np.mean(np.abs(pred_finger_right - gt_finger_right)):.6f} "
          f"RMSE={np.sqrt(np.mean((pred_finger_right - gt_finger_right)**2)):.6f}")
    print(f"  overall left MAE={np.mean(np.abs(pred_finger_left - gt_finger_left)):.6f} "
          f"RMSE={np.sqrt(np.mean((pred_finger_left - gt_finger_left)**2)):.6f}")
    for name, _, _, pred_norm_arr in arm_norm_groups:
        abs_norm = np.abs(pred_norm_arr)
        print(
            f"  raw normalized {name}: abs_max={np.max(abs_norm):.3f}, mean_abs={np.mean(abs_norm):.3f}, "
            f"p99={np.quantile(abs_norm, 0.99):.3f}, min={np.min(pred_norm_arr):.3f}, max={np.max(pred_norm_arr):.3f}"
        )
    print(f"  raw normalized right_gripper[0:5] abs max={np.max(np.abs(pred_finger_right_norm[:, :5])):.3f}, "
          f"min={np.min(pred_finger_right_norm[:, :5]):.3f}, max={np.max(pred_finger_right_norm[:, :5]):.3f}")
    print(f"  raw normalized left_gripper[0:5] abs max={np.max(np.abs(pred_finger_left_norm[:, :5])):.3f}, "
          f"min={np.min(pred_finger_left_norm[:, :5]):.3f}, max={np.max(pred_finger_left_norm[:, :5]):.3f}")

    plot_path = os.path.join(args.output_dir, f"{cfg['dataset_name']}_{demo_id}_finger_teacher_forcing.png")
    txt_path = os.path.join(args.output_dir, f"{cfg['dataset_name']}_{demo_id}_finger_teacher_forcing.txt")
    _plot_gripper_comparison(
        gt_finger_right,
        pred_finger_right,
        gt_finger_left,
        pred_finger_left,
        plot_path,
        demo_id,
    )
    _write_teacher_forcing_txt(
        txt_path,
        demo_id,
        cfg,
        gt_arm_right_pos,
        pred_arm_right_pos,
        gt_arm_right_rot6d,
        pred_arm_right_rot6d,
        gt_arm_left_pos,
        pred_arm_left_pos,
        gt_arm_left_rot6d,
        pred_arm_left_rot6d,
        gt_finger_right,
        pred_finger_right,
        gt_finger_left,
        pred_finger_left,
        pred_arm_right_pos_norm,
        pred_arm_right_rot6d_norm,
        pred_arm_left_pos_norm,
        pred_arm_left_rot6d_norm,
        pred_finger_right_norm,
        pred_finger_left_norm,
    )
    print(f"[teacher_forcing] saved plot to {plot_path}")
    print(f"[teacher_forcing] saved txt summary to {txt_path}")


def _plot_gripper_comparison(
    gt_right: np.ndarray,
    pred_right: np.ndarray,
    gt_left: np.ndarray,
    pred_left: np.ndarray,
    out_path: str,
    demo_id: str,
) -> None:
    ncols = max(1, max(gt_right.shape[-1], gt_left.shape[-1]))
    fig, axes = plt.subplots(2, ncols, figsize=(ncols * 3.2, 6.5), sharex=True)
    if ncols == 1:
        axes = np.asarray([axes]).reshape(2, 1)
    if gt_right.shape[-1] == 0:
        gt_right = np.zeros_like(pred_right)
    if gt_left.shape[-1] == 0:
        gt_left = np.zeros_like(pred_left)

    def plot_panel(ax, gt, pred, title):
        t = np.arange(len(gt))
        ax.plot(t, gt, color="tab:blue", linewidth=1.8, label="GT")
        ax.plot(t, pred, color="tab:orange", linewidth=1.4, linestyle="--", label="Pred")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    for i in range(ncols):
        if i < gt_right.shape[-1]:
            plot_panel(axes[0, i], gt_right[:, i], pred_right[:, i], f"right_gripper[{i}]")
        else:
            axes[0, i].axis("off")
        if i < gt_left.shape[-1]:
            plot_panel(axes[1, i], gt_left[:, i], pred_left[:, i], f"left_gripper[{i}]")
        else:
            axes[1, i].axis("off")

    fig.suptitle(f"Dexora teacher-forcing finger comparison\n{demo_id}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _write_teacher_forcing_txt(
    out_path: str,
    demo_id: str,
    cfg: DatasetConfig,
    gt_right_arm_pos: np.ndarray,
    pred_right_arm_pos: np.ndarray,
    gt_right_arm_rot6d: np.ndarray,
    pred_right_arm_rot6d: np.ndarray,
    gt_left_arm_pos: np.ndarray,
    pred_left_arm_pos: np.ndarray,
    gt_left_arm_rot6d: np.ndarray,
    pred_left_arm_rot6d: np.ndarray,
    gt_right: np.ndarray,
    pred_right: np.ndarray,
    gt_left: np.ndarray,
    pred_left: np.ndarray,
    pred_right_arm_pos_norm: np.ndarray,
    pred_right_arm_rot6d_norm: np.ndarray,
    pred_left_arm_pos_norm: np.ndarray,
    pred_left_arm_rot6d_norm: np.ndarray,
    pred_right_norm: np.ndarray,
    pred_left_norm: np.ndarray,
) -> None:
    max_print = 5
    with open(out_path, "w") as f:
        f.write(f"Dexora teacher-forcing summary\n")
        f.write(f"demo_id={demo_id}\n")
        f.write(f"dataset_name={cfg['dataset_name']}\n")
        f.write(f"embodiment_group={cfg['embodiment_group']}\n")
        f.write(f"normalize_mode=rms\n")
        f.write(f"T={len(pred_right_norm)}\n\n")

        arm_entries = [
            ("right_arm_pos", gt_right_arm_pos, pred_right_arm_pos, pred_right_arm_pos_norm),
            ("right_arm_rot6d", gt_right_arm_rot6d, pred_right_arm_rot6d, pred_right_arm_rot6d_norm),
            ("left_arm_pos", gt_left_arm_pos, pred_left_arm_pos, pred_left_arm_pos_norm),
            ("left_arm_rot6d", gt_left_arm_rot6d, pred_left_arm_rot6d, pred_left_arm_rot6d_norm),
            ("right_gripper", gt_right, pred_right, pred_right_norm),
            ("left_gripper", gt_left, pred_left, pred_left_norm),
        ]
        for name, gt_arr, pred_arr, pred_norm_arr in arm_entries:
            if pred_norm_arr.size == 0:
                continue
            f.write(f"== {name} (first {min(max_print, pred_norm_arr.shape[1])} dims in normalized space) ==\n")
            f.write("t\t")
            for j in range(min(max_print, pred_norm_arr.shape[1])):
                f.write(f"{name}[{j}]_norm\t{name}[{j}]_denorm\t{name}[{j}]_gt\t")
            f.write("\n")

            for t_idx in range(len(pred_norm_arr)):
                f.write(f"{t_idx}\t")
                for j in range(min(max_print, pred_norm_arr.shape[1])):
                    norm_v = float(pred_norm_arr[t_idx, j])
                    denorm_v = float(pred_arr[t_idx, j])
                    gt_v = float(gt_arr[t_idx, j])
                    f.write(f"{norm_v:.6f}\t{denorm_v:.6f}\t{gt_v:.6f}\t")
                f.write("\n")
            f.write("\n")

        f.write("== raw normalized extrema (first 5 dims only) ==\n")
        for name, pred_norm_arr in [
            ("right_arm_pos", pred_right_arm_pos_norm),
            ("right_arm_rot6d", pred_right_arm_rot6d_norm),
            ("left_arm_pos", pred_left_arm_pos_norm),
            ("left_arm_rot6d", pred_left_arm_rot6d_norm),
            ("right_gripper", pred_right_norm),
            ("left_gripper", pred_left_norm),
        ]:
            first = pred_norm_arr[:, : min(max_print, pred_norm_arr.shape[1])]
            f.write(f"{name}: max_abs_norm={np.max(np.abs(first)):.6f}, min={np.min(first):.6f}, max={np.max(first):.6f}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline teacher-forcing comparison for dexmimicgen demos.")
    parser.add_argument("--hdf5", type=str, required=True, help="单个 demo hdf5 文件路径")
    parser.add_argument("--dataset_root", type=str, default=None, help="dexmimicgen 数据集目录；默认用 hdf5 所在目录")
    parser.add_argument("--schema_cache_dir", type=str, default=None, help="schema cache 目录，默认等于 dataset_root")
    parser.add_argument("--model_path", type=str, default=None, help="Dexora checkpoint 目录或 .bin 文件")
    parser.add_argument("--model_config_path", type=str, default="configs/base_400m.yaml", help="训练时的 Dexora YAML 配置")
    parser.add_argument("--stats_file", type=str, default=None, help="compute_dexmg_stats.py 生成的 dataset_statistics.json")
    parser.add_argument("--normalize_mode", type=str, default="rms", choices=["min_max", "mean_std", "rms"])
    parser.add_argument("--demo_idx", type=int, default=0, help="hdf5 内 demo 的索引；默认取第 0 个")
    parser.add_argument("--demo_name", type=str, default=None, help="显式指定 demo 名称；不传时用 demo_idx")
    parser.add_argument("--instruction", type=str, default="", help="若不填则用 dexmg_config.py 的默认 lang")
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--ctrl_freq", type=float, default=20.0)
    parser.add_argument("--warm_start_t", type=int, default=None,
                        help="从哪个 diffusion timestep 开始 warm-start；若设定则用 GT+轻噪声作为起点")
    parser.add_argument("--max_steps", type=int, default=None, help="可选地只评估前 N 帧，便于快速诊断")
    parser.add_argument("--output_dir", type=str, default="./offline_teacher_forcing", help="输出目录")
    parser.add_argument("--device", type=str, default="cuda", help="policy 运行设备，例如 cuda / cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    run_teacher_forcing(args)


if __name__ == "__main__":
    main()
