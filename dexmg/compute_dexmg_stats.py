# -*- coding: utf-8 -*-
"""
compute_dexmg_stats.py (第三次重写 —— 适配单一 schema.dim)

state 和 action 现在共用同一个维度 M（对齐 Dexora RDTRunner 的
action_dim=config["common"]["state_dim"] 约束），但统计量依然要分开
算：
    - state 和 action 数值语义不同，不能共用一份统计量
    - 同一侧（state 或 action）里，panda 组和 humanoid 组数值语义也
      不同（相对位移 vs 绝对位置），也不能混

用法：
    python compute_dexmg_stats.py \
        --dataset_root /path/to/dexmimicgen/datasets \
        --out_dir configs/
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import h5py
import numpy as np

from dexmg.dexmg_config import DATASET_CONFIGS
from dexmg.dexmg_convert import build_unified_action, build_unified_state
from dexmg.dexmg_schema import build_schema


def _iter_demo_low_dim_and_actions(hdf5_path: str, cfg):
    needed_action_keys = set(cfg["action_keys"]) | {"right_gripper", "left_gripper"}
    with h5py.File(hdf5_path, "r") as f:
        for demo_id in f["data"].keys():
            obs = {k: f[f"data/{demo_id}/obs/{k}"][()] for k in cfg["low_dim_keys"]}
            action_grp = f[f"data/{demo_id}/action_dict"]
            action_dict = {k: action_grp[k][()] for k in needed_action_keys}
            yield obs, action_dict


def _stats_from_stack(x: np.ndarray, valid_mask: np.ndarray) -> dict:
    D = x.shape[-1]
    mean = np.zeros(D, dtype=np.float64)
    std = np.ones(D, dtype=np.float64)
    norm = np.ones(D, dtype=np.float64)  # RMS，不等于 std，单独算
    mn = np.zeros(D, dtype=np.float64)
    mx = np.zeros(D, dtype=np.float64)
    q01 = np.zeros(D, dtype=np.float64)
    q99 = np.zeros(D, dtype=np.float64)

    valid_dims = np.where(valid_mask > 0)[0]
    xv = x[:, valid_dims]
    mean[valid_dims] = xv.mean(axis=0)
    std[valid_dims] = xv.std(axis=0) + 1e-6
    # RMS = sqrt(mean(x^2))，源码里 state_norm 就是这么算的，不是 std 的别名
    norm[valid_dims] = np.sqrt(np.mean(xv ** 2, axis=0)) + 1e-6
    mn[valid_dims] = xv.min(axis=0)
    mx[valid_dims] = xv.max(axis=0)
    q01[valid_dims] = np.quantile(xv, 0.01, axis=0)
    q99[valid_dims] = np.quantile(xv, 0.99, axis=0)

    return {
        "mean": mean.tolist(), "std": std.tolist(), "norm": norm.tolist(),
        "min": mn.tolist(), "max": mx.tolist(),
        "q01": q01.tolist(), "q99": q99.tolist(),
    }


def main(dataset_root: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    schema = build_schema(dataset_root=dataset_root, cache_dir=out_dir)

    group_states = defaultdict(list)
    group_actions = defaultdict(list)
    dataset_name_to_group = {}

    for hdf5_name, cfg in DATASET_CONFIGS.items():
        hdf5_path = os.path.join(dataset_root, hdf5_name)
        if not os.path.exists(hdf5_path):
            print(f"[skip] 找不到 {hdf5_path}，跳过")
            continue
        group = cfg["embodiment_group"]
        dataset_name_to_group[cfg["dataset_name"]] = group
        print(f"[scan] {hdf5_name} (group={group}) ...")

        for obs, action_dict in _iter_demo_low_dim_and_actions(hdf5_path, cfg):
            unified_state, _ = build_unified_state(obs, cfg, schema)
            unified_action, _ = build_unified_action(action_dict, cfg, schema)
            group_states[group].append(unified_state.reshape(-1, schema.dim))
            group_actions[group].append(unified_action.reshape(-1, schema.dim))

    group_state_stats = {}
    group_action_stats = {}
    for group in group_states:
        states = np.concatenate(group_states[group], axis=0)
        actions = np.concatenate(group_actions[group], axis=0)

        s_mask = schema.state_group_mask(group)
        a_mask = schema.action_group_mask(group)
        group_state_stats[group] = _stats_from_stack(states, s_mask)
        group_action_stats[group] = _stats_from_stack(actions, a_mask)
        print(f"[stats] group={group}: {states.shape[0]} frames")

    stats_path = os.path.join(out_dir, "dataset_statistics.json")
    all_stats = {}
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            all_stats = json.load(f)
    for dataset_name, group in dataset_name_to_group.items():
        all_stats[dataset_name] = {
            "state": group_state_stats[group],
            "action": group_action_stats[group],
        }
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"写入 {stats_path}")

    stat_ours_path = os.path.join(out_dir, "dataset_stat_ours.json")
    stat_ours = {}
    if os.path.exists(stat_ours_path):
        with open(stat_ours_path, "r") as f:
            stat_ours = json.load(f)
    for dataset_name, group in dataset_name_to_group.items():
        stat_ours[dataset_name] = {
            "state_mean": group_state_stats[group]["mean"],
            "state_std": group_state_stats[group]["std"],
            "state_norm": group_state_stats[group]["norm"],  # RMS，不是 std
            "action_mean": group_action_stats[group]["mean"],
            "action_std": group_action_stats[group]["std"],
            "action_norm": group_action_stats[group]["norm"],
        }
    with open(stat_ours_path, "w") as f:
        json.dump(stat_ours, f, indent=2)
    print(f"写入 {stat_ours_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="configs")
    args = parser.parse_args()
    main(args.dataset_root, args.out_dir)
