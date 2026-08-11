# -*- coding: utf-8 -*-
"""
compute_dexmg_stats.py

训练前离线跑一次：扫描所有配置好的 dexmimicgen hdf5，把每条 demo 的
state/action 拼进统一 54(action)/state_dim 维 schema，然后【只在每个
embodiment group 自己的有效槽位上】算 mean/std/min/max/q01/q99——
不能把两个 group 的数据混在一起算统计量，否则 humanoid 组大量的 0
（panda槽位）会把 panda 槽位的统计量拉偏，反之亦然。

用法：
    python compute_dexmg_stats.py \
        --dataset_root /path/to/dexmimicgen/datasets/generated \
        --out_dir configs/

输出两个文件（增量合并进已有文件，不覆盖真机数据那部分）：
    - configs/dataset_stat_ours.json      : 每个 dataset_name -> state_mean（给 mask 替换用）
    - configs/dataset_statistics.json     : 每个 dataset_name -> state/action 的完整统计量
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import h5py
import numpy as np

from dexmg_config import DATASET_CONFIGS
from dexmg_convert import build_unified_action, build_unified_state
from dexmg_schema import ACTION_DIM, build_state_schema


def _iter_demo_low_dim_and_actions(hdf5_path: str, cfg):
    with h5py.File(hdf5_path, "r") as f:
        for demo_id in f["data"].keys():
            obs = {k: f[f"data/{demo_id}/obs/{k}"][()] for k in cfg["low_dim_keys"]}
            action_dict = {}
            offset = 0
            flat_action = f[f"data/{demo_id}/actions"][()]
            from dexmg_hdf5_vla_dataset import _ACTION_KEY_DIMS  # 复用同一份维度表

            for key in cfg["action_keys"]:
                dim = _ACTION_KEY_DIMS[key]
                action_dict[key] = flat_action[..., offset: offset + dim]
                offset += dim
            yield obs, action_dict


def _stats_from_stack(x: np.ndarray, valid_mask: np.ndarray) -> dict:
    """x: [N, D]，valid_mask: [D] 里为 1 的维度才参与统计，其余维度直接填 0。"""
    D = x.shape[-1]
    mean = np.zeros(D, dtype=np.float64)
    std = np.ones(D, dtype=np.float64)
    mn = np.zeros(D, dtype=np.float64)
    mx = np.zeros(D, dtype=np.float64)
    q01 = np.zeros(D, dtype=np.float64)
    q99 = np.zeros(D, dtype=np.float64)

    valid_dims = np.where(valid_mask > 0)[0]
    xv = x[:, valid_dims]
    mean[valid_dims] = xv.mean(axis=0)
    std[valid_dims] = xv.std(axis=0) + 1e-6
    mn[valid_dims] = xv.min(axis=0)
    mx[valid_dims] = xv.max(axis=0)
    q01[valid_dims] = np.quantile(xv, 0.01, axis=0)
    q99[valid_dims] = np.quantile(xv, 0.99, axis=0)

    return {
        "mean": mean.tolist(), "std": std.tolist(),
        "min": mn.tolist(), "max": mx.tolist(),
        "q01": q01.tolist(), "q99": q99.tolist(),
    }


def main(dataset_root: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    state_slots, state_dim = build_state_schema(dataset_root=dataset_root, cache_dir=out_dir)

    # 按 group 汇总所有帧，最后统一切片算 group 内统计量
    group_states = defaultdict(list)
    group_actions = defaultdict(list)
    # 同时也要记住每个 dataset_name 对应哪个 group，最后每个数据集写一份
    # （同 group 内几个数据集共用同一份统计量，和 ACG 的 MetaDexmgDataset 做法一致）
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
            unified_state, _ = build_unified_state(obs, cfg, state_slots, state_dim)
            unified_action, _ = build_unified_action(action_dict, cfg)
            group_states[group].append(unified_state.reshape(-1, state_dim))
            group_actions[group].append(unified_action.reshape(-1, ACTION_DIM))

    group_state_stats = {}
    group_action_stats = {}
    for group in group_states:
        states = np.concatenate(group_states[group], axis=0)
        actions = np.concatenate(group_actions[group], axis=0)
        from dexmg_schema import action_group_mask, state_group_mask

        s_mask = state_group_mask(state_slots, state_dim, group)
        a_mask = action_group_mask(group)
        group_state_stats[group] = _stats_from_stack(states, s_mask)
        group_action_stats[group] = _stats_from_stack(actions, a_mask)
        print(f"[stats] group={group}: {states.shape[0]} frames")

    # ---- 写 dataset_statistics.json（增量合并） ----
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

    # ---- 写 dataset_stat_ours.json（增量合并，state_mean 给 mask 替换用） ----
    stat_ours_path = os.path.join(out_dir, "dataset_stat_ours.json")
    stat_ours = {}
    if os.path.exists(stat_ours_path):
        with open(stat_ours_path, "r") as f:
            stat_ours = json.load(f)
    for dataset_name, group in dataset_name_to_group.items():
        stat_ours[dataset_name] = {
            "state_mean": group_state_stats[group]["mean"],
            "state_std": group_state_stats[group]["std"],
            # state_norm 目前和 std 同义使用；如下游有别的定义按需调整
            "state_norm": group_state_stats[group]["std"],
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
