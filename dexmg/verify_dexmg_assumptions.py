#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_dexmg_left_right_v2.py

v1（verify_dexmg_assumptions.py 里的检查1）把整条 demo 的位移拼成一个长向量
算一次余弦相似度，容易被大量"没怎么在动"的帧稀释掉真实信号，单条 demo 上
四个相似度经常卡在 0.4~0.7 分不出来（实测 two_arm_box_cleanup demo_10 就是
这样）。

这版改成：
  1. 只统计"这只手确实在明显移动"的帧（按这条 demo 自己的位移幅度动态定阈值），
     而不是不分青红皂白地把所有帧都算进去；
  2. 逐帧算余弦相似度再取平均，而不是把整条轨迹拼成一个大向量算一次；
  3. 一次跑多条 demo，把结果累加，摊薄单条 demo 的偶然性。

用法：
    python verify_dexmg_left_right.py \
        --dataset_root /home/mayuhang/datasets/dexmimicgen_datasets \
        --hdf5 two_arm_box_cleanup.hdf5 \
        --num_demos 20
"""
from __future__ import annotations

import argparse

import h5py
import numpy as np


def per_frame_cos_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a, b: (T, 3) -> (T,) 逐帧余弦相似度。"""
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    denom = np.clip(na * nb, 1e-8, None)
    return np.sum(a * b, axis=-1) / denom


def collect_stats(f: h5py.File, demo_keys: list[str], active_ratio: float = 0.3):
    """在多条 demo 上累积 4 种配对(robot0/1 x right/left)的"活跃帧"余弦相似度。"""
    buckets = {
        "r0_right": [], "r0_left": [],
        "r1_right": [], "r1_left": [],
    }
    n_used_demos = 0

    for demo in demo_keys:
        obs = f[f"data/{demo}/obs"]
        action_dict = f[f"data/{demo}/action_dict"]
        if not all(k in obs for k in ["robot0_eef_pos", "robot1_eef_pos"]):
            continue
        if not all(k in action_dict for k in ["right_rel_pos", "left_rel_pos"]):
            continue

        robot0_pos = obs["robot0_eef_pos"][()]
        robot1_pos = obs["robot1_eef_pos"][()]
        right_rel_pos = action_dict["right_rel_pos"][()]
        left_rel_pos = action_dict["left_rel_pos"][()]

        T = min(len(robot0_pos) - 1, len(right_rel_pos), len(left_rel_pos))
        if T < 5:
            continue
        robot0_delta = np.diff(robot0_pos, axis=0)[:T]
        robot1_delta = np.diff(robot1_pos, axis=0)[:T]
        right_rel_pos = right_rel_pos[:T]
        left_rel_pos = left_rel_pos[:T]

        # 每条 demo 自己定"活跃帧"阈值：用这条 demo 里该动作序列位移幅度的
        # 90分位数 * active_ratio 做阈值，只在该动作明显发生位移的帧上比较。
        right_mag = np.linalg.norm(right_rel_pos, axis=-1)
        left_mag = np.linalg.norm(left_rel_pos, axis=-1)
        right_thresh = np.quantile(right_mag, 0.9) * active_ratio
        left_thresh = np.quantile(left_mag, 0.9) * active_ratio

        right_active = right_mag > max(right_thresh, 1e-6)
        left_active = left_mag > max(left_thresh, 1e-6)

        if right_active.sum() >= 3:
            buckets["r0_right"].append(
                per_frame_cos_sim(robot0_delta[right_active], right_rel_pos[right_active]))
            buckets["r1_right"].append(
                per_frame_cos_sim(robot1_delta[right_active], right_rel_pos[right_active]))
        if left_active.sum() >= 3:
            buckets["r0_left"].append(
                per_frame_cos_sim(robot0_delta[left_active], left_rel_pos[left_active]))
            buckets["r1_left"].append(
                per_frame_cos_sim(robot1_delta[left_active], left_rel_pos[left_active]))

        n_used_demos += 1

    print(f"实际用上的 demo 数: {n_used_demos} / {len(demo_keys)}")
    return {k: (np.concatenate(v) if len(v) > 0 else np.array([])) for k, v in buckets.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--hdf5", type=str, default="two_arm_box_cleanup.hdf5")
    parser.add_argument("--num_demos", type=int, default=20,
                         help="从头开始取多少条 demo 一起统计（越多越稳，但会变慢）")
    parser.add_argument("--active_ratio", type=float, default=0.3,
                         help="活跃帧阈值 = 该动作位移幅度的90分位数 * active_ratio，"
                              "数值越小纳入的帧越多（含更多噪声），越大越严格（帧数变少）")
    args = parser.parse_args()

    hdf5_path = f"{args.dataset_root}/{args.hdf5}"
    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
        demo_keys = demo_keys[: args.num_demos]
        print(f"数据集: {hdf5_path}")
        print(f"计划使用前 {len(demo_keys)} 条 demo，active_ratio={args.active_ratio}")

        stats = collect_stats(f, demo_keys, active_ratio=args.active_ratio)

    print("\n" + "=" * 70)
    print("活跃帧（该动作明显发生位移的帧）上的逐帧余弦相似度统计")
    print("=" * 70)
    for key in ["r0_right", "r0_left", "r1_right", "r1_left"]:
        arr = stats[key]
        if len(arr) == 0:
            print(f"{key}: 没有足够的活跃帧，跳过")
            continue
        print(f"{key}: n_frames={len(arr):5d}  mean={arr.mean():+.4f}  "
              f"median={np.median(arr):+.4f}  std={arr.std():.4f}")

    r0_right = stats["r0_right"].mean() if len(stats["r0_right"]) else np.nan
    r0_left = stats["r0_left"].mean() if len(stats["r0_left"]) else np.nan
    r1_right = stats["r1_right"].mean() if len(stats["r1_right"]) else np.nan
    r1_left = stats["r1_left"].mean() if len(stats["r1_left"]) else np.nan

    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    hypothesis_a = (r0_right if not np.isnan(r0_right) else -1) + \
                   (r1_left if not np.isnan(r1_left) else -1)
    hypothesis_b = (r0_left if not np.isnan(r0_left) else -1) + \
                   (r1_right if not np.isnan(r1_right) else -1)
    print(f"假设A (robot0=右, robot1=左) 得分: {hypothesis_a:+.4f}  "
          f"(= r0_right {r0_right:+.4f} + r1_left {r1_left:+.4f})")
    print(f"假设B (robot0=左, robot1=右) 得分: {hypothesis_b:+.4f}  "
          f"(= r0_left {r0_left:+.4f} + r1_right {r1_right:+.4f})")
    gap = abs(hypothesis_a - hypothesis_b)
    if gap < 0.3:
        print(f"\n>>> 两个假设得分差距只有 {gap:.4f}，还是不够明确，建议加大 --num_demos "
              f"或换一个双臂配合更紧密的数据集（two_arm_lift_tray 通常两臂同时动，"
              f"信号可能更强）")
    elif hypothesis_a > hypothesis_b:
        print("\n>>> 假设A胜出: robot0 = 右臂, robot1 = 左臂 —— 和 dexmg_schema.py 当前假设一致 [OK]")
    else:
        print("\n>>> 假设B胜出: robot0 = 左臂, robot1 = 右臂 —— 和 dexmg_schema.py 当前假设相反 [FAIL]，"
              "需要去 dexmg_schema.py::_low_dim_key_roles 把左右判断反过来")


if __name__ == "__main__":
    main()
