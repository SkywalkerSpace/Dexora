#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_demo_frame_alignment.py

最直接的核对脚本：
  1) 训练数据集的 __getitem__ / HDF5 reader 路径里，取 demo_1006 的第 50 帧
  2) 抽出 item['actions'] 中的 right_gripper 原始值（归一化前）
  3) 直接用 h5py 读同一帧 action_dict，并走 build_unified_action() 重新拼装
  4) 对比两者是否一致；若一致则说明训练数据管线和统一转换函数一致。

示例：
    python Dexora/dexmg/verify_demo_frame_alignment.py \
        --dataset_root /path/to/dexmimicgen_dataset \
        --hdf5 /path/to/dexmimicgen_dataset/two_arm_box_cleanup.hdf5 \
        --demo_id demo_1006 \
        --frame_idx 50 \
        --stats_file Dexora/dexmg/configs/dataset_statistics.json
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np

from dexmg.dexmg_config import DATASET_CONFIGS, get_dataset_config
from dexmg.dexmg_convert import build_unified_action
from dexmg.dexmg_hdf5_vla_dataset import DexmgHDF5VLADataset
from dexmg.dexmg_schema import build_schema


def _extract_right_gripper_from_unified(unified: np.ndarray, schema, group: str) -> np.ndarray:
    slot = schema.slots["right_gripper"]
    w = int(schema.action_real_width[group]["right_gripper"])
    return unified[..., slot.offset: slot.offset + w].reshape(-1)


def _find_reader_for_hdf5(dataset, target_hdf5: str):
    target = os.path.abspath(target_hdf5)
    for reader in dataset._readers:
        if os.path.abspath(reader.hdf5_path) == target:
            return reader
    raise RuntimeError(f"未在 dataset._readers 中找到 {target}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="dexmimicgen 数据集根目录，例如 /data/dexmg")
    parser.add_argument("--hdf5", type=str, required=True,
                        help="需要核对的单个 hdf5 文件，例如 .../two_arm_box_cleanup.hdf5")
    parser.add_argument("--demo_id", type=str, default="demo_1006")
    parser.add_argument("--frame_idx", type=int, default=50)
    parser.add_argument("--stats_file", type=str, default="Dexora/dexmg/configs/dataset_statistics.json",
                        help="dataset_statistics.json 路径；只要训练时加载的同一份即可")
    parser.add_argument("--schema_cache_dir", type=str, default=None,
                        help="schema cache 目录，默认与 dataset_root 一致")
    parser.add_argument("--seq_length", type=int, default=1,
                        help="将训练读取设为单帧序列，保证 item['actions'] 正好对应目标帧")
    args = parser.parse_args()

    hdf5_path = os.path.abspath(args.hdf5)
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"找不到 hdf5: {hdf5_path}")
    if not os.path.exists(args.stats_file):
        raise FileNotFoundError(f"找不到 stats_file: {args.stats_file}")

    cfg = get_dataset_config(hdf5_path)
    schema = build_schema(dataset_root=args.dataset_root, cache_dir=(args.schema_cache_dir or args.dataset_root))

    # 1) 直接走训练数据集的读取路径：DexmgHDF5VLADataset.get_item(...)
    dataset = DexmgHDF5VLADataset(
        dataset_root=args.dataset_root,
        stats_file=args.stats_file,
        seq_length=args.seq_length,
        frame_stack=1,
        schema_cache_dir=(args.schema_cache_dir or args.dataset_root),
    )
    reader = _find_reader_for_hdf5(dataset, hdf5_path)
    demo_start = int(reader._seq_ds._demo_id_to_start_indices[args.demo_id])
    sample_idx = demo_start + args.frame_idx
    item = reader.get_item(sample_idx)

    # 训练实际使用的原始目标（归一化前）
    train_raw = item["actions"][0]
    train_right = _extract_right_gripper_from_unified(train_raw, schema, cfg["embodiment_group"])

    # 2) 直接从 hdf5 读 action_dict，并用 build_unified_action() 重算
    with h5py.File(hdf5_path, "r") as f:
        action_dict = {
            key: np.asarray(f[f"data/{args.demo_id}/action_dict/{key}"][args.frame_idx])
            for key in cfg["action_keys"]
        }
        unified_action, _ = build_unified_action(action_dict, cfg, schema)
        h5_right = _extract_right_gripper_from_unified(unified_action, schema, cfg["embodiment_group"])

    # 3) 对比
    diff = np.abs(train_right - h5_right)
    max_abs_diff = float(np.max(diff)) if diff.size else 0.0
    allclose = bool(np.allclose(train_right, h5_right, rtol=0.0, atol=1e-8))

    print(f"demo_id={args.demo_id} frame_idx={args.frame_idx}")
    print(f"dataset_name={cfg['dataset_name']} embodiment_group={cfg['embodiment_group']}")
    print(f"dataset sample index={sample_idx}")
    print(f"right_gripper width={len(train_right)}")
    print()

    # 打印每个 action key 的原始值与统一转换后对应片段，便于直接定位字段偏差
    print("[raw action_dict per key]")
    for key in cfg["action_keys"]:
        val = np.asarray(action_dict[key]).reshape(-1)
        print(f"  {key}: shape={val.shape} values={np.array2string(val, precision=8, suppress_small=False)}")
    print()

    print("[dataset __getitem__ / item['actions']] raw right_gripper (pre-norm):")
    print(np.array2string(train_right, precision=8, suppress_small=False))
    print()
    print("[h5py + build_unified_action] raw right_gripper (pre-norm):")
    print(np.array2string(h5_right, precision=8, suppress_small=False))
    print()

    print("[逐维 diff 表: dim | train | h5 | abs_diff]")
    for i in range(len(train_right)):
        print(f"  {i:02d}: {train_right[i]: .8f} | {h5_right[i]: .8f} | {diff[i]: .8e}")
    print()

    print(f"max_abs_diff={max_abs_diff:.12e}")
    print(f"allclose={allclose}")

    if not allclose:
        print("\n差异位置：")
        idx = int(np.argmax(diff))
        print(f"argmax diff at dim={idx}, train={train_right[idx]}, h5={h5_right[idx]}, diff={diff[idx]}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()



