# -*- coding: utf-8 -*-
"""
dexmg_schema.py

方案 B：不做 abs<->rel 表示转换，两个 embodiment group 的原始语义
各自保留独立的槽位（slot），拼进一个更大的统一 (superset) 维度 M。
每条样本只在自己 group 对应的槽位上写真实值，其余槽位补零，并且
用 mask 标记哪些槽位对这条样本有效。

action superset 布局（固定、可静态算出，不依赖读数据）：

    偏移  长度  含义                          来源 group
    0     6    right_arm  (pos3 + axis_angle3)     panda
    6     6    right_gripper                        panda
    12    6    left_arm   (pos3 + axis_angle3)      panda
    18    6    left_gripper                         panda
    24    9    right_arm  (pos3 + rot6d6)           humanoid
    33    9    left_arm   (pos3 + rot6d6)           humanoid
    42    6    right_gripper                        humanoid
    48    6    left_gripper                         humanoid
    ------------------------------------------------------
    ACTION_DIM = 54

state superset 布局：低维 key 的实际宽度（尤其 gripper_qpos）在不同
夹爪/灵巧手上不一定一样，这里不硬编码数字，而是在
`build_state_schema()` 里现场探测一遍 hdf5 拿到真实宽度，探测结果
缓存进一个 json，避免每次都重新开 hdf5。
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import h5py
import numpy as np

from dexmg_config import DATASET_CONFIGS, list_hdf5_by_group


# ============================================================
# Action superset schema —— 固定值，可以直接算出来，不依赖数据
# ============================================================

@dataclass(frozen=True)
class ActionSlot:
    name: str
    group: str  # "panda" | "humanoid"
    offset: int
    dim: int


ACTION_SLOTS: List[ActionSlot] = [
    ActionSlot("right_arm_rel", "panda", 0, 6),
    ActionSlot("right_gripper_rel", "panda", 6, 6),
    ActionSlot("left_arm_rel", "panda", 12, 6),
    ActionSlot("left_gripper_rel", "panda", 18, 6),
    ActionSlot("right_arm_abs", "humanoid", 24, 9),
    ActionSlot("left_arm_abs", "humanoid", 33, 9),
    ActionSlot("right_gripper_abs", "humanoid", 42, 6),
    ActionSlot("left_gripper_abs", "humanoid", 48, 6),
]
ACTION_DIM = sum(s.dim for s in ACTION_SLOTS)  # 54
assert ACTION_DIM == 54

ACTION_SLOTS_BY_GROUP: Dict[str, List[ActionSlot]] = {
    "panda": [s for s in ACTION_SLOTS if s.group == "panda"],
    "humanoid": [s for s in ACTION_SLOTS if s.group == "humanoid"],
}


def action_group_mask(group: str) -> np.ndarray:
    """返回长度 ACTION_DIM 的 0/1 mask：该 group 的槽位为 1，其余为 0。"""
    mask = np.zeros(ACTION_DIM, dtype=np.float32)
    for slot in ACTION_SLOTS_BY_GROUP[group]:
        mask[slot.offset: slot.offset + slot.dim] = 1.0
    return mask


# ============================================================
# State superset schema —— 宽度依赖实际 hdf5 数据，需要探测
# ============================================================

_SCHEMA_CACHE_FILENAME = "dexmg_state_schema_cache.json"


def _probe_low_dim_widths(hdf5_path: str, low_dim_keys: List[str]) -> Dict[str, int]:
    """打开一个 hdf5，读第一条 demo 的第一帧，拿到每个 low_dim key 的真实宽度。"""
    widths: Dict[str, int] = {}
    with h5py.File(hdf5_path, "r") as f:
        demo_keys = list(f["data"].keys())
        assert len(demo_keys) > 0, f"{hdf5_path} 里没有找到任何 demo"
        demo0 = demo_keys[0]
        for key in low_dim_keys:
            ds_path = f"data/{demo0}/obs/{key}"
            assert ds_path in f, f"{hdf5_path} 缺少 obs key: {key}"
            shape = f[ds_path].shape
            # shape = (T, D) 或 (T,) -> D=1
            widths[key] = int(shape[1]) if len(shape) > 1 else 1
    return widths


def build_state_schema(
    dataset_root: str,
    cache_dir: str,
    force_recompute: bool = False,
) -> Tuple[Dict[str, "StateSlot"], int]:
    """
    探测 panda / humanoid 两个 group 的 state 宽度，构造 state superset schema。
    每个 group 内部按 low_dim_keys 的固定顺序把各 key 的真实宽度拼起来，
    作为该 group 在 superset 里的一段连续槽位。

    结果缓存进 `cache_dir/dexmg_state_schema_cache.json`，
    第二次调用直接读缓存，不用重新打开 hdf5（除非 force_recompute=True）。
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, _SCHEMA_CACHE_FILENAME)

    if os.path.exists(cache_path) and not force_recompute:
        with open(cache_path, "r") as f:
            cached = json.load(f)
        slots = {
            name: StateSlot(**s) for name, s in cached["slots"].items()
        }
        return slots, cached["state_dim"]

    slots: "OrderedDict[str, StateSlot]" = OrderedDict()
    offset = 0
    for group in ("panda", "humanoid"):
        hdf5_names = list_hdf5_by_group(group)
        assert len(hdf5_names) > 0, f"group={group} 没有配置任何数据集"
        # 同一个 group 内所有数据集的 low_dim_keys 理论上应完全一致，
        # 用第一个数据集探测即可；这里额外做一次一致性检查。
        cfg0 = DATASET_CONFIGS[hdf5_names[0]]
        probe_path = os.path.join(dataset_root, hdf5_names[0])
        widths = _probe_low_dim_widths(probe_path, cfg0["low_dim_keys"])

        for other_name in hdf5_names[1:]:
            cfg_i = DATASET_CONFIGS[other_name]
            assert cfg_i["low_dim_keys"] == cfg0["low_dim_keys"], (
                f"{other_name} 的 low_dim_keys 和同 group 的 {hdf5_names[0]} 不一致，"
                f"需要人工核实是否真的属于同一个 group"
            )

        group_dim = sum(widths[k] for k in cfg0["low_dim_keys"])
        slots[f"{group}_state"] = StateSlot(
            group=group, offset=offset, dim=group_dim,
            key_order=list(cfg0["low_dim_keys"]),
            key_widths={k: widths[k] for k in cfg0["low_dim_keys"]},
        )
        offset += group_dim

    state_dim = offset
    with open(cache_path, "w") as f:
        json.dump(
            {
                "state_dim": state_dim,
                "slots": {name: s.__dict__ for name, s in slots.items()},
            },
            f,
            indent=2,
        )
    return dict(slots), state_dim


@dataclass
class StateSlot:
    group: str
    offset: int
    dim: int
    key_order: List[str]
    key_widths: Dict[str, int]


def state_group_mask(state_slots: Dict[str, StateSlot], state_dim: int, group: str) -> np.ndarray:
    mask = np.zeros(state_dim, dtype=np.float32)
    slot = state_slots[f"{group}_state"]
    mask[slot.offset: slot.offset + slot.dim] = 1.0
    return mask
