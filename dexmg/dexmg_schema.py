# -*- coding: utf-8 -*-
"""
dexmg_schema.py (第二次重写 —— state 和 action 的 gripper 宽度都探测，不假设)

上一版的 bug：action 侧的 right_gripper/left_gripper 宽度硬编码成 6
（从 panda 组的数据打印抄的），humanoid 组实际宽度不一样，直接崩了。

教训：只要是"某个 embodiment 的某个 key 到底几维"这种问题，一律不
猜、不抄别处打印出来的数字，统一走"探测 hdf5 实际 shape"这条路——
state 侧的 gripper_qpos 之前就是这么做的，这版把 action 侧的
right_gripper/left_gripper 也纳入同一套探测机制。

槽位（state 和 action 现在共用同一份 Schema 对象，一次探测、一次缓存）：
    right_arm_pos     (3，固定)
    right_arm_rot6d   (6，固定，两侧旋转统一转成6D后写入)
    right_gripper     (探测得到，= 两组里较大的真实宽度)
    left_arm_pos      (3，固定)
    left_arm_rot6d    (6，固定)
    left_gripper       (探测得到)

state 和 action 各自维护一份独立的宽度探测结果（同名槽位，但
state 的 gripper_qpos 宽度和 action 的 gripper 控制量宽度不是同一
个东西，不能混用同一个探测结果）。
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List

import h5py
import numpy as np

from dexmg_config import DATASET_CONFIGS, list_hdf5_by_group

_SCHEMA_CACHE_FILENAME = "dexmg_unified_schema_cache.json"

GROUPS = ("panda", "humanoid")
SLOT_NAMES_FIXED = [  # (name, dim) —— 这几个宽度是设计上固定的，不用探测
    ("right_arm_pos", 3), ("right_arm_rot6d", 6),
    ("left_arm_pos", 3), ("left_arm_rot6d", 6),
]
GRIPPER_SLOT_NAMES = ["right_gripper", "left_gripper"]


@dataclass(frozen=True)
class SharedSlot:
    name: str
    offset: int
    dim: int


@dataclass
class SubSchema:
    """state 或 action 各自的一份槽位表 + gripper 真实宽度记录。"""
    slots: "OrderedDict[str, SharedSlot]"
    dim: int
    # {group: {"right_gripper": real_width, "left_gripper": real_width}}
    group_gripper_real_width: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def group_mask(self, group: str) -> np.ndarray:
        mask = np.ones(self.dim, dtype=np.float32)
        for gripper_name in GRIPPER_SLOT_NAMES:
            slot = self.slots[gripper_name]
            real_w = self.group_gripper_real_width[group][gripper_name]
            if real_w < slot.dim:
                mask[slot.offset + real_w: slot.offset + slot.dim] = 0.0
        return mask


@dataclass
class Schema:
    state: SubSchema
    action: SubSchema


def _low_dim_key_roles(cfg) -> Dict[str, Dict[str, str]]:
    """返回 {"right": {"pos": key, "quat": key, "gripper": key}, "left": {...}}"""
    keys = cfg["low_dim_keys"]
    roles: Dict[str, Dict[str, str]] = {"right": {}, "left": {}}
    for k in keys:
        if "gripper_qpos" in k:
            side = "left" if "left" in k or k.startswith("robot1") else "right"
            roles[side]["gripper"] = k
        elif "eef_quat" in k:
            side = "left" if "left" in k or k.startswith("robot1") else "right"
            roles[side]["quat"] = k
        elif "eef_pos" in k:
            side = "left" if "left" in k or k.startswith("robot1") else "right"
            roles[side]["pos"] = k
    for side in ("right", "left"):
        assert set(roles[side].keys()) == {"pos", "quat", "gripper"}, (
            f"low_dim_keys 里 {side} 侧缺字段，解析出来的是 {roles[side]}，"
            f"原始 low_dim_keys={keys}"
        )
    return roles


def _probe_obs_width(h5file: h5py.File, demo0: str, key: str) -> int:
    shape = h5file[f"data/{demo0}/obs/{key}"].shape
    return int(shape[1]) if len(shape) > 1 else 1


def _probe_action_dict_width(h5file: h5py.File, demo0: str, key: str) -> int:
    """action 分量存在 data/{demo}/action_dict/{key} 下，直接读真实 shape。"""
    ds_path = f"data/{demo0}/action_dict/{key}"
    assert ds_path in h5file, (
        f"{ds_path} 不存在，检查这个 hdf5 的 action_dict 分组结构是否和预期一致"
    )
    shape = h5file[ds_path].shape
    return int(shape[1]) if len(shape) > 1 else 1


def _build_sub_schema(
    dataset_root: str,
    fixed_names: List[tuple],
    probe_gripper_width_fn,
) -> SubSchema:
    group_gripper_width: Dict[str, Dict[str, int]] = {}
    for group in GROUPS:
        hdf5_names = list_hdf5_by_group(group)
        cfg0 = DATASET_CONFIGS[hdf5_names[0]]
        probe_path = os.path.join(dataset_root, hdf5_names[0])
        with h5py.File(probe_path, "r") as f:
            demo0 = next(iter(f["data"].keys()))
            widths = probe_gripper_width_fn(f, demo0, cfg0)
        group_gripper_width[group] = widths

    right_gripper_dim = max(w["right_gripper"] for w in group_gripper_width.values())
    left_gripper_dim = max(w["left_gripper"] for w in group_gripper_width.values())

    slots: "OrderedDict[str, SharedSlot]" = OrderedDict()
    offset = 0
    ordered = [
        ("right_arm_pos", 3), ("right_arm_rot6d", 6), ("right_gripper", right_gripper_dim),
        ("left_arm_pos", 3), ("left_arm_rot6d", 6), ("left_gripper", left_gripper_dim),
    ]
    for name, dim in ordered:
        slots[name] = SharedSlot(name, offset, dim)
        offset += dim

    return SubSchema(slots=slots, dim=offset, group_gripper_real_width=group_gripper_width)


def _state_gripper_probe(f: h5py.File, demo0: str, cfg) -> Dict[str, int]:
    roles = _low_dim_key_roles(cfg)
    return {
        "right_gripper": _probe_obs_width(f, demo0, roles["right"]["gripper"]),
        "left_gripper": _probe_obs_width(f, demo0, roles["left"]["gripper"]),
    }


def _action_gripper_probe(f: h5py.File, demo0: str, cfg) -> Dict[str, int]:
    return {
        "right_gripper": _probe_action_dict_width(f, demo0, "right_gripper"),
        "left_gripper": _probe_action_dict_width(f, demo0, "left_gripper"),
    }


def build_schema(dataset_root: str, cache_dir: str, force_recompute: bool = False) -> Schema:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, _SCHEMA_CACHE_FILENAME)

    if os.path.exists(cache_path) and not force_recompute:
        with open(cache_path, "r") as f:
            cached = json.load(f)

        def _load(sub):
            slots = OrderedDict((n, SharedSlot(**s)) for n, s in sub["slots"].items())
            return SubSchema(slots=slots, dim=sub["dim"],
                              group_gripper_real_width=sub["group_gripper_real_width"])

        return Schema(state=_load(cached["state"]), action=_load(cached["action"]))

    state_schema = _build_sub_schema(dataset_root, SLOT_NAMES_FIXED, _state_gripper_probe)
    action_schema = _build_sub_schema(dataset_root, SLOT_NAMES_FIXED, _action_gripper_probe)
    schema = Schema(state=state_schema, action=action_schema)

    def _dump(sub: SubSchema):
        return {
            "dim": sub.dim,
            "slots": {n: s.__dict__ for n, s in sub.slots.items()},
            "group_gripper_real_width": sub.group_gripper_real_width,
        }

    with open(cache_path, "w") as f:
        json.dump({"state": _dump(state_schema), "action": _dump(action_schema)}, f, indent=2)
    return schema
