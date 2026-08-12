# -*- coding: utf-8 -*-
"""
dexmg_schema.py (重写版 —— 共享物理槽位，而不是按 group 各占一段)

设计原则（对齐 RDT-1B 论文的 unified action space 思路）：
    槽位按"物理含义"分配，不按"来自哪个 embodiment group"分配。
    panda 组和 humanoid 组都是双臂+双爪，物理结构完全一样，只是
    控制模式不同（相对位姿+axis-angle vs 绝对位姿+rot_6d）——这种
    差异属于"数值语义"差异，不是"自由度种类"差异，所以不需要开
    两套槽位，而是：
        1. 旋转统一存成 6D（axis-angle -> 6D 是无损格式转换，不是
           有信息损失的物理量转换，两组共用同一个 6D 槽位没问题）
        2. rel vs abs 这个语义差异本身不占用额外维度，靠"每个数据集
           有自己独立的归一化统计量" + RDT 的 dataset/instruction
           conditioning 让模型隐式区分，不做显式转换（不像方案A那样
           把 abs 转成 rel，避免引入需要额外验证正确性的有损转换）

action 槽位（6个，都是"两组都有"的物理量，天然不需要 mask）：
    right_arm_pos    (3)   两组都是 right_arm 位置增量或绝对位置
    right_arm_rot6d  (6)   两组都转成 6D（panda 从 axis_angle 转，humanoid 原生就是）
    right_gripper    (6)   两组维度恰好都是 6，直接共用
    left_arm_pos     (3)
    left_arm_rot6d   (6)
    left_gripper     (6)
    ------------------------------------------------
    ACTION_DIM = 30   （比方案B的54维小，且两组共享，不是各占一段）

state 槽位结构一样，只是 gripper 部分的真实宽度需要从 hdf5 探测
（不同夹爪/灵巧手 qpos 维度可能不同），取两组里较大的宽度作为槽位
宽度，宽度不足的那组padding补零+mask。arm_pos/arm_rot6d 两组都是
满的，不需要 mask。

如果未来接入的新 embodiment 缺胳膊少腿（比如单臂机器人没有
left_arm），mask 机制仍然保留、依旧有意义——只是当前这两组数据
碰巧都是完整双臂双爪，所以 action 侧 mask 目前恒为全 1。
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import h5py
import numpy as np

from dexmg_config import DATASET_CONFIGS, list_hdf5_by_group

_SCHEMA_CACHE_FILENAME = "dexmg_unified_schema_cache.json"


# ============================================================
# 共享槽位定义
# ============================================================

@dataclass(frozen=True)
class SharedSlot:
    name: str
    offset: int
    dim: int  # 这是槽位在 unified 向量里的宽度（= 两组里较大的真实宽度）


# ---- action：宽度固定已知（两组的 pos/rot6d/gripper 维度天然一致） ----
ACTION_SLOTS: List[SharedSlot] = [
    SharedSlot("right_arm_pos", 0, 3),
    SharedSlot("right_arm_rot6d", 3, 6),
    SharedSlot("right_gripper", 9, 6),
    SharedSlot("left_arm_pos", 15, 3),
    SharedSlot("left_arm_rot6d", 18, 6),
    SharedSlot("left_gripper", 24, 6),
]
ACTION_DIM = sum(s.dim for s in ACTION_SLOTS)
assert ACTION_DIM == 30

ACTION_SLOT_BY_NAME: Dict[str, SharedSlot] = {s.name: s for s in ACTION_SLOTS}


def action_group_mask(group: str) -> np.ndarray:
    """当前两组（panda/humanoid）action 侧都是满槽位，恒为全 1。

    保留这个函数（而不是直接删掉、处处用 np.ones）是为了以后接入
    缺自由度的新 embodiment（比如单臂）时，只需要在这里改，不用动
    convert.py / dataset.py 的调用方式。
    """
    return np.ones(ACTION_DIM, dtype=np.float32)


# ---- state：gripper 部分宽度需要探测，pos/rot6d 宽度固定 ----

@dataclass
class UnifiedStateSchema:
    slots: "OrderedDict[str, SharedSlot]"
    state_dim: int
    # 每个 group 在 gripper 槽位上的真实宽度（可能小于槽位宽度，其余是padding）
    # 结构: {group: {"right_gripper": real_width, "left_gripper": real_width}}
    group_gripper_real_width: Dict[str, Dict[str, int]] = field(default_factory=dict)


def _probe_key_width(hdf5_path: str, key: str) -> int:
    with h5py.File(hdf5_path, "r") as f:
        demo0 = next(iter(f["data"].keys()))
        shape = f[f"data/{demo0}/obs/{key}"].shape
        return int(shape[1]) if len(shape) > 1 else 1


# 每个数据集 low_dim_keys 里，哪个 key 对应 pos/quat/gripper_qpos
# (key 名字本身已经暗示了顺序：xxx_eef_pos, xxx_eef_quat, xxx_gripper_qpos)
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
            f"原始 low_dim_keys={keys}，需要检查 dexmg_config.py 里的命名"
        )
    return roles


def build_state_schema(
    dataset_root: str,
    cache_dir: str,
    force_recompute: bool = False,
) -> UnifiedStateSchema:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, _SCHEMA_CACHE_FILENAME)

    if os.path.exists(cache_path) and not force_recompute:
        with open(cache_path, "r") as f:
            cached = json.load(f)
        slots = OrderedDict(
            (name, SharedSlot(**s)) for name, s in cached["slots"].items()
        )
        return UnifiedStateSchema(
            slots=slots,
            state_dim=cached["state_dim"],
            group_gripper_real_width=cached["group_gripper_real_width"],
        )

    # 1) 探测每个 group 的 right/left gripper_qpos 真实宽度
    group_gripper_width: Dict[str, Dict[str, int]] = {}
    for group in ("panda", "humanoid"):
        hdf5_names = list_hdf5_by_group(group)
        cfg0 = DATASET_CONFIGS[hdf5_names[0]]
        probe_path = os.path.join(dataset_root, hdf5_names[0])
        roles = _low_dim_key_roles(cfg0)
        group_gripper_width[group] = {
            "right_gripper": _probe_key_width(probe_path, roles["right"]["gripper"]),
            "left_gripper": _probe_key_width(probe_path, roles["left"]["gripper"]),
        }

    right_gripper_dim = max(w["right_gripper"] for w in group_gripper_width.values())
    left_gripper_dim = max(w["left_gripper"] for w in group_gripper_width.values())

    # 2) 组装 state 槽位（pos=3, rot6d=6 固定；gripper 用探测出的最大宽度）
    slots: "OrderedDict[str, SharedSlot]" = OrderedDict()
    offset = 0
    for name, dim in [
        ("right_arm_pos", 3), ("right_arm_rot6d", 6), ("right_gripper", right_gripper_dim),
        ("left_arm_pos", 3), ("left_arm_rot6d", 6), ("left_gripper", left_gripper_dim),
    ]:
        slots[name] = SharedSlot(name, offset, dim)
        offset += dim
    state_dim = offset

    schema = UnifiedStateSchema(
        slots=slots, state_dim=state_dim, group_gripper_real_width=group_gripper_width,
    )
    with open(cache_path, "w") as f:
        json.dump(
            {
                "state_dim": state_dim,
                "slots": {n: s.__dict__ for n, s in slots.items()},
                "group_gripper_real_width": group_gripper_width,
            },
            f, indent=2,
        )
    return schema


def state_group_mask(schema: UnifiedStateSchema, group: str) -> np.ndarray:
    """pos/rot6d 槽位两组都是满的（mask=1）；gripper 槽位按该组真实宽度置 1，padding 部分置 0。"""
    mask = np.ones(schema.state_dim, dtype=np.float32)
    for side in ("right", "left"):
        slot = schema.slots[f"{side}_gripper"]
        real_w = schema.group_gripper_real_width[group][f"{side}_gripper"]
        if real_w < slot.dim:
            mask[slot.offset + real_w: slot.offset + slot.dim] = 0.0
    return mask
