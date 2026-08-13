# -*- coding: utf-8 -*-
"""
dexmg_schema.py (第三次重写 —— state/action 强制共用同一个 M)

关键约束（来自 Dexora 实际训练代码）：
    RDTRunner(action_dim=config["common"]["state_dim"], ...)
即 state_dim 和 action_dim 天生是同一个数字 M，不是两个独立维度。
之前的版本让 state(42) / action(30) 各自探测、各自不同，跟这个约束
不兼容。

这版把 gripper 槽位宽度改成"state 和 action、panda 和 humanoid，一共
4 个探测结果里取最大值"，state 和 action 强制使用同一套 slots 布局
（同样的 offset/dim）。已知的 4 个探测结果：
    state  panda    right/left gripper: 12
    state  humanoid right/left gripper: 11 (GR1/fourier hand raw qpos)
    action panda    right/left gripper: 6
    action humanoid right/left gripper: 6
取 max -> gripper 槽位宽度 = 12，M = 3+6+12+3+6+12 = 42
（正好等于之前 state 侧探测出来的 42，因为 state 侧本来就比 action
宽；action 侧那 6 维会 padding 到 12，多出来的 6 维 mask=0）。
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
GRIPPER_SLOT_NAMES = ["right_gripper", "left_gripper"]


@dataclass(frozen=True)
class SharedSlot:
    name: str
    offset: int
    dim: int


@dataclass
class Schema:
    dim: int  # M —— state 和 action 共用同一个数字
    slots: "OrderedDict[str, SharedSlot]"
    # 分别记录 state / action 在每个 group 下 gripper 槽位的真实宽度
    # （槽位宽度是共享的，但实际数据宽度不同：state 是 11/12，action 是 6/6）
    state_real_width: Dict[str, Dict[str, int]] = field(default_factory=dict)
    action_real_width: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def _group_mask(self, real_width_table: Dict[str, Dict[str, int]], group: str) -> np.ndarray:
        mask = np.ones(self.dim, dtype=np.float32)
        for gripper_name in GRIPPER_SLOT_NAMES:
            slot = self.slots[gripper_name]
            real_w = real_width_table[group][gripper_name]
            if real_w < slot.dim:
                mask[slot.offset + real_w: slot.offset + slot.dim] = 0.0
        return mask

    def state_group_mask(self, group: str) -> np.ndarray:
        return self._group_mask(self.state_real_width, group)

    def action_group_mask(self, group: str) -> np.ndarray:
        return self._group_mask(self.action_real_width, group)


def _low_dim_key_roles(cfg) -> Dict[str, Dict[str, str]]:
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
    ds_path = f"data/{demo0}/action_dict/{key}"
    assert ds_path in h5file, f"{ds_path} 不存在，检查 action_dict 分组结构"
    shape = h5file[ds_path].shape
    return int(shape[1]) if len(shape) > 1 else 1


def _probe_group_widths(dataset_root: str) -> Dict[str, Dict[str, Dict[str, int]]]:
    """返回 {"state": {group: {"right_gripper": w, "left_gripper": w}}, "action": {...}}"""
    out = {"state": {}, "action": {}}
    for group in GROUPS:
        hdf5_names = list_hdf5_by_group(group)
        cfg0 = DATASET_CONFIGS[hdf5_names[0]]
        probe_path = os.path.join(dataset_root, hdf5_names[0])
        roles = _low_dim_key_roles(cfg0)
        with h5py.File(probe_path, "r") as f:
            demo0 = next(iter(f["data"].keys()))
            out["state"][group] = {
                "right_gripper": _probe_obs_width(f, demo0, roles["right"]["gripper"]),
                "left_gripper": _probe_obs_width(f, demo0, roles["left"]["gripper"]),
            }
            out["action"][group] = {
                "right_gripper": _probe_action_dict_width(f, demo0, "right_gripper"),
                "left_gripper": _probe_action_dict_width(f, demo0, "left_gripper"),
            }
    return out


def build_schema(dataset_root: str, cache_dir: str, force_recompute: bool = False) -> Schema:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, _SCHEMA_CACHE_FILENAME)

    if os.path.exists(cache_path) and not force_recompute:
        with open(cache_path, "r") as f:
            cached = json.load(f)
        slots = OrderedDict((n, SharedSlot(**s)) for n, s in cached["slots"].items())
        return Schema(
            dim=cached["dim"], slots=slots,
            state_real_width=cached["state_real_width"],
            action_real_width=cached["action_real_width"],
        )

    widths = _probe_group_widths(dataset_root)  # {"state":{...}, "action":{...}}

    # 每一侧(right/left)的槽位宽度 = state/action、panda/humanoid 四个探测值里的最大值
    def _max_over_all(gripper_name: str) -> int:
        candidates = [
            widths["state"]["panda"][gripper_name],
            widths["state"]["humanoid"][gripper_name],
            widths["action"]["panda"][gripper_name],
            widths["action"]["humanoid"][gripper_name],
        ]
        return max(candidates)

    right_gripper_dim = _max_over_all("right_gripper")
    left_gripper_dim = _max_over_all("left_gripper")

    slots: "OrderedDict[str, SharedSlot]" = OrderedDict()
    offset = 0
    for name, dim in [
        ("right_arm_pos", 3), ("right_arm_rot6d", 6), ("right_gripper", right_gripper_dim),
        ("left_arm_pos", 3), ("left_arm_rot6d", 6), ("left_gripper", left_gripper_dim),
    ]:
        slots[name] = SharedSlot(name, offset, dim)
        offset += dim

    schema = Schema(
        dim=offset, slots=slots,
        state_real_width=widths["state"], action_real_width=widths["action"],
    )

    with open(cache_path, "w") as f:
        json.dump(
            {
                "dim": schema.dim,
                "slots": {n: s.__dict__ for n, s in slots.items()},
                "state_real_width": widths["state"],
                "action_real_width": widths["action"],
            },
            f, indent=2,
        )
    return schema
