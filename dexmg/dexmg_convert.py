# -*- coding: utf-8 -*-
"""
dexmg_convert.py (第二次重写 —— gripper 宽度按探测结果放置，不假设6维)

action_dict/obs_dict 传进来的每个分量，直接按它自己的真实宽度写进
槽位前段；槽位宽度 = 两组里较大的真实宽度（来自 dexmg_schema.py 的
探测结果），narrower 的那组自动 padding 补0，mask 里对应位置为0。
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from dexmg_config import DatasetConfig
from dexmg_rotation import axis_angle_to_rot6d, quat_to_rot6d
from dexmg_schema import Schema, SubSchema, _low_dim_key_roles


def _place_into(unified: np.ndarray, sub_schema: SubSchema, name: str, value: np.ndarray):
    slot = sub_schema.slots[name]
    w = value.shape[-1]
    assert w <= slot.dim, (
        f"{name}: 槽位宽度{slot.dim} < 实际数据宽度{w}，探测缓存可能过期，"
        f"重新跑一次 build_schema(force_recompute=True)"
    )
    unified[..., slot.offset: slot.offset + w] = value


def build_unified_action(
    action_dict: Dict[str, np.ndarray], cfg: DatasetConfig, schema: Schema
) -> Tuple[np.ndarray, np.ndarray]:
    """
    action_dict: {"right_gripper": (...,W), "right_rel_pos": (...,3), ...}
                 每个分量的宽度就是它在 hdf5 action_dict 里的真实宽度，
                 不用再猜/查静态表。
    """
    group = cfg["embodiment_group"]
    action_schema = schema.action

    if group == "panda":
        right_pos = np.asarray(action_dict["right_rel_pos"])
        right_rot6d = axis_angle_to_rot6d(action_dict["right_rel_rot_axis_angle"])
        left_pos = np.asarray(action_dict["left_rel_pos"])
        left_rot6d = axis_angle_to_rot6d(action_dict["left_rel_rot_axis_angle"])
    elif group == "humanoid":
        right_pos = np.asarray(action_dict["right_abs_pos"])
        right_rot6d = np.asarray(action_dict["right_abs_rot_6d"])
        left_pos = np.asarray(action_dict["left_abs_pos"])
        left_rot6d = np.asarray(action_dict["left_abs_rot_6d"])
    else:
        raise ValueError(f"未知 embodiment_group: {group}")

    right_gripper = np.asarray(action_dict["right_gripper"])  # 真实宽度，可能不是6
    left_gripper = np.asarray(action_dict["left_gripper"])

    lead_shape = right_pos.shape[:-1]
    unified = np.zeros((*lead_shape, action_schema.dim), dtype=np.float32)

    _place_into(unified, action_schema, "right_arm_pos", right_pos)
    _place_into(unified, action_schema, "right_arm_rot6d", right_rot6d)
    _place_into(unified, action_schema, "right_gripper", right_gripper)
    _place_into(unified, action_schema, "left_arm_pos", left_pos)
    _place_into(unified, action_schema, "left_arm_rot6d", left_rot6d)
    _place_into(unified, action_schema, "left_gripper", left_gripper)

    mask = action_schema.group_mask(group)
    return unified, mask


def build_unified_state(
    obs_dict: Dict[str, np.ndarray], cfg: DatasetConfig, schema: Schema
) -> Tuple[np.ndarray, np.ndarray]:
    group = cfg["embodiment_group"]
    state_schema = schema.state
    roles = _low_dim_key_roles(cfg)

    right_pos = np.asarray(obs_dict[roles["right"]["pos"]])
    right_rot6d = quat_to_rot6d(obs_dict[roles["right"]["quat"]])
    right_gripper = np.asarray(obs_dict[roles["right"]["gripper"]])
    left_pos = np.asarray(obs_dict[roles["left"]["pos"]])
    left_rot6d = quat_to_rot6d(obs_dict[roles["left"]["quat"]])
    left_gripper = np.asarray(obs_dict[roles["left"]["gripper"]])

    lead_shape = right_pos.shape[:-1]
    unified = np.zeros((*lead_shape, state_schema.dim), dtype=np.float32)

    _place_into(unified, state_schema, "right_arm_pos", right_pos)
    _place_into(unified, state_schema, "right_arm_rot6d", right_rot6d)
    _place_into(unified, state_schema, "right_gripper", right_gripper)
    _place_into(unified, state_schema, "left_arm_pos", left_pos)
    _place_into(unified, state_schema, "left_arm_rot6d", left_rot6d)
    _place_into(unified, state_schema, "left_gripper", left_gripper)

    mask = state_schema.group_mask(group)
    return unified, mask
