# -*- coding: utf-8 -*-
"""
dexmg_convert.py (重写版 —— 写入共享物理槽位)

两组的 action_dict/obs_dict 字段名不同，但物理含义一一对应：

    槽位            panda组来源                          humanoid组来源
    right_arm_pos   right_rel_pos (3)                    right_abs_pos (3)
    right_arm_rot6d axis_angle_to_rot6d(right_rel_rot_axis_angle)   right_abs_rot_6d (已是6D，直接用)
    right_gripper   right_gripper (6)                     right_gripper (6)
    left_*          同理

    state 的 pos/quat 同理：quat 統一转成 rot6d
    (quat_to_rot6d)，gripper_qpos 按各自真实宽度写入槽位前段，
    其余(padding)部分保持0，对应 mask 里已经标为0。

不做 rel<->abs 转换：right_arm_pos 槽位里 panda 组存的是"相对位移"，
humanoid 组存的是"绝对位置"，数值语义不同，但物理意义（右臂位置这个
自由度）相同，靠各自独立的归一化统计量 + 数据集条件让模型学会区分。
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from dexmg_config import DatasetConfig
from dexmg_rotation import axis_angle_to_rot6d, quat_to_rot6d
from dexmg_schema import ACTION_DIM, ACTION_SLOT_BY_NAME, UnifiedStateSchema, action_group_mask, state_group_mask


def build_unified_action(action_dict: Dict[str, np.ndarray], cfg: DatasetConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    action_dict: robomimic 读出的原始 action_dict（key 为 cfg["action_keys"]）
    返回 (unified_action[..., ACTION_DIM], mask[ACTION_DIM])
    """
    group = cfg["embodiment_group"]

    if group == "panda":
        right_pos = np.asarray(action_dict["right_rel_pos"])
        right_rot6d = axis_angle_to_rot6d(action_dict["right_rel_rot_axis_angle"])
        right_gripper = np.asarray(action_dict["right_gripper"])
        left_pos = np.asarray(action_dict["left_rel_pos"])
        left_rot6d = axis_angle_to_rot6d(action_dict["left_rel_rot_axis_angle"])
        left_gripper = np.asarray(action_dict["left_gripper"])
    elif group == "humanoid":
        right_pos = np.asarray(action_dict["right_abs_pos"])
        right_rot6d = np.asarray(action_dict["right_abs_rot_6d"])  # 已经是 6D，不用转换
        right_gripper = np.asarray(action_dict["right_gripper"])
        left_pos = np.asarray(action_dict["left_abs_pos"])
        left_rot6d = np.asarray(action_dict["left_abs_rot_6d"])
        left_gripper = np.asarray(action_dict["left_gripper"])
    else:
        raise ValueError(f"未知 embodiment_group: {group}")

    lead_shape = right_pos.shape[:-1]
    unified = np.zeros((*lead_shape, ACTION_DIM), dtype=np.float32)

    def _place(name: str, value: np.ndarray):
        slot = ACTION_SLOT_BY_NAME[name]
        assert value.shape[-1] == slot.dim, f"{name}: 期望{slot.dim}维, 实际{value.shape[-1]}维"
        unified[..., slot.offset: slot.offset + slot.dim] = value

    _place("right_arm_pos", right_pos)
    _place("right_arm_rot6d", right_rot6d)
    _place("right_gripper", right_gripper)
    _place("left_arm_pos", left_pos)
    _place("left_arm_rot6d", left_rot6d)
    _place("left_gripper", left_gripper)

    mask = action_group_mask(group)
    return unified, mask


def build_unified_state(
    obs_dict: Dict[str, np.ndarray],
    cfg: DatasetConfig,
    schema: UnifiedStateSchema,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    obs_dict: robomimic 读出的原始 obs 字典（key 为 cfg["low_dim_keys"]）
    返回 (unified_state[..., state_dim], mask[state_dim])
    """
    from dexmg_schema import _low_dim_key_roles  # 复用同一份 key 角色解析逻辑

    group = cfg["embodiment_group"]
    roles = _low_dim_key_roles(cfg)

    right_pos = np.asarray(obs_dict[roles["right"]["pos"]])
    right_rot6d = quat_to_rot6d(obs_dict[roles["right"]["quat"]])
    right_gripper = np.asarray(obs_dict[roles["right"]["gripper"]])
    left_pos = np.asarray(obs_dict[roles["left"]["pos"]])
    left_rot6d = quat_to_rot6d(obs_dict[roles["left"]["quat"]])
    left_gripper = np.asarray(obs_dict[roles["left"]["gripper"]])

    lead_shape = right_pos.shape[:-1]
    unified = np.zeros((*lead_shape, schema.state_dim), dtype=np.float32)

    def _place(name: str, value: np.ndarray):
        slot = schema.slots[name]
        w = value.shape[-1]
        assert w <= slot.dim, f"{name}: 槽位宽度{slot.dim} < 实际数据宽度{w}，探测缓存可能过期"
        # 真实值放在槽位前 w 维，其余(padding)保持0，对应 mask 会把这部分标 0
        unified[..., slot.offset: slot.offset + w] = value

    _place("right_arm_pos", right_pos)
    _place("right_arm_rot6d", right_rot6d)
    _place("right_gripper", right_gripper)
    _place("left_arm_pos", left_pos)
    _place("left_arm_rot6d", left_rot6d)
    _place("left_gripper", left_gripper)

    mask = state_group_mask(schema, group)
    return unified, mask
