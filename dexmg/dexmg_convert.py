# -*- coding: utf-8 -*-
"""
dexmg_convert.py

方案 B 的核心：把某个 embodiment group 原生的 action_dict / obs_dict
按固定顺序拼接，写进 dexmg_schema.py 定义的 superset 向量里对应的槽
位，其余槽位保持 0。不做任何 abs<->rel / axis_angle<->rot6d 的格式
转换——两种表示原样保留，靠模型自己在训练里学会按 mask/embodiment
区分。
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from dexmg_config import DatasetConfig
from dexmg_schema import (
    ACTION_DIM,
    ACTION_SLOTS_BY_GROUP,
    StateSlot,
    action_group_mask,
    state_group_mask,
)


def _concat_action_dict(action_dict: Dict[str, np.ndarray], key_order) -> np.ndarray:
    """按 key_order 顺序把 action_dict 里的分量拼接成一个向量。

    action_dict[key] 形状假设为 (T, d_k) 或 (d_k,)；返回 (T, sum(d_k)) 或 (sum(d_k),)。
    """
    parts = [np.asarray(action_dict[k]) for k in key_order]
    return np.concatenate(parts, axis=-1)


def build_unified_action(action_dict: Dict[str, np.ndarray], cfg: DatasetConfig) -> "tuple[np.ndarray, np.ndarray]":
    """
    action_dict: robomimic 从 hdf5 读出来的原始 action_dict
                 (key 为 cfg["action_keys"] 里那几个，如 right_rel_pos 等)
    返回 (unified_action, mask)：
        - unified_action: (..., ACTION_DIM)，只有本 group 的槽位有真实值，其余为0
        - mask:           (ACTION_DIM,)，本 group 槽位=1，其余=0
    """
    group = cfg["embodiment_group"]
    flat = _concat_action_dict(action_dict, cfg["action_keys"])  # (..., group_dim)

    lead_shape = flat.shape[:-1]
    unified = np.zeros((*lead_shape, ACTION_DIM), dtype=np.float32)

    slot_offset = 0
    for slot in ACTION_SLOTS_BY_GROUP[group]:
        # cfg["action_keys"] 里每个 key 的维度之和要正好等于该 group 槽位的总维度，
        # 这里按 flat 里的顺序依次填入每个槽位。
        unified[..., slot.offset: slot.offset + slot.dim] = flat[..., slot_offset: slot_offset + slot.dim]
        slot_offset += slot.dim
    assert slot_offset == flat.shape[-1], (
        f"group={group} 的 action_keys 总维度({flat.shape[-1]}) 和 schema 里"
        f"该 group 槽位总维度({slot_offset}) 不一致，检查 dexmg_schema.py 的 ACTION_SLOTS"
    )

    mask = action_group_mask(group)
    return unified, mask


def build_unified_state(
    obs_dict: Dict[str, np.ndarray],
    cfg: DatasetConfig,
    state_slots: Dict[str, StateSlot],
    state_dim: int,
) -> "tuple[np.ndarray, np.ndarray]":
    """
    obs_dict: robomimic 读出的原始 obs 字典 (key 为 cfg["low_dim_keys"])
    返回 (unified_state, mask)，语义同 build_unified_action。
    """
    group = cfg["embodiment_group"]
    slot = state_slots[f"{group}_state"]

    flat = _concat_action_dict(obs_dict, cfg["low_dim_keys"])  # (..., group_dim)
    assert flat.shape[-1] == slot.dim, (
        f"group={group} 的 low_dim 实际总维度({flat.shape[-1]}) 和探测缓存里的"
        f"槽位维度({slot.dim}) 不一致，数据可能变了，重新跑一次 "
        f"build_state_schema(force_recompute=True)"
    )

    lead_shape = flat.shape[:-1]
    unified = np.zeros((*lead_shape, state_dim), dtype=np.float32)
    unified[..., slot.offset: slot.offset + slot.dim] = flat

    mask = state_group_mask(state_slots, state_dim, group)
    return unified, mask
