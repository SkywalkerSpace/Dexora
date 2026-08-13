# -*- coding: utf-8 -*-
"""
dexmg_config.py

six个 dexmimicgen hdf5 数据集的原始 key 配置表，直接抄自官方
`generate_training_config.py`（不做任何推断/假设）。这是后续所有
模块（schema / camera / convert / dataset）共用的唯一真源（single
source of truth）。

两个 embodiment group：
    - "panda"    : two_arm_box_cleanup / two_arm_lift_tray / two_arm_drawer_cleanup
                   相对位姿 + axis-angle 旋转，right_arm/right_gripper/left_arm/left_gripper
    - "humanoid" : two_arm_pouring / two_arm_coffee / two_arm_can_sort_random
                   绝对位姿 + rot_6d 旋转，right_arm/left_arm/right_gripper/left_gripper

相机注意点（已核实，不是简单的"3路视频"）：
    - agentview_image  -> 对应 Dexora 的 cam_high（头部视角）
    - frontview_image  -> 对应 Dexora 的 cam_third_view（第三视角/前视）
    - two_arm_can_sort_random 用 frontview_image 代替 agentview_image，
      是六个数据集里唯一真正缺 cam_high 的数据集，需要在读取层补零+mask。
    - panda 组的 robot0_eye_in_hand_image / robot1_eye_in_hand_image
      到底哪个是左手哪个是右手，建议用 low_dim 里 robot0_eef_pos /
      robot1_eef_pos 的实际数值符号核实一次，不要凭命名猜
      （下面先按 robot0=right / robot1=left 放，用之前务必验证）。
"""

from __future__ import annotations

from typing import Dict, List, Optional, TypedDict


class DatasetConfig(TypedDict):
    embodiment_group: str  # "panda" | "humanoid"
    dataset_name: str  # 用于 control_freq / dataset_stat 等下游 key
    image_keys: List[str]  # 顺序固定：[第三视角/头部视角键, 手腕1, 手腕2]
    has_cam_high: bool  # 这个 hdf5 里 image_keys[0] 是否其实是 agentview(=cam_high)
    low_dim_keys: List[str]
    action_keys: List[str]
    action_config: Dict[str, Dict[str, Optional[str]]]
    lang: str


# ---------------------------------------------------------------------------
# panda 组：24维，相对位姿 + axis-angle
# ---------------------------------------------------------------------------
_PANDA_LOW_DIM = [
    "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
    "robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos",
]
_PANDA_ACTION_KEYS = [
    "right_rel_pos", "right_rel_rot_axis_angle", "right_gripper",
    "left_rel_pos", "left_rel_rot_axis_angle", "left_gripper",
]
# robomimic SequenceDataset 的必填参数：每个 action_keys 里的 key 都要在这里有对应条目。
# normalization 统一设为 None —— 归一化/格式转换已经在 dexmg_convert.py / dexmg_rotation.py /
# compute_dexmg_stats.py 里自己离线做了，这里只让 robomimic 老实读出原始值，不要让它做
# 二次归一化或运行时格式转换（不设 format / convert_at_runtime）。
_PANDA_ACTION_CONFIG = {k: {"normalization": None} for k in _PANDA_ACTION_KEYS}

# ---------------------------------------------------------------------------
# humanoid 组：绝对位姿 + rot_6d
# ---------------------------------------------------------------------------
_HUMANOID_LOW_DIM = [
    "robot0_right_eef_pos", "robot0_right_eef_quat", "robot0_right_gripper_qpos",
    "robot0_left_eef_pos", "robot0_left_eef_quat", "robot0_left_gripper_qpos",
]
_HUMANOID_ACTION_KEYS = [
    "right_abs_pos", "right_abs_rot_6d", "left_abs_pos", "left_abs_rot_6d",
    "right_gripper", "left_gripper",
]
_HUMANOID_ACTION_CONFIG = {k: {"normalization": None} for k in _HUMANOID_ACTION_KEYS}

DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "two_arm_box_cleanup.hdf5": {
        "embodiment_group": "panda",
        "dataset_name": "dexmg_two_arm_box_cleanup",
        "image_keys": ["agentview_image", "robot0_eye_in_hand_image", "robot1_eye_in_hand_image"],
        "has_cam_high": True,
        "low_dim_keys": _PANDA_LOW_DIM,
        "action_keys": _PANDA_ACTION_KEYS,
        "action_config": _PANDA_ACTION_CONFIG,
        "lang": "move the box lid onto the box",
    },
    "two_arm_lift_tray.hdf5": {
        "embodiment_group": "panda",
        "dataset_name": "dexmg_two_arm_lift_tray",
        "image_keys": ["agentview_image", "robot0_eye_in_hand_image", "robot1_eye_in_hand_image"],
        "has_cam_high": True,
        "low_dim_keys": _PANDA_LOW_DIM,
        "action_keys": _PANDA_ACTION_KEYS,
        "action_config": _PANDA_ACTION_CONFIG,
        "lang": "put the two objects in the tray and then lift the tray",
    },
    "two_arm_drawer_cleanup.hdf5": {
        "embodiment_group": "panda",
        "dataset_name": "dexmg_two_arm_drawer_cleanup",
        "image_keys": ["agentview_image", "robot0_eye_in_hand_image", "robot1_eye_in_hand_image"],
        "has_cam_high": True,
        "low_dim_keys": _PANDA_LOW_DIM,
        "action_keys": _PANDA_ACTION_KEYS,
        "action_config": _PANDA_ACTION_CONFIG,
        "lang": "pick the cup and open the drawer, then put the cup in the drawer and close the drawer",
    },
    "two_arm_pouring.hdf5": {
        "embodiment_group": "humanoid",
        "dataset_name": "dexmg_two_arm_pouring",
        "image_keys": ["agentview_image", "robot0_eye_in_left_hand_image", "robot0_eye_in_right_hand_image"],
        "has_cam_high": True,
        "low_dim_keys": _HUMANOID_LOW_DIM,
        "action_keys": _HUMANOID_ACTION_KEYS,
        "action_config": _HUMANOID_ACTION_CONFIG,
        "lang": "pick the cup and open the drawer, then put the cup in the drawer and close the drawer",
    },
    "two_arm_coffee.hdf5": {
        "embodiment_group": "humanoid",
        "dataset_name": "dexmg_two_arm_coffee",
        "image_keys": ["agentview_image", "robot0_eye_in_left_hand_image", "robot0_eye_in_right_hand_image"],
        "has_cam_high": True,
        "low_dim_keys": _HUMANOID_LOW_DIM,
        "action_keys": _HUMANOID_ACTION_KEYS,
        "action_config": _HUMANOID_ACTION_CONFIG,
        "lang": "insert the coffee pod into the coffee machine and close the lid",
    },
    "two_arm_can_sort_random.hdf5": {
        "embodiment_group": "humanoid",
        "dataset_name": "dexmg_two_arm_can_sort_random",
        # 注意：这里第三视角是 frontview_image，不是 agentview_image
        "image_keys": ["frontview_image", "robot0_eye_in_left_hand_image", "robot0_eye_in_right_hand_image"],
        "has_cam_high": False,  # 六个数据集里唯一真的缺 cam_high 的
        "low_dim_keys": _HUMANOID_LOW_DIM,
        "action_keys": _HUMANOID_ACTION_KEYS,
        "action_config": _HUMANOID_ACTION_CONFIG,
        "lang": "pick the can and put it in the box",
    },
}


def get_dataset_config(hdf5_path: str) -> DatasetConfig:
    import os

    name = os.path.basename(hdf5_path)
    if name not in DATASET_CONFIGS:
        raise KeyError(
            f"未知的 dexmimicgen hdf5 文件名: {name!r}，"
            f"请先在 DATASET_CONFIGS 里补充它的相机/状态/动作 key 配置。"
        )
    return DATASET_CONFIGS[name]


def list_hdf5_by_group(group: str) -> List[str]:
    return [name for name, cfg in DATASET_CONFIGS.items() if cfg["embodiment_group"] == group]


EMBODIMENT_GROUPS: List[str] = ["panda", "humanoid"]
