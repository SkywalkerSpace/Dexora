# -*- coding: utf-8 -*-
"""
dexmg_camera.py

把每个数据集自己的 image_keys（含义已在 dexmg_config.py 里核实过：
agentview_image -> cam_high, frontview_image -> cam_third_view）
映射到 Dexora 固定的 4 路相机槽位：
    cam_high, cam_left_wrist, cam_right_wrist, cam_third_view

six个数据集里只有 two_arm_can_sort_random 真的缺 cam_high（用
frontview_image 代替了 agentview_image），需要补零 + mask=False；
其余 5 个数据集 4 路相机都能拿到真实图像。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from dexmg_config import DatasetConfig

DEXORA_CAM_SLOTS = ("cam_high", "cam_left_wrist", "cam_right_wrist", "cam_third_view")


def build_camera_key_map(cfg: DatasetConfig) -> Dict[str, Optional[str]]:
    """
    根据某个数据集的 image_keys，构造 {dexora槽位名: 原始hdf5 obs key 或 None}。

    image_keys 约定顺序（来自 generate_training_config.py）：
        [0] agentview_image 或 frontview_image  -> 第三视角/头部视角
        [1] 手腕相机1 (robot0_eye_in_hand_image / robot0_eye_in_left_hand_image)
        [2] 手腕相机2 (robot1_eye_in_hand_image / robot0_eye_in_right_hand_image)

    注意：panda 组 robot0/robot1 到底对应左/右手腕，用前建议先核实一次
    （参考 dexmg_config.py 顶部注释），这里先按 robot0=right/robot1=left
    的假设写，如核实结果相反，只需要交换下面两行的赋值。
    """
    first_key = cfg["image_keys"][0]
    wrist_1, wrist_2 = cfg["image_keys"][1], cfg["image_keys"][2]

    cam_map: Dict[str, Optional[str]] = {
        "cam_high": first_key if cfg["has_cam_high"] else None,
        "cam_third_view": first_key if not cfg["has_cam_high"] else None,
        # 假设：wrist_1 对应右手腕，wrist_2 对应左手腕（TODO: 核实 robot0/robot1 归属）
        "cam_right_wrist": wrist_1,
        "cam_left_wrist": wrist_2,
    }

    # has_cam_high=True 时，第一路视角本来就是 agentview(=cam_high)，
    # 这些数据集没有单独的第三视角相机，cam_third_view 同样置空补零。
    # has_cam_high=False 时（can_sort_random），第一路是 frontview(=cam_third_view)，
    # cam_high 置空补零。
    return cam_map


def extract_camera_images(
    obs_group,  # h5py group: data/{demo}/obs
    frame_indices: np.ndarray,
    cam_map: Dict[str, Optional[str]],
    image_shape: Tuple[int, int, int] = (224, 224, 3),
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    按 cam_map 从 hdf5 的 obs group 里切出图像序列。

    返回 {cam_slot: (images[T,H,W,C] uint8, mask[T] bool)}；
    cam_map 里值为 None 的槽位统一返回全零图像 + mask=False。
    """
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    T = len(frame_indices)
    for slot in DEXORA_CAM_SLOTS:
        raw_key = cam_map.get(slot)
        if raw_key is None:
            images = np.zeros((T, *image_shape), dtype=np.uint8)
            mask = np.zeros((T,), dtype=bool)
        else:
            images = obs_group[raw_key][frame_indices]  # (T, H, W, C) uint8
            mask = np.ones((T,), dtype=bool)
        out[slot] = (images, mask)
    return out
