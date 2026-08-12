# -*- coding: utf-8 -*-
"""
dexmg_hdf5_vla_dataset.py

Dexora 的新数据后端：直接从 dexmimicgen 原始 hdf5 读取，不经过 LeRobot
转换。对接 Dexora-dataset.py 里 VLAConsumerDataset 的 use_hdf5 分支：

    elif use_hdf5 == "dexmg_hdf5":
        from data.dexmg_hdf5_vla_dataset import DexmgHDF5VLADataset
        self.hdf5_dataset = DexmgHDF5VLADataset(...)

契约（VLAConsumerDataset.__getitem__ 里用到的字段）：
    __len__()
    get_item() -> {
        "meta": {"dataset_name": str, "instruction": str},
        "state": np.ndarray [T, STATE_DIM],
        "actions": np.ndarray [T, ACTION_DIM],
        "state_indicator": np.ndarray [STATE_DIM],   # 复用作 mask
        "cam_high": [T,H,W,C] uint8, "cam_high_mask": [T] bool,
        "cam_right_wrist": ..., "cam_right_wrist_mask": ...,
        "cam_left_wrist": ..., "cam_left_wrist_mask": ...,
        "cam_third_view": ..., "cam_third_view_mask": ...,
        "state_std": [STATE_DIM], "state_mean": [STATE_DIM], "state_norm": [STATE_DIM],
    }

依赖 robomimic（pip install -e libs/robomimic），复用它的
SequenceDataset 做 hdf5 缓存/帧堆叠/滑窗/demo过滤，不重复造轮子。

维度（共享物理槽位版，见 dexmg_schema.py）：
    ACTION_DIM = 30，panda 组和 humanoid 组共用同一套槽位
    （right_arm_pos/right_arm_rot6d/right_gripper/left_arm_pos/
    left_arm_rot6d/left_gripper），不是各占一段；旋转统一存 6D。
    STATE_DIM 探测得到，同样是共享槽位 + gripper 部分 padding。
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional

import h5py
import numpy as np

from robomimic.utils.dataset import SequenceDataset

from dexmg_camera import build_camera_key_map, extract_camera_images
from dexmg_config import DATASET_CONFIGS, DatasetConfig, get_dataset_config
from dexmg_convert import build_unified_action, build_unified_state
from dexmg_schema import ACTION_DIM, UnifiedStateSchema, build_state_schema


class _SingleDexmgReader:
    """包装单个 hdf5 文件的 robomimic SequenceDataset + 本文件专属的语义拼接。"""

    def __init__(
        self,
        hdf5_path: str,
        cfg: DatasetConfig,
        state_schema: UnifiedStateSchema,
        seq_length: int,
        frame_stack: int,
        filter_key: Optional[str],
        image_shape=(224, 224, 3),
    ):
        self.hdf5_path = hdf5_path
        self.cfg = cfg
        self.state_schema = state_schema
        self.image_shape = image_shape
        self.cam_map = build_camera_key_map(cfg)

        # robomimic SequenceDataset 只负责按 demo/滑窗把 low_dim obs + action
        # 的原始序列取出来；图像我们自己按 frame_indices 单独切（见 __getitem_raw__），
        # 避免图像也走 robomimic 的 cache_mode="all" 导致爆内存。
        self._seq_ds = SequenceDataset(
            hdf5_path=hdf5_path,
            obs_keys=cfg["low_dim_keys"],
            action_keys=cfg["action_keys"],
            dataset_keys=["actions"],
            load_next_obs=False,
            frame_stack=frame_stack,
            seq_length=seq_length,
            pad_frame_stack=True,
            pad_seq_length=True,
            get_pad_mask=False,
            hdf5_cache_mode="low_dim",  # 低维数据全量进内存，图像不缓存
            hdf5_use_swmr=True,
            hdf5_normalize_obs=False,
            filter_by_attribute=filter_key,
        )

        # 图像单独用 h5py 直接开（不走 robomimic 的 cache），懒加载
        self._h5file = h5py.File(hdf5_path, "r", swmr=True)

    def __len__(self) -> int:
        return len(self._seq_ds)

    def _frame_indices_for(self, index: int):
        """从 robomimic 内部结构反推这条样本对应的 demo_id 和帧下标区间。"""
        demo_id = self._seq_ds._index_to_demo_id[index]
        demo_start = self._seq_ds._demo_id_to_start_indices[demo_id]
        offset = 0 if self._seq_ds.pad_frame_stack else (self._seq_ds.n_frame_stack - 1)
        index_in_demo = index - demo_start + offset
        seq_len = self._seq_ds.seq_length
        demo_length = self._seq_ds._demo_id_to_demo_length[demo_id]

        start = max(index_in_demo, 0)
        end = min(index_in_demo + seq_len, demo_length)
        frame_indices = np.arange(start, end)
        # pad_seq_length=True 时，SequenceDataset 自己会在 low_dim/action 序列上
        # 做首尾复制补齐；图像这里做同样的补齐，保证和 state/action 的时间轴对齐。
        if len(frame_indices) < seq_len:
            pad_left = max(0, -index_in_demo)
            pad_right = seq_len - len(frame_indices) - pad_left
            frame_indices = np.concatenate([
                np.full(pad_left, frame_indices[0] if len(frame_indices) else 0, dtype=int),
                frame_indices,
                np.full(pad_right, frame_indices[-1] if len(frame_indices) else 0, dtype=int),
            ])
        return demo_id, frame_indices

    def get_item(self, index: int) -> dict:
        raw = self._seq_ds.get_item(index)  # {"obs": {...}, "actions": (T, group_action_dim), ...}
        demo_id, frame_indices = self._frame_indices_for(index)

        # --- state / action 拼进统一(共享物理槽位) schema ---
        unified_state, state_mask = build_unified_state(
            raw["obs"], self.cfg, self.state_schema
        )
        # robomimic 已经把 action_keys 按顺序 concat 好放进 raw["actions"]，
        # 这里再拆回 action_dict 是为了复用 build_unified_action 的按-key拼接逻辑，
        # 也方便以后某个 key 的宽度变化时只改 dexmg_config.py。
        action_dict = self._split_action_vector(raw["actions"])
        unified_action, action_mask = build_unified_action(action_dict, self.cfg)

        # --- 图像 ---
        obs_group = self._h5file[f"data/{demo_id}/obs"]
        cams = extract_camera_images(obs_group, frame_indices, self.cam_map, self.image_shape)

        item = {
            "meta": {
                "dataset_name": self.cfg["dataset_name"],
                "instruction": self.cfg["lang"],
            },
            "state": unified_state,
            "actions": unified_action,
            "state_indicator": state_mask,
            "action_mask": action_mask,  # 训练脚本里传给 RDTRunner.compute_loss 的 action_mask
        }
        for slot, (imgs, mask) in cams.items():
            item[slot] = imgs
            item[f"{slot}_mask"] = mask
        return item

    def _split_action_vector(self, flat_action: np.ndarray) -> Dict[str, np.ndarray]:
        out = {}
        offset = 0
        for key in self.cfg["action_keys"]:
            # 各 key 的宽度可以从 schema 的 group 槽位里反查，这里用一个简单的
            # 静态表代替（和 dexmg_config.py 里 action_keys 的定义一一对应）。
            dim = _ACTION_KEY_DIMS[key]
            out[key] = flat_action[..., offset: offset + dim]
            offset += dim
        assert offset == flat_action.shape[-1]
        return out


# action_dict 里每个原始 key 的维度（来自 dexmimicgen 的 action_dict 定义，
# 和上一轮确认过的 24 维/30 维拆分一致）
_ACTION_KEY_DIMS = {
    "right_rel_pos": 3, "right_rel_rot_axis_angle": 3, "right_gripper": 6,
    "left_rel_pos": 3, "left_rel_rot_axis_angle": 3, "left_gripper": 6,
    "right_abs_pos": 3, "right_abs_rot_6d": 6, "left_abs_pos": 3, "left_abs_rot_6d": 6,
}


class DexmgHDF5VLADataset:
    """多个 dexmimicgen hdf5 文件的加权混合，供 Dexora VLAConsumerDataset 使用。"""

    def __init__(
        self,
        dataset_root: str,
        stats_file: str,
        filter_keys: Optional[Dict[str, str]] = None,
        dataset_weights: Optional[Dict[str, float]] = None,
        seq_length: int = 16,
        frame_stack: int = 1,
        image_shape=(224, 224, 3),
        schema_cache_dir: Optional[str] = None,
    ):
        self.dataset_root = dataset_root
        filter_keys = filter_keys or {}
        dataset_weights = dataset_weights or {}
        schema_cache_dir = schema_cache_dir or dataset_root

        self.state_schema = build_state_schema(
            dataset_root=dataset_root, cache_dir=schema_cache_dir
        )
        self.state_dim = self.state_schema.state_dim
        self.action_dim = ACTION_DIM

        self._readers: List[_SingleDexmgReader] = []
        self._weights: List[float] = []
        for hdf5_name, cfg in DATASET_CONFIGS.items():
            hdf5_path = os.path.join(dataset_root, hdf5_name)
            if not os.path.exists(hdf5_path):
                continue  # 允许只放部分数据集
            reader = _SingleDexmgReader(
                hdf5_path=hdf5_path,
                cfg=cfg,
                state_schema=self.state_schema,
                seq_length=seq_length,
                frame_stack=frame_stack,
                filter_key=filter_keys.get(hdf5_name),
                image_shape=image_shape,
            )
            self._readers.append(reader)
            self._weights.append(dataset_weights.get(hdf5_name, 1.0))

        assert len(self._readers) > 0, f"{dataset_root} 下没有找到任何配置好的 dexmg hdf5 文件"
        self._lens = np.array([len(r) for r in self._readers])
        self._bins = np.cumsum([0] + list(self._lens))

        # 归一化统计量：离线预计算好的 json，key 为 dataset_name
        with open(stats_file, "r") as f:
            self._stats = json.load(f)

    def __len__(self) -> int:
        return int(self._lens.sum())

    def _locate(self, index: int):
        reader_idx = int(np.searchsorted(self._bins, index, side="right") - 1)
        local_idx = index - self._bins[reader_idx]
        return reader_idx, local_idx

    def get_item(self, index: Optional[int] = None) -> dict:
        if index is None:
            # 按数据集长度加权随机采样（也可以换成按 self._weights 采样，
            # 这里用长度加权是 robomimic MetaDataset 的默认行为）
            index = random.randrange(len(self))
        reader_idx, local_idx = self._locate(index)
        reader = self._readers[reader_idx]
        item = reader.get_item(local_idx)

        ds_name = item["meta"]["dataset_name"]
        stat = self._stats[ds_name]
        item["state_mean"] = np.array(stat["state_mean"], dtype=np.float32)
        item["state_std"] = np.array(stat["state_std"], dtype=np.float32)
        item["state_norm"] = np.array(stat["state_norm"], dtype=np.float32)
        return item
