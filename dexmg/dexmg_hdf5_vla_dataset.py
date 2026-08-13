# -*- coding: utf-8 -*-
"""
dexmg_hdf5_vla_dataset.py (第三次重写 —— state_dim == action_dim == schema.dim)

对接 Dexora-dataset.py 的 VLAConsumerDataset:

    elif use_hdf5 == "dexmg_hdf5":
        from data.dexmg_hdf5_vla_dataset import DexmgHDF5VLADataset
        self.hdf5_dataset = DexmgHDF5VLADataset(...)

get_item() 契约同前几版，唯一变化是 state/actions 现在共用同一个维度
M = self.state_dim = self.action_dim = schema.dim（因为 Dexora 实例化
RDTRunner 时用 action_dim=config["common"]["state_dim"]，两者天生是
同一个数字）。
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
from dexmg_config import DATASET_CONFIGS, DatasetConfig
from dexmg_convert import build_unified_action, build_unified_state
from dexmg_schema import Schema, build_schema


class _SingleDexmgReader:
    def __init__(
        self,
        hdf5_path: str,
        cfg: DatasetConfig,
        schema: Schema,
        seq_length: int,
        frame_stack: int,
        filter_key: Optional[str],
        image_shape=(224, 224, 3),
    ):
        self.hdf5_path = hdf5_path
        self.cfg = cfg
        self.schema = schema
        self.image_shape = image_shape
        self.cam_map = build_camera_key_map(cfg)

        self._seq_ds = SequenceDataset(
            hdf5_path=hdf5_path,
            obs_keys=cfg["low_dim_keys"],
            dataset_keys=["actions"],  # 只用来让 robomimic 内部索引结构正常工作，值不使用
            load_next_obs=False,
            frame_stack=frame_stack,
            seq_length=seq_length,
            pad_frame_stack=True,
            pad_seq_length=True,
            get_pad_mask=False,
            hdf5_cache_mode="low_dim",
            hdf5_use_swmr=True,
            hdf5_normalize_obs=False,
            filter_by_attribute=filter_key,
        )

        self._h5file = h5py.File(hdf5_path, "r", swmr=True)

    def __len__(self) -> int:
        return len(self._seq_ds)

    def _frame_indices_for(self, index: int):
        demo_id = self._seq_ds._index_to_demo_id[index]
        demo_start = self._seq_ds._demo_id_to_start_indices[demo_id]
        offset = 0 if self._seq_ds.pad_frame_stack else (self._seq_ds.n_frame_stack - 1)
        index_in_demo = index - demo_start + offset
        seq_len = self._seq_ds.seq_length
        demo_length = self._seq_ds._demo_id_to_demo_length[demo_id]

        start = max(index_in_demo, 0)
        end = min(index_in_demo + seq_len, demo_length)
        frame_indices = np.arange(start, end)
        if len(frame_indices) < seq_len:
            pad_left = max(0, -index_in_demo)
            pad_right = seq_len - len(frame_indices) - pad_left
            frame_indices = np.concatenate([
                np.full(pad_left, frame_indices[0] if len(frame_indices) else 0, dtype=int),
                frame_indices,
                np.full(pad_right, frame_indices[-1] if len(frame_indices) else 0, dtype=int),
            ])
        return demo_id, frame_indices

    def _read_action_dict(self, demo_id: str, frame_indices: np.ndarray) -> Dict[str, np.ndarray]:
        out = {}
        needed_keys = set(self.cfg["action_keys"]) | {"right_gripper", "left_gripper"}
        grp = self._h5file[f"data/{demo_id}/action_dict"]
        for key in needed_keys:
            out[key] = grp[key][frame_indices]
        return out

    def get_item(self, index: int) -> dict:
        raw = self._seq_ds.get_item(index)
        demo_id, frame_indices = self._frame_indices_for(index)

        unified_state, state_mask = build_unified_state(raw["obs"], self.cfg, self.schema)

        action_dict = self._read_action_dict(demo_id, frame_indices)
        unified_action, action_mask = build_unified_action(action_dict, self.cfg, self.schema)

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
            "action_mask": action_mask,
        }
        for slot, (imgs, mask) in cams.items():
            item[slot] = imgs
            item[f"{slot}_mask"] = mask
        return item


class DexmgHDF5VLADataset:
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

        self.schema = build_schema(dataset_root=dataset_root, cache_dir=schema_cache_dir)
        # state_dim 和 action_dim 是同一个数字 M，对齐 Dexora RDTRunner 的
        # action_dim=config["common"]["state_dim"] 约束
        self.state_dim = self.schema.dim
        self.action_dim = self.schema.dim

        self._readers: List[_SingleDexmgReader] = []
        self._weights: List[float] = []
        for hdf5_name, cfg in DATASET_CONFIGS.items():
            hdf5_path = os.path.join(dataset_root, hdf5_name)
            if not os.path.exists(hdf5_path):
                continue
            reader = _SingleDexmgReader(
                hdf5_path=hdf5_path,
                cfg=cfg,
                schema=self.schema,
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
