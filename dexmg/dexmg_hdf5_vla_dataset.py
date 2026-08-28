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

本版本新增：对齐 dataset_robocasa.py 里 DexmgDataset 的 hdf5_file_video
机制 —— 训练时可以从单独的、分辨率更高的 hdf5 文件里读取相机图像，而不是
只用原始 demo hdf5（通常只存了 84x84 的小图）。分辨率通过 video_res 参数
（或环境变量 DEXMG_VIDEO_RESOLUTION）指定，默认 "84x84" 即不启用、行为与
之前完全一致。启用后，图像会从
    {dataset_root}/videos_{video_res}/{原hdf5文件名去掉.hdf5}_videos.hdf5
读取，该文件内部结构（data/{ep}/obs/{key}）与原 demo hdf5 保持一致，只是
图像分辨率不同。
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional

import h5py
import numpy as np

from robomimic.utils.dataset import SequenceDataset

from dexmg.dexmg_camera import build_camera_key_map, extract_camera_images
from dexmg.dexmg_config import DATASET_CONFIGS, DatasetConfig
from dexmg.dexmg_convert import build_unified_action, build_unified_state
from dexmg.dexmg_schema import Schema, build_schema

import robomimic.utils.obs_utils as ObsUtils
from dexmg.dexmg_config import DATASET_CONFIGS

_obs_utils_initialized = False

def _ensure_obs_utils_initialized():
    global _obs_utils_initialized
    if _obs_utils_initialized:
        return
    low_dim_keys = set()
    rgb_keys = set()
    for cfg in DATASET_CONFIGS.values():
        low_dim_keys.update(cfg["low_dim_keys"])
        rgb_keys.update(cfg["image_keys"])
    ObsUtils.initialize_obs_utils_with_obs_specs({
        "obs": {
            "low_dim": sorted(low_dim_keys),
            "rgb": sorted(rgb_keys),
        }
    })
    _obs_utils_initialized = True

_ensure_obs_utils_initialized()

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
        video_res: Optional[str] = None,
    ):
        self.hdf5_path = hdf5_path
        self.cfg = cfg
        self.schema = schema
        self.image_shape = image_shape
        self.cam_map = build_camera_key_map(cfg)

        # ------------------------------------------------------------------
        # 高分辨率视频 hdf5（对齐 dataset_robocasa.py 的 DexmgDataset.hdf5_file_video）
        #
        # video_res 优先取构造参数，其次取环境变量 DEXMG_VIDEO_RESOLUTION，默认
        # "84x84"（即不启用，图像仍从原始 demo hdf5 读取，行为与之前完全一致）。
        # 启用时，图像观测改为从同目录下 videos_{video_res}/ 子目录里的独立 hdf5
        # 文件读取，该文件内部结构（data/{ep}/obs/{key}）与原文件一致，只是分辨率
        # 更高；state/action 等低维数据仍然从原始 demo hdf5（self._h5file）读取。
        self.video_res = video_res or os.environ.get("DEXMG_VIDEO_RESOLUTION", "84x84")
        self._h5file_video: Optional[h5py.File] = None
        if self.video_res != "84x84":
            hdf5_path_video = os.path.join(
                os.path.dirname(hdf5_path),
                f"videos_{self.video_res}",
                os.path.basename(hdf5_path).replace(".hdf5", "_videos.hdf5"),
            )
            assert os.path.exists(hdf5_path_video), (
                f"未找到分辨率 {self.video_res} 对应的视频 hdf5 文件: {hdf5_path_video}\n"
                f"（对齐 dataset_robocasa.py 的 DexmgDataset：这个文件应由预处理脚本"
                f"预先生成在 videos_{self.video_res}/ 目录下，文件名与原 demo hdf5 同名，"
                f"仅将 .hdf5 后缀替换为 _videos.hdf5）"
            )
            self._h5file_video = h5py.File(hdf5_path_video, "r", swmr=True)
            self._hdf5_path_video = hdf5_path_video
            # 视频 hdf5 里的 demo 数量可能少于原始 demo hdf5（例如渲染视频时
            # 跳过了个别失败/质量不合格的 demo）。逐 demo 判断存在性，缺失时
            # 回退读原始分辨率图像，而不是直接崩掉；每个缺失的 demo 只警告一次。
            self._missing_video_demos: set = set()

        # 注意：dexmimicgen 把 action 分量存在 data/{ep}/action_dict/{key} 下（不是
        # data/{ep}/{key} 顶层），而 robomimic SequenceDataset 内部（get_action_traj 等）
        # 是直接拼接 "data/{ep}/{action_key}" 去读的，所以传给 SequenceDataset 的
        # action_keys/action_config 必须带上 "action_dict/" 前缀。
        # 这个前缀只在这里、局部用于构造 SequenceDataset —— 不写回 self.cfg，
        # 因为 _read_action_dict()/dexmg_convert.build_unified_action() 用的是
        # cfg["action_keys"] 的裸 key 名（grp 本身已经是 action_dict group，
        # 再加前缀会多找一层不存在的子 group）。
        seq_action_keys = [f"action_dict/{k}" for k in cfg["action_keys"]]
        seq_action_config = {f"action_dict/{k}": v for k, v in cfg["action_config"].items()}

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
            action_keys=seq_action_keys,
            action_config=seq_action_config,
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

    def _get_camera_obs_group(self, demo_id: str):
        """
        返回用于读取相机图像的 obs group。若启用了高分辨率视频文件
        (self._h5file_video is not None) 且该 demo 在视频文件里存在，则从
        视频文件读；否则（未启用高分辨率视频，或该 demo 在视频文件里缺失）
        回退到原始 demo hdf5 (self._h5file)，保证不会因为个别 demo 没有
        渲染高分辨率视频而导致训练崩溃。
        """
        if self._h5file_video is not None:
            path = f"data/{demo_id}/obs"
            if path in self._h5file_video:
                return self._h5file_video[path]
            if demo_id not in self._missing_video_demos:
                self._missing_video_demos.add(demo_id)
                print(
                    f"[dexmg_hdf5_vla_dataset] 警告: {demo_id} 在视频 hdf5 "
                    f"{self._hdf5_path_video} 中不存在，该 demo 的图像回退到 "
                    f"原始分辨率读取 (仅提示一次)"
                )
        return self._h5file[f"data/{demo_id}/obs"]

    def _read_action_dict(self, demo_id: str, frame_indices: np.ndarray) -> Dict[str, np.ndarray]:
        grp = self._h5file[f"data/{demo_id}/action_dict"]
        frame_indices = np.asarray(frame_indices)

        # h5py fancy indexing 要求严格递增且不重复
        unique_idx, inverse = np.unique(frame_indices, return_inverse=True)

        out = {}
        for key in self.cfg["action_keys"]:
            raw = grp[key][unique_idx]      # 用去重递增索引读取（h5py支持）
            out[key] = raw[inverse]          # 映射回原始顺序（含重复/padding）
        return out

    def get_item(self, index: int) -> dict:
        raw = self._seq_ds.get_item(index)
        demo_id, frame_indices = self._frame_indices_for(index)

        unified_state, state_mask = build_unified_state(raw["obs"], self.cfg, self.schema)

        action_dict = self._read_action_dict(demo_id, frame_indices)
        unified_action, action_mask = build_unified_action(action_dict, self.cfg, self.schema)

        # 图像观测优先从高分辨率视频 hdf5 读取（若已启用且该 demo 存在）；否则
        # （未启用，或该 demo 在视频文件里缺失）回退到原始 demo hdf5。
        obs_group = self._get_camera_obs_group(demo_id)
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
        video_res: Optional[str] = None,
    ):
        """
        video_res: 相机图像的读取分辨率。None（默认）时读取环境变量
            DEXMG_VIDEO_RESOLUTION，再默认为 "84x84"（不启用高分辨率视频，
            直接从原始 demo hdf5 读图，行为与之前完全一致）。传入例如
            "256x256" 时，会改从
            {dataset_root}/videos_256x256/{hdf5文件名}_videos.hdf5 读取图像，
            对齐 dataset_robocasa.py 里 DexmgDataset 的 hdf5_file_video 机制。
        """
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
                video_res=video_res,
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
        item["state_mean"] = np.array(stat["state"]["mean"], dtype=np.float32)
        item["state_std"] = np.array(stat["state"]["std"], dtype=np.float32)
        item["state_norm"] = np.array(stat["state"]["norm"], dtype=np.float32)
        item["action_norm"] = np.array(stat["action"]["norm"], dtype=np.float32)
        return item
