#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在训练数据中的随机 demo 上运行 Dexora，并保存视频和 action 对比图。

每个 demo 从 HDF5 中恢复到它自己的初始 simulator state。图中的 prediction
和 dataset action 都位于 dexmg 的统一 schema 空间，因此 Panda 的四组维度为
6+6+6+6；humanoid 会按实际有效维度显示为 9+6+9+6。


cd /Users/skywalker/code/experiment/Dexora

python eval_trained_demos.py \
  --dataset_root /home/mayuhang/datasets/dexmimicgen_datasets/ \
  --model_path /tmp/.X11-unix/checkpoints/dexora-1b-finetune/checkpoint-30000 \
  --model_config_path /home/mayuhang/Dexora/configs/base.yaml \
  --stats_file /home/mayuhang/Dexora/dexmg/configs/dataset_statistics.json \
  --schema_cache_dir /home/mayuhang/Dexora/dexmg/configs \
  --num_demos 5 \
  --horizon 400 \
  --output_dir ./trained_demo_eval

"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import h5py
import imageio
import matplotlib.pyplot as plt
import numpy as np
import robosuite
from robosuite import load_composite_controller_config

import dexmimicgen  # noqa: F401  注册 dexmimicgen 环境

from deploy.dexora_policy import DexoraPolicy, DexoraPolicyConfig
from dexmg.dexmg_config import DatasetConfig, get_dataset_config
from dexmg.dexmg_convert import build_unified_action
from dexmg.dexmg_schema import Schema, build_schema
from dexmg.sim_eval_dexora_dexmg import (
    DEXMG_SLOT_TO_POLICY_CAM,
    ENV_ROBOTS,
    IMAGE_SIZE,
    build_images_for_policy,
    build_state_from_obs,
    infer_gripper_widths,
    make_env,
    normalize,
    unified_action_to_env,
)
from dexmg.dexmg_camera import build_camera_key_map


@dataclass
class DemoRecord:
    hdf5_path: str
    demo_id: str
    env_name: str
    controller_configs: dict
    states: np.ndarray
    model_file: Optional[str]
    ep_meta: Optional[str]
    length: int


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode(value.item())
    return value


def load_demo_records(hdf5_path: str) -> List[DemoRecord]:
    cfg = get_dataset_config(hdf5_path)
    with h5py.File(hdf5_path, "r") as handle:
        env_args = _decode(handle["data"].attrs["env_args"])
        env_args = json.loads(env_args)
        env_name = env_args["env_name"]
        controller_configs = env_args["env_kwargs"]["controller_configs"]
        records = []
        for demo_id in sorted(handle["data"].keys()):
            demo = handle[f"data/{demo_id}"]
            attrs = demo.attrs
            model_file = _decode(attrs.get("model_file"))
            ep_meta = _decode(attrs.get("ep_meta"))
            records.append(
                DemoRecord(
                    hdf5_path=hdf5_path,
                    demo_id=demo_id,
                    env_name=env_name,
                    controller_configs=controller_configs,
                    states=np.asarray(demo["states"][0]),
                    model_file=model_file,
                    ep_meta=ep_meta,
                    length=int(demo["action_dict"][cfg["action_keys"][0]].shape[0]),
                )
            )
    return records


def reset_to_demo(env, record: DemoRecord):
    if record.model_file is not None:
        ep_meta = json.loads(record.ep_meta) if record.ep_meta else {}
        if hasattr(env, "set_attrs_from_ep_meta"):
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):
            env.set_ep_meta(ep_meta)
        env.reset()
        version_id = int(robosuite.__version__.split(".")[1])
        if version_id <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml

            xml = postprocess_model_xml(record.model_file)
        else:
            xml = env.edit_model_xml(record.model_file)
        env.reset_from_xml_string(xml)
        env.sim.reset()
    env.sim.set_state_from_flattened(record.states)
    env.sim.forward()
    if hasattr(env, "update_sites"):
        env.update_sites()
    if hasattr(env, "update_state"):
        env.update_state()
    return env._get_observations(force_update=True)


def read_demo_actions(handle, demo_id: str, cfg: DatasetConfig, schema: Schema):
    group = handle[f"data/{demo_id}/action_dict"]
    action_dict = {key: np.asarray(group[key]) for key in cfg["action_keys"]}
    actions, _ = build_unified_action(action_dict, cfg, schema)
    return actions


def save_action_comparison(predicted, target, action_mask, schema, output_path, title):
    predicted = np.asarray(predicted, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    count = min(len(predicted), len(target))
    predicted, target = predicted[:count], target[:count]
    groups = [
        ("right arm", ["right_arm_pos", "right_arm_rot6d"]),
        ("right gripper", ["right_gripper"]),
        ("left arm", ["left_arm_pos", "left_arm_rot6d"]),
        ("left gripper", ["left_gripper"]),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    time_axis = np.arange(count)
    for axis, (name, slot_names) in zip(axes, groups):
        dims = []
        for slot_name in slot_names:
            slot = schema.slots[slot_name]
            dims.extend(range(slot.offset, slot.offset + slot.dim))
        valid_dims = [dim for dim in dims if action_mask[dim] > 0.5]
        for local_index, dim in enumerate(valid_dims):
            axis.plot(time_axis, target[:, dim], linewidth=1.1,
                      label=f"dataset d{local_index}")
            axis.plot(time_axis, predicted[:, dim], "--", linewidth=1.0,
                      label=f"prediction d{local_index}")
        axis.set_title(f"{name} ({len(valid_dims)} valid dimensions)")
        axis.set_ylabel("action")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right", fontsize=7, ncol=2)
    axes[-1].set_xlabel("demo timestep")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, format="jpg", bbox_inches="tight")
    plt.close(fig)


def collect_records(dataset_root: str, seed: int, num_demos: int):
    rng = np.random.default_rng(seed)
    paths = sorted(glob.glob(os.path.join(dataset_root, "*.hdf5")))
    if not paths:
        raise FileNotFoundError(f"{dataset_root} 下没有找到 *.hdf5")
    all_records = []
    for path in paths:
        try:
            all_records.extend(load_demo_records(path))
        except KeyError as exc:
            print(f"[跳过] {path}: {exc}")
    if not all_records:
        raise RuntimeError("没有找到可用的 data/demo_* 训练样本")
    rng.shuffle(all_records)
    return all_records[: min(num_demos, len(all_records))]


def evaluate_demo(record, policy, schema, stats, args, output_dir):
    cfg = get_dataset_config(record.hdf5_path)
    with h5py.File(record.hdf5_path, "r") as handle:
        dataset_actions = read_demo_actions(handle, record.demo_id, cfg, schema)

    camera_names = sorted(
        {key[:-len("_image")] for key in cfg["image_keys"]} | {args.viz_camera}
    )
    env = make_env(
        record.env_name, camera_names, args.camera_height, args.camera_width,
        has_renderer=False, controller_configs=record.controller_configs,
    )
    action_mask = schema.action_group_mask(cfg["embodiment_group"])
    policy._state_mask = __import__("torch").from_numpy(
        schema.state_group_mask(cfg["embodiment_group"])
    )[None, None, :].to(policy.device, dtype=policy.cfg.dtype)
    policy._action_mask = __import__("torch").from_numpy(action_mask)[None, None, :].to(
        policy.device, dtype=policy.cfg.dtype
    )

    video_path = os.path.join(output_dir, f"{record.demo_id}_policy.mp4")
    image_path = os.path.join(output_dir, f"{record.demo_id}_action_compare.jpg")
    writer = imageio.get_writer(video_path, fps=20)
    predicted_actions = []
    action_queue = []
    unified_action_queue = []
    gripper_widths = None
    try:
        obs = reset_to_demo(env, record)
        gripper_widths = infer_gripper_widths(
            env.action_dim, cfg["embodiment_group"], schema
        )
        steps = min(record.length, len(dataset_actions))
        if args.horizon > 0:
            steps = min(steps, args.horizon)
        for step in range(steps):
            if not action_queue or step % args.replan_interval == 0:
                state_raw, _ = build_state_from_obs(obs, cfg, schema)
                policy_obs = {
                    "state": normalize(state_raw, stats[cfg["dataset_name"]]["state"], args.normalize_mode),
                    "images": build_images_for_policy(obs, cfg),
                    "instruction": args.instruction or cfg["lang"],
                    "ctrl_freq": 20.0,
                }
                chunk = policy.get_action(policy_obs)
                chunk = denormalize_actions(
                    chunk, stats[cfg["dataset_name"]]["action"], args.normalize_mode
                )
                unified_action_queue = list(chunk)
                action_queue = [
                    unified_action_to_env(action, cfg, schema, gripper_widths, env.action_dim)
                    for action in chunk
                ]
            predicted_actions.append(unified_action_queue.pop(0))
            obs, _, done, _ = env.step(action_queue.pop(0))
            video_key = f"{args.viz_camera}_image"
            if video_key in obs:
                writer.append_data(obs[video_key][::-1])
            if done:
                break
    finally:
        writer.close()
        env.close()

    save_action_comparison(
        predicted_actions, dataset_actions, action_mask, schema, image_path,
        f"{cfg['dataset_name']} / {record.demo_id} / prediction vs dataset",
    )
    print(f"[{record.demo_id}] video={video_path} actions={image_path}")


def denormalize_actions(data, stats_entry, mode):
    data = np.asarray(data, dtype=np.float64)
    if mode == "mean_std":
        out = data * np.asarray(stats_entry["std"]) + np.asarray(stats_entry["mean"])
    elif mode == "min_max":
        out = data * (np.asarray(stats_entry["q99"]) - np.asarray(stats_entry["q01"])) + np.asarray(stats_entry["q01"])
    elif mode == "rms":
        out = data * np.asarray(stats_entry["norm"])
    else:
        raise ValueError(f"未知 normalize_mode: {mode}")
    return out.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--schema_cache_dir", default=None)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_config_path", default="configs/base_400m.yaml")
    parser.add_argument("--stats_file", required=True)
    parser.add_argument("--normalize_mode", choices=["min_max", "mean_std", "rms"], default="rms")
    parser.add_argument("--output_dir", default="./trained_demo_eval")
    parser.add_argument("--num_demos", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=0, help="0 表示使用 demo 全长")
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--replan_interval", type=int, default=6)
    parser.add_argument("--camera_height", type=int, default=384)
    parser.add_argument("--camera_width", type=int, default=384)
    parser.add_argument("--viz_camera", default="agentview")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    schema = build_schema(args.dataset_root, args.schema_cache_dir or args.dataset_root)
    with open(args.stats_file, "r") as handle:
        stats = json.load(handle)
    records = collect_records(args.dataset_root, args.seed, args.num_demos)
    os.makedirs(args.output_dir, exist_ok=True)
    policy = DexoraPolicy(
        args.model_path,
        DexoraPolicyConfig(
            model_config_path=args.model_config_path,
            state_dim=schema.dim,
            chunk_size=args.chunk_size,
        ),
    )
    for record in records:
        evaluate_demo(record, policy, schema, stats, args, args.output_dir)


if __name__ == "__main__":
    main()