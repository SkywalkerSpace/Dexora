"""
check_action_order.py

对 TwoArmBoxCleanup 一帧数据，打印：
  1) dexmimicgen 官方 action_keys 顺序表（来自 generate_training_config.py）
  2) 直接从原始 hdf5 按官方顺序拼接的 "ground truth" 向量
  3) 走完 build_unified_action -> unified_action_to_env 整条 pipeline 后的输出
逐维对照，标出差异超过阈值的位置（旋转部分允许有 rot6d 往返的小数值误差，
其它部分应该几乎精确相等，不相等就是顺序/映射错了）。
"""
import argparse
import numpy as np
import h5py

# === 按你实际的模块路径改这里 ===
from dexmg.dexmg_config import DATASET_CONFIGS
from dexmg.dexmg_schema import build_schema
from dexmg.dexmg_convert import build_unified_action, unified_action_to_env

# 来自 dexmimicgen 官方 generate_training_config.py 的 panda_action_config，
# 顺序是 ground truth —— 每项 (action_dict 的 key, 该 key 的维度)
PANDA_ACTION_KEYS_OFFICIAL = [
    ("right_rel_pos", 3),
    ("right_rel_rot_axis_angle", 3),
    ("right_gripper", None),   # 维度从 hdf5 里实际探测，别硬编码
    ("left_rel_pos", 3),
    ("left_rel_rot_axis_angle", 3),
    ("left_gripper", None),
]


def build_ground_truth_vector(f, demo_id: str, frame_idx: int):
    """按官方顺序，直接从原始 action_dict 拼出这一帧的 env-native 动作向量，
    同时记录每一段在向量里的 (start, end, name) 供后面打印对照表。"""
    grp = f[f"data/{demo_id}/action_dict"]
    parts = []
    layout = []
    cursor = 0
    for key, _dim in PANDA_ACTION_KEYS_OFFICIAL:
        val = np.asarray(grp[key][frame_idx]).reshape(-1)
        parts.append(val)
        layout.append((key, cursor, cursor + len(val)))
        cursor += len(val)
    return np.concatenate(parts), layout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--dataset_name", default="two_arm_box_cleanup",
                     help="要对应 dexmg_config.py 里 DATASET_CONFIGS 的 key")
    ap.add_argument("--demo_id", default="demo_0")
    ap.add_argument("--frame_idx", type=int, default=10)
    ap.add_argument("--tol", type=float, default=1e-3,
                     help="非旋转维度允许的误差；超过这个值标记为可疑")
    ap.add_argument("--rot_tol", type=float, default=0.1,
                     help="旋转维度（axis-angle 分量）允许的误差，稍微放宽因为有 rot6d 往返数值误差")
    args = ap.parse_args()

    cfg = DATASET_CONFIGS[args.dataset_name]
    schema = build_schema(dataset_root=None, cache_dir=None)  # 按实际签名调整

    with h5py.File(args.hdf5, "r") as f:
        gt_vec, gt_layout = build_ground_truth_vector(f, args.demo_id, args.frame_idx)

        # 走 pipeline：读 action_dict -> 统一 schema -> 再转回 env 原生顺序
        action_dict = {
            key: np.asarray(f[f"data/{args.demo_id}/action_dict/{key}"][args.frame_idx])
            for key, _ in PANDA_ACTION_KEYS_OFFICIAL
        }
        unified_action, action_mask = build_unified_action(action_dict, cfg, schema)
        pipeline_vec = unified_action_to_env(unified_action, cfg)  # 按你实际签名调整

    print(f"ground truth dim = {len(gt_vec)}, pipeline output dim = {len(pipeline_vec)}")
    if len(gt_vec) != len(pipeline_vec):
        print("!! 维度数量都对不上，顺序问题基本坐实，不用往下逐维看了 !!")
        return

    print(f"\n{'idx':>4} {'segment':<28} {'ground_truth':>14} {'pipeline_out':>14} "
          f"{'diff':>10}  flag")
    print("-" * 80)

    for key, start, end in gt_layout:
        is_rot = "rot" in key
        tol = args.rot_tol if is_rot else args.tol
        for i in range(start, end):
            diff = abs(gt_vec[i] - pipeline_vec[i])
            flag = "  <-- MISMATCH" if diff > tol else ""
            print(f"{i:>4} {key:<28} {gt_vec[i]:>14.5f} {pipeline_vec[i]:>14.5f} "
                  f"{diff:>10.5f}{flag}")

    n_mismatch = sum(
        1 for key, start, end in gt_layout for i in range(start, end)
        if abs(gt_vec[i] - pipeline_vec[i]) > (args.rot_tol if "rot" in key else args.tol)
    )
    print(f"\n共 {n_mismatch} 个维度超出容差。")
    if n_mismatch == 0:
        print("没有发现顺序/映射错位。")
    else:
        print("存在错位——重点看上面标 MISMATCH 的维度落在哪个 segment，"
              "尤其如果集中在 rot6d/gripper 交界处，大概率是切片下标算错了。")


if __name__ == "__main__":
    main()
