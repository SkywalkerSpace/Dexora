#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_dexmg_assumptions.py

一次性验证 dexmg 数据处理管线里两个"没有真正验证过、纯靠命名习惯猜测"的假设：

  1. state 侧 robot0 / robot1 到底对应右臂还是左臂
     （对应 dexmg_schema.py::_low_dim_key_roles 里的假设）
  2. eef 四元数到底是 xyzw 还是 wxyz 顺序
     （对应 dexmg_rotation.py::quat_to_matrix 默认的 xyzw 假设）

只对 panda 组（相对位姿动作，right_rel_pos / left_rel_pos 有明确物理意义）
做左右臂检查——humanoid 组的 low_dim_keys 本身已经带 right_/left_ 前缀，
不存在"猜"的问题，跳过即可。

用法：
    python verify_dexmg_assumptions.py \
        --dataset_root /home/mayuhang/datasets/dexmimicgen_datasets \
        --hdf5 two_arm_box_cleanup.hdf5 \
        --out_dir ./dexmg_verify_out

建议用 two_arm_box_cleanup / two_arm_lift_tray / two_arm_drawer_cleanup
三个 panda 组数据集之一（双臂动作明显、不容易出现"两臂都不怎么动"导致
相关性判断不出来的情况）。跑完看终端输出的结论 + --out_dir 下保存的
手腕相机截图（肉眼再确认一次）。

不依赖 robomimic / dexmimicgen，只用 h5py + numpy + PIL。
"""
from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
from PIL import Image

WRIST_IMAGE_KEYS = ("robot0_eye_in_hand_image", "robot1_eye_in_hand_image")


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """整段轨迹拉平后的余弦相似度：a、b 都是 (T, 3) 的位移序列。"""
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def check_left_right(f: h5py.File, demo: str, out_dir: str) -> None:
    print("\n" + "=" * 70)
    print("[检查 1] state 侧 robot0 / robot1 到底对应右臂还是左臂")
    print("=" * 70)

    obs = f[f"data/{demo}/obs"]
    action_dict = f[f"data/{demo}/action_dict"]

    required = ["robot0_eef_pos", "robot1_eef_pos"]
    if not all(k in obs for k in required) or \
       not all(k in action_dict for k in ["right_rel_pos", "left_rel_pos"]):
        print("[跳过] 这个 hdf5 不是 panda 组（没有 robot0/robot1_eef_pos 或 "
              "right/left_rel_pos），左右臂检查只对 panda 组有意义，直接跳过。")
        return

    robot0_pos = obs["robot0_eef_pos"][()]  # (T, 3)
    robot1_pos = obs["robot1_eef_pos"][()]  # (T, 3)
    right_rel_pos = action_dict["right_rel_pos"][()]  # (T, 3)
    left_rel_pos = action_dict["left_rel_pos"][()]    # (T, 3)

    # 用 eef_pos 的逐帧差分（真实位移）去和 action 里记录的相对位移比较，
    # 二者理论上应该高度一致（同符号、同量级），除非左右被搞反了。
    robot0_delta = np.diff(robot0_pos, axis=0)
    robot1_delta = np.diff(robot1_pos, axis=0)
    T = len(robot0_delta)
    right_rel_pos_aligned = right_rel_pos[:T]
    left_rel_pos_aligned = left_rel_pos[:T]

    corr_r0_right = _cos_sim(robot0_delta, right_rel_pos_aligned)
    corr_r0_left = _cos_sim(robot0_delta, left_rel_pos_aligned)
    corr_r1_right = _cos_sim(robot1_delta, right_rel_pos_aligned)
    corr_r1_left = _cos_sim(robot1_delta, left_rel_pos_aligned)

    print(f"robot0_eef_pos 差分 与 right_rel_pos 的余弦相似度: {corr_r0_right:+.4f}")
    print(f"robot0_eef_pos 差分 与 left_rel_pos  的余弦相似度: {corr_r0_left:+.4f}")
    print(f"robot1_eef_pos 差分 与 right_rel_pos 的余弦相似度: {corr_r1_right:+.4f}")
    print(f"robot1_eef_pos 差分 与 left_rel_pos  的余弦相似度: {corr_r1_left:+.4f}")

    # 正确映射应该是 (robot_X, right/left) 两两之间相似度接近 +1，
    # 错配的那一对应该明显更低（两条手臂各自独立运动，互相关性弱）。
    if corr_r0_right > 0.8 and corr_r1_left > 0.8:
        verdict = "robot0 = 右臂, robot1 = 左臂 —— 和 dexmg_schema.py 当前假设一致 [OK]"
    elif corr_r0_left > 0.8 and corr_r1_right > 0.8:
        verdict = ("robot0 = 左臂, robot1 = 右臂 —— 和 dexmg_schema.py 当前假设相反 [FAIL]，"
                   "需要去 dexmg_schema.py::_low_dim_key_roles 把左右判断反过来")
    else:
        verdict = ("四个相似度都不够高（<0.8），可能是这条 demo 里两臂运动相关性本来就弱，"
                   "建议换一条动作更明显的 demo，或者换 two_arm_lift_tray / "
                   "two_arm_drawer_cleanup 重跑一次再综合判断。")
    print(f"\n>>> 结论: {verdict}\n")

    # 顺手把手腕相机首帧存出来，方便肉眼确认"抓取动作发生在画面里的哪只手"
    for key in WRIST_IMAGE_KEYS:
        if key in obs:
            img = obs[key][0]
            Image.fromarray(img).save(os.path.join(out_dir, f"{demo}_{key}_frame0.png"))
    if "agentview_image" in obs:
        Image.fromarray(obs["agentview_image"][0]).save(
            os.path.join(out_dir, f"{demo}_agentview_frame0.png"))
    print(f"[已保存] 手腕相机 + 第三视角首帧图片到 {out_dir}，可以肉眼再确认一遍")


def check_quat_order(f: h5py.File, demo: str) -> None:
    print("\n" + "=" * 70)
    print("[检查 2] eef 四元数是 xyzw 还是 wxyz")
    print("=" * 70)

    obs = f[f"data/{demo}/obs"]
    quat_keys = [k for k in obs.keys() if k.endswith("eef_quat")]
    if not quat_keys:
        print("[跳过] 这个 hdf5 里没找到 *eef_quat 字段。")
        return

    for key in quat_keys:
        quat = obs[key][()]  # (T, 4)
        mean = quat.mean(axis=0)
        std = quat.std(axis=0)
        abs_mean = np.abs(quat).mean(axis=0)
        norm = np.linalg.norm(quat, axis=1)

        print(f"\n{key}: (T={quat.shape[0]})")
        print(f"  每维 mean     : {mean}")
        print(f"  每维 std      : {std}")
        print(f"  每维 |mean|   : {abs_mean}")
        print(f"  norm 范围     : min={norm.min():.4f}, max={norm.max():.4f} "
              f"(应非常接近 1；不接近的话说明这4维根本不是同一个四元数，先查 key 对不对)")

        # 启发式：标量分量(w)在末端夹爪姿态大体稳定的任务里，通常绝对值偏大、
        # 方差偏小（末端姿态只在小范围内摆动，不会连续转过大角度）。
        idx_largest_absmean = int(np.argmax(abs_mean))
        idx_smallest_std = int(np.argmin(std))
        print(f"  |mean| 最大的分量下标: {idx_largest_absmean}")
        print(f"  std   最小的分量下标: {idx_smallest_std}")
        if idx_largest_absmean == 3 and idx_smallest_std == 3:
            print("  -> 下标3（最后一维）最像标量部分 w，支持当前的 xyzw 假设")
        elif idx_largest_absmean == 0 and idx_smallest_std == 0:
            print("  -> 下标0（第一维）最像标量部分 w，和当前 xyzw 假设相反，应该是 wxyz")
        else:
            print("  -> 不够明显，光靠统计量判断不了，建议结合下面的 robosuite 惯例交叉验证")

    print(
        "\n[补充参考] robosuite/robomimic 的 obs 惯例：eef_quat 来自 "
        "robosuite.utils.transform_utils.mat2quat()，这个函数按官方实现返回的是 "
        "[x, y, z, w] 顺序（和 scipy.spatial.transform.Rotation 的惯例一致）。"
        "如果你用的 robosuite 版本没改过这个函数，代码里 quat_to_matrix 默认的 "
        "xyzw 假设大概率是对的；但不同 fork/版本可能有出入，建议以下面的自动检测"
        "（如果当前环境装了 robosuite）或上面的统计结果为准。"
    )
    try:
        import inspect

        import robosuite.utils.transform_utils as T
        src = inspect.getsource(T.mat2quat)
        print("\n[已在当前环境找到 robosuite，mat2quat 源码如下，可直接确认顺序]")
        print(src)
    except Exception as e:
        print(f"\n[提示] 当前环境没装 robosuite 或读取源码失败（{e}），跳过自动交叉验证。")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--hdf5", type=str, default="two_arm_box_cleanup.hdf5",
                         help="建议用双臂协作、动作明显的 panda 组数据集之一："
                              "two_arm_box_cleanup / two_arm_lift_tray / two_arm_drawer_cleanup")
    parser.add_argument("--demo", type=str, default=None, help="不填则自动取第一条 demo")
    parser.add_argument("--out_dir", type=str, default="./dexmg_verify_out")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    hdf5_path = os.path.join(args.dataset_root, args.hdf5)

    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
        demo = args.demo or demo_keys[0]
        print(f"数据集: {hdf5_path}")
        print(f"使用 demo: {demo}  (共 {len(demo_keys)} 条 demo)")

        check_left_right(f, demo, args.out_dir)
        check_quat_order(f, demo)


if __name__ == "__main__":
    main()
