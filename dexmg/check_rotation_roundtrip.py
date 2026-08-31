"""
check_rotation_roundtrip.py

对 TwoArmBoxCleanup (panda group) 的真实训练数据做
axis_angle -> rot6d -> axis_angle 往返一致性检查，
用旋转角误差（而不是分量 L2 距离）判断是否出问题，
并单独标记 Gram-Schmidt 重建时输入接近退化（两个 6D 子向量接近共线）的帧。

用法:
    python check_rotation_roundtrip.py \
        --hdf5 /path/to/two_arm_box_cleanup.hdf5 \
        --num_demos 5 --max_frames_per_demo 200
"""
import argparse
import numpy as np
import h5py

# === 按你实际的模块路径改这里 ===
from dexmg.dexmg_rotation import (
    axis_angle_to_rot6d,
    rot6d_to_axis_angle,
    rot6d_to_matrix,
    matrix_to_axis_angle,
)


def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """aa: (3,) axis*angle -> (3,3) rotation matrix, 用 Rodrigues 公式，
    只用来做误差评估的独立实现，不复用 dexmg_rotation.py，避免用
    同一套代码的 bug 掩盖同一套代码的 bug。"""
    theta = np.linalg.norm(aa)
    if theta < 1e-8:
        return np.eye(3)
    k = aa / theta
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0],
    ])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def geodesic_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """两个旋转矩阵之间的测地距离（角度误差），单位度。
    这是判断"往返后是不是同一个旋转"的正确度量，不受 axis-angle
    表示歧义（轴反向+2π-θ）影响。"""
    R = R1.T @ R2
    trace = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    return np.degrees(np.arccos(trace))


def gram_schmidt_degeneracy(rot6d: np.ndarray) -> float:
    """rot6d 前 3 维和后 3 维两个向量的夹角正弦值。
    越接近 0 说明两个向量越接近共线/退化，Gram-Schmidt 重建出的
    第二个基向量会非常不稳定，对模型输出里极小的扰动敏感——
    这是导致"手腕突然翻转"的一个常见结构性原因。"""
    a, b = rot6d[:3], rot6d[3:]
    a_n = a / (np.linalg.norm(a) + 1e-8)
    b_n = b / (np.linalg.norm(b) + 1e-8)
    cross_norm = np.linalg.norm(np.cross(a_n, b_n))
    return cross_norm  # 越接近 0 越退化


def check_one_segment(name: str, aa_seq: np.ndarray, angle_thresh_deg=1.0, degen_thresh=0.05):
    """aa_seq: (T, 3) 一个 demo 里某个旋转分量（如 right_rel_rot_axis_angle）
    随时间的序列。"""
    n_bad = 0
    n_degenerate = 0
    max_err = 0.0
    worst_frame = -1
    for t, aa in enumerate(aa_seq):
        rot6d = axis_angle_to_rot6d(aa)          # 正向转换（训练时用的那条路径）
        aa_recon = rot6d_to_axis_angle(rot6d)     # 反向转换（推理/eval 时用的那条路径）

        R_orig = axis_angle_to_matrix(aa)
        R_recon = axis_angle_to_matrix(aa_recon)
        err_deg = geodesic_angle_deg(R_orig, R_recon)

        degen = gram_schmidt_degeneracy(rot6d)

        if err_deg > angle_thresh_deg:
            n_bad += 1
        if degen < degen_thresh:
            n_degenerate += 1
        if err_deg > max_err:
            max_err = err_deg
            worst_frame = t

    print(f"  [{name}] frames={len(aa_seq)}  "
          f"err>{angle_thresh_deg}deg: {n_bad}  "
          f"near-degenerate(<{degen_thresh}): {n_degenerate}  "
          f"max_err={max_err:.3f}deg @frame{worst_frame}")
    return n_bad, n_degenerate, max_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True, help="TwoArmBoxCleanup 原始 demo hdf5 路径")
    ap.add_argument("--num_demos", type=int, default=5)
    ap.add_argument("--max_frames_per_demo", type=int, default=200)
    ap.add_argument("--angle_thresh_deg", type=float, default=1.0,
                     help="超过这个角度误差算“往返失败”")
    args = ap.parse_args()

    # 直接读原始 action_dict，绕开整条 pipeline，只测 rotation 转换本身
    segments = ["right_rel_rot_axis_angle", "left_rel_rot_axis_angle"]

    with h5py.File(args.hdf5, "r") as f:
        demo_ids = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))[:args.num_demos]
        print(f"Checking {len(demo_ids)} demos from {args.hdf5}\n")

        totals = {s: [0, 0, 0.0] for s in segments}  # n_bad, n_degenerate, max_err

        for demo_id in demo_ids:
            print(f"demo={demo_id}")
            for seg in segments:
                key = f"data/{demo_id}/action_dict/{seg}"
                if key not in f:
                    print(f"  [WARN] key not found: {key}")
                    continue
                aa_seq = f[key][:args.max_frames_per_demo]
                n_bad, n_degen, max_err = check_one_segment(seg, aa_seq, args.angle_thresh_deg)
                totals[seg][0] += n_bad
                totals[seg][1] += n_degen
                totals[seg][2] = max(totals[seg][2], max_err)

        print("\n=== 汇总 ===")
        for seg, (n_bad, n_degen, max_err) in totals.items():
            flag = "  <-- 建议重点排查" if (n_bad > 0 or max_err > args.angle_thresh_deg * 3) else ""
            print(f"{seg}: total_bad_frames={n_bad}  total_near_degenerate={n_degen}  "
                  f"global_max_err={max_err:.3f}deg{flag}")


if __name__ == "__main__":
    main()
