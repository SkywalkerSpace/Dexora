# -*- coding: utf-8 -*-
"""
dexmg_rotation.py

纯粹的旋转"格式"转换（axis-angle -> 6D, quaternion -> 6D），无损、
确定性，不涉及"相对位姿 vs 绝对位姿"这种物理语义转换（那个是我们
明确决定不做的）。

6D 旋转表示（Zhou et al. 2019, "On the Continuity of Rotation
Representations..."）：取旋转矩阵 R (3x3) 的前两列拼成 (6,)。
这里只做单向转换（写入 unified schema 用），不需要 6D -> 矩阵 的
反向重建（Gram-Schmidt），因为下游只在推理时才需要把模型输出的
6D 转回 env 需要的格式，那部分留在部署脚本里做，不属于这次数据
读取层的范围。
"""

from __future__ import annotations

import numpy as np


def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """aa: (..., 3) axis-angle (方向=轴，模长=角度，弧度)。返回 (..., 3, 3) 旋转矩阵。

    Rodrigues 公式，纯 numpy 实现，支持任意 batch 维度。
    """
    aa = np.asarray(aa, dtype=np.float64)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)  # (..., 1)
    eps = 1e-8
    axis = aa / np.clip(theta, eps, None)  # (..., 3)

    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zeros = np.zeros_like(x)
    K = np.stack([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros,
    ], axis=-1).reshape(*aa.shape[:-1], 3, 3)  # 反对称矩阵

    theta_ = theta[..., None]  # (..., 1, 1)
    I = np.eye(3, dtype=np.float64)
    I = np.broadcast_to(I, K.shape)
    R = I + np.sin(theta_) * K + (1 - np.cos(theta_)) * (K @ K)

    # theta ~ 0 时（无旋转），Rodrigues 公式数值上退化为单位阵，上面已经自然处理，
    # 但 axis 归一化时的 0/0 需要显式兜底：
    small = (theta[..., 0] < eps)
    if np.any(small):
        R[small] = np.eye(3, dtype=np.float64)
    return R


def quat_to_matrix(quat: np.ndarray, order: str = "xyzw") -> np.ndarray:
    """quat: (..., 4)。返回 (..., 3, 3)。

    robosuite/robomimic 的 obs 里四元数通常是 (x, y, z, w) 顺序，
    这里默认按这个假设写；如果实际数据是 (w, x, y, z)，把 order 改成 "wxyz"，
    或者拿到数据后自己核实一下（比如检查静止姿态下 w 分量是否接近 1）。
    """
    quat = np.asarray(quat, dtype=np.float64)
    if order == "xyzw":
        x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    elif order == "wxyz":
        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    else:
        raise ValueError(f"未知的四元数顺序: {order}")

    norm = np.sqrt(x * x + y * y + z * z + w * w)
    eps = 1e-8
    x, y, z, w = x / np.clip(norm, eps, None), y / np.clip(norm, eps, None), \
        z / np.clip(norm, eps, None), w / np.clip(norm, eps, None)

    R = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(*quat.shape[:-1], 3, 3)
    return R


def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """R: (..., 3, 3) -> (..., 6)，取前两列拼接。"""
    R = np.asarray(R, dtype=np.float64)
    col0 = R[..., :, 0]
    col1 = R[..., :, 1]
    return np.concatenate([col0, col1], axis=-1).astype(np.float32)


def axis_angle_to_rot6d(aa: np.ndarray) -> np.ndarray:
    return matrix_to_rot6d(axis_angle_to_matrix(aa))


def quat_to_rot6d(quat: np.ndarray, order: str = "xyzw") -> np.ndarray:
    return matrix_to_rot6d(quat_to_matrix(quat, order=order))


# ============================================================
# 反方向：6D -> 旋转矩阵 -> axis-angle
# 训练时只用到了正向转换（写统一 schema），这里补上推理时需要的
# 反向转换（模型输出 6D，panda 组 env.step 需要 axis-angle）。
# ============================================================

def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """rot6d: (..., 6) -> (..., 3, 3)，标准 Gram-Schmidt 重建（Zhou et al. 2019）。"""
    rot6d = np.asarray(rot6d, dtype=np.float64)
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]

    eps = 1e-8
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), eps, None)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), eps, None)
    b3 = np.cross(b1, b2)

    R = np.stack([b1, b2, b3], axis=-1)  # 列向量拼成矩阵
    return R


def matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """R: (..., 3, 3) -> (..., 3) axis-angle。"""
    R = np.asarray(R, dtype=np.float64)
    # theta 由迹算出来
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)  # (...,)

    # 反对称部分提取旋转轴
    rx = R[..., 2, 1] - R[..., 1, 2]
    ry = R[..., 0, 2] - R[..., 2, 0]
    rz = R[..., 1, 0] - R[..., 0, 1]
    axis_unnorm = np.stack([rx, ry, rz], axis=-1)  # (..., 3)

    sin_theta = np.sin(theta)
    eps = 1e-8
    small = sin_theta < eps

    axis = np.zeros_like(axis_unnorm)
    denom = np.clip(2.0 * sin_theta, eps, None)[..., None]
    axis = axis_unnorm / denom

    aa = axis * theta[..., None]
    # theta ~ 0（无旋转）时上面的除法数值不稳定，直接置零向量
    if np.any(small):
        aa[small] = 0.0
    return aa.astype(np.float32)


def rot6d_to_axis_angle(rot6d: np.ndarray) -> np.ndarray:
    return matrix_to_axis_angle(rot6d_to_matrix(rot6d))
