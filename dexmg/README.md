# dexmimicgen 直读 hdf5 方案（方案 B：统一 schema + mask，不做表示转换）

## 文件说明

| 文件 | 作用 |
|---|---|
| `dexmg_config.py` | 六个 hdf5 的原始相机/状态/动作 key 配置表（唯一真源，抄自官方 `generate_training_config.py`） |
| `dexmg_schema.py` | 统一 (superset) state/action schema：action 固定 54 维（panda 24 + humanoid 30 两段槽位），state 维度靠探测 hdf5 动态算出并缓存 |
| `dexmg_camera.py` | 相机 key → Dexora 4 路 canonical 相机槽位的映射，处理 `cam_high` 缺失补零 |
| `dexmg_convert.py` | 把 robomimic 的 action_dict/obs_dict 按 schema 拼进 superset 向量 + 生成 mask |
| `dexmg_hdf5_vla_dataset.py` | 真正的 Dataset 类 `DexmgHDF5VLADataset`，满足 Dexora `VLAConsumerDataset` 要求的 `get_item()`/`__len__()` 契约 |
| `compute_dexmg_stats.py` | 训练前离线跑一次的统计量脚本，按 group 分别算，避免互相污染 |

## 维度

- `ACTION_DIM = 54`：`[0:24]` 是 panda 组的 right_arm/right_gripper/left_arm/left_gripper（相对位姿+axis-angle），`[24:54]` 是 humanoid 组的 right_arm/left_arm/right_gripper/left_gripper（绝对位姿+rot_6d）。panda 组样本只有 `[0:24]` 非零，humanoid 组样本只有 `[24:54]` 非零，`action_mask` 标出哪段有效。
- `STATE_DIM`：运行 `build_state_schema()` 时探测得到（依赖 `robot0_gripper_qpos` 等 key 的实际宽度），缓存进 `dexmg_state_schema_cache.json`。

## 使用步骤

1. **首次探测 state 维度 + 算统计量**（同一个命令会先探测 schema 再扫全部 hdf5）：

   ```bash
   python compute_dexmg_stats.py \
       --dataset_root /path/to/dexmimicgen/datasets/generated \
       --out_dir configs/
   ```

   这一步会生成/更新：
   - `configs/dexmg_state_schema_cache.json`（state schema 缓存）
   - `configs/dataset_statistics.json`（增量合并进已有的）
   - `configs/dataset_stat_ours.json`（增量合并进已有的）

2. **接入 `Dexora-dataset.py`**：在 `VLAConsumerDataset.__init__` 里加一个分支（其余代码不用动）：

   ```python
   elif use_hdf5 == "dexmg_hdf5":
       from data.dexmg_hdf5_vla_dataset import DexmgHDF5VLADataset
       self.hdf5_dataset = DexmgHDF5VLADataset(
           dataset_root="/path/to/dexmimicgen/datasets/generated",
           stats_file="configs/dataset_statistics.json",
           filter_keys={
               "two_arm_box_cleanup.hdf5": "1000_demos",
               # ... 其余数据集的 demo split，不填则用全部 demo
           },
           schema_cache_dir="configs/",
       )
   ```

   然后训练时用 `use_hdf5="dexmg_hdf5"` 启动即可。

3. **模型侧**：`RDTRunner` 的 `state_dim`/`action_dim` 改成新的 `STATE_DIM`/`ACTION_DIM=54`（这两个数字变大了，`state_adaptor`/`final_layer.ffn_final` 会随之重新初始化，符合你们之前确认过的"只有这两层需要按新维度重建，其余 28 层 backbone 不受影响"的结论）。
4. **`compute_loss` 调用处**：把 `_SingleDexmgReader.get_item()` 里新增的 `action_mask` 字段传给 `RDTRunner.compute_loss(..., action_mask=...)`（现有代码大概率已经在传某种 action_mask，确认一下是不是直接复用这个字段，还是需要在 collator 里再包一层）。

## 待你核实/决定的点（代码里都留了标记）

- `dexmg_camera.py`：panda 组 `robot0_eye_in_hand_image`/`robot1_eye_in_hand_image` 到底对应右手腕还是左手腕，目前按 `robot0=right` 假设写的，建议跑一条 demo 目视核实。
- `compute_dexmg_stats.py` 里 `state_norm` 目前直接等于 `state_std`，如果 Dexora 下游对 `state_norm` 有别的定义（比如是每个 episode 自己的统计量而不是数据集级别的），需要调整。
- `filter_keys`（demo split json）路径没有硬编码，按你实际的 split 文件传进去。
