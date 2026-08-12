# dexmimicgen 直读 hdf5 方案（方案 B 重写版：RDT-1B 风格共享物理槽位，不是按 embodiment 各占一段）

## 文件说明

| 文件 | 作用 |
|---|---|
| `dexmg_config.py` | 六个 hdf5 的原始相机/状态/动作 key 配置表（唯一真源，抄自官方 `generate_training_config.py`） |
| `dexmg_rotation.py` | 纯旋转格式转换：axis-angle/quaternion 统一转成 6D（无损，不涉及 rel/abs 语义转换） |
| `dexmg_schema.py` | 统一 schema：**按物理含义共享槽位**（right_arm_pos/right_arm_rot6d/right_gripper/left_arm_pos/left_arm_rot6d/left_gripper），panda组和humanoid组写进同一套槽位，不是各占一段；state 的 gripper 槽位宽度探测 hdf5 动态算出并缓存 |
| `dexmg_camera.py` | 相机 key → Dexora 4 路 canonical 相机槽位的映射，处理 `cam_high` 缺失补零 |
| `dexmg_convert.py` | 把 robomimic 的 action_dict/obs_dict 按物理含义写进共享槽位 + 生成 mask |
| `dexmg_hdf5_vla_dataset.py` | 真正的 Dataset 类 `DexmgHDF5VLADataset`，满足 Dexora `VLAConsumerDataset` 要求的 `get_item()`/`__len__()` 契约 |
| `compute_dexmg_stats.py` | 训练前离线跑一次的统计量脚本；槽位共享但数值语义不同，**统计量依然按 group 分别算** |

## 维度（比上一版的 54 维小，且两组共享，不是各占一段）

- `ACTION_DIM = 30`：`right_arm_pos(3) / right_arm_rot6d(6) / right_gripper(6) / left_arm_pos(3) / left_arm_rot6d(6) / left_gripper(6)`。panda 组的 axis-angle 转成 6D 写进同一槽位，humanoid 组本来就是 6D 直接写。两组当前都是满槽位，`action_mask` 恒为全 1（为以后接入缺自由度的新 embodiment 保留这个接口）。
- `STATE_DIM`：槽位结构同上（pos=3, rot6d=6 固定），只有 `right_gripper`/`left_gripper` 两个槽位的宽度是探测出来的（两组里较大的真实宽度），较窄那组会 padding+mask=0。缓存进 `dexmg_unified_schema_cache.json`。
- **旋转格式**：quaternion（state）/axis-angle（panda action）都转成 6D；这是无损格式转换，和"要不要把 abs 转成 rel"（我们明确决定不做）是两回事。
- **rel vs abs 语义差异**：不用额外维度区分，靠每个数据集独立的归一化统计量 + RDT 自带的 dataset/instruction conditioning 让模型隐式学会区分。

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

3. **模型侧**：`RDTRunner` 的 `state_dim`/`action_dim` 改成新的 `STATE_DIM`/`ACTION_DIM=30`（`state_adaptor`/`final_layer.ffn_final` 会随之重新初始化，符合你们之前确认过的"只有这两层需要按新维度重建，其余 28 层 backbone 不受影响"的结论）。
4. **`compute_loss` 调用处**：把 `_SingleDexmgReader.get_item()` 里新增的 `action_mask` 字段传给 `RDTRunner.compute_loss(..., action_mask=...)`（现有代码大概率已经在传某种 action_mask，确认一下是不是直接复用这个字段，还是需要在 collator 里再包一层）。

## 待你核实/决定的点（代码里都留了标记）

- `dexmg_camera.py`：panda 组 `robot0_eye_in_hand_image`/`robot1_eye_in_hand_image` 到底对应右手腕还是左手腕，目前按 `robot0=right` 假设写的，建议跑一条 demo 目视核实。
- `compute_dexmg_stats.py` 里 `state_norm` 目前直接等于 `state_std`，如果 Dexora 下游对 `state_norm` 有别的定义（比如是每个 episode 自己的统计量而不是数据集级别的），需要调整。
- `filter_keys`（demo split json）路径没有硬编码，按你实际的 split 文件传进去。
