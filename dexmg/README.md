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

## 维度（state_dim 强制等于 action_dim，因为 Dexora 的实例化代码就是这么用的）

**重要约束**：Dexora 实际训练脚本里 `RDTRunner(action_dim=config["common"]["state_dim"], ...)`——`action_dim` 和 `state_dim` 天生是同一个数字 M，不是两个独立维度。这版把 gripper 槽位宽度改成**四个探测结果（state×2组 + action×2组）里的最大值**，state 和 action 强制共用同一套槽位布局：

```
right_arm_pos(3) / right_arm_rot6d(6) / right_gripper(探测最大值) /
left_arm_pos(3)  / left_arm_rot6d(6)  / left_gripper(探测最大值)
```

以你们实际数据为例：state 侧 gripper 真实宽度 panda=12/humanoid=11（GR1/fourier 手 raw qpos 11维），action 侧 gripper 真实宽度两组都是 6 —— 取四者最大值 12，`M = 3+6+12+3+6+12 = 42`。也就是说 **action 侧原本只有 6 维真实数据，会 padding 到 12 维**，多出来的 6 维在 `action_mask` 里标为 0（跑 mock 数据验证过：panda 组 `action_mask` 有效维度 30/42，humanoid 组 `state_mask` 有效维度 40/42，符合预期）。

- `dim`：单一的 M，`DexmgHDF5VLADataset.state_dim == .action_dim == schema.dim`
- **旋转格式**：quaternion（state）/axis-angle（panda action）都转成 6D；rel vs abs 语义差异不占维度，靠各自数据集独立的归一化统计量 + RDT 的 dataset/instruction conditioning 隐式区分。

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

3. **模型侧**：实例化 `DexmgHDF5VLADataset` 后用 `.state_dim`（等于 `.action_dim`）填 `config["common"]["state_dim"]` 和 `config["model"]["state_token_dim"]`，`RDTRunner` 构造时 `action_dim=config["common"]["state_dim"]` 这行不用改，两者本来就该是同一个数。
4. **`compute_loss` 调用处**：把 `_SingleDexmgReader.get_item()` 里新增的 `action_mask` 字段传给 `RDTRunner.compute_loss(..., action_mask=...)`（现有代码大概率已经在传某种 action_mask，确认一下是不是直接复用这个字段，还是需要在 collator 里再包一层）。

## 待你核实/决定的点（代码里都留了标记）

- `dexmg_camera.py`：panda 组 `robot0_eye_in_hand_image`/`robot1_eye_in_hand_image` 到底对应右手腕还是左手腕，目前按 `robot0=right` 假设写的，建议跑一条 demo 目视核实。
- `compute_dexmg_stats.py` 里 `state_norm` 目前直接等于 `state_std`，如果 Dexora 下游对 `state_norm` 有别的定义（比如是每个 episode 自己的统计量而不是数据集级别的），需要调整。
- `filter_keys`（demo split json）路径没有硬编码，按你实际的 split 文件传进去。
