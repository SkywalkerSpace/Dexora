# 分析 Dexora 项目训练流程

> 读 /Users/skywalker/code/experiment/Dexora/README.md， /Users/skywalker/code/experiment/Dexora/*.sh, /Users/skywalker/code/experiment/Dexora/scripts/* 和其他有关py代码，这个项目做了什么，如何3阶段训练，训练的模型架构，模型每个部分输入输出是什么。结合代码解释 采集数据，训练，测试，机械手臂测试 都怎么做

<details><summary>5 previous messages</summary>

> 我会先梳理 README、Shell 入口、scripts 目录和 Python 模块的调用关系，再沿着“数据采集 → 三阶段训练 → 离线测试 → 机械臂测试”的链路核对输入输出与模型结构，最后按代码给你一份中文说明。
>
> <details><summary>Explored 5 files, a search, ran a command</summary>
>
> - Ran `printf '%s\n' '--- files ---'; rg --files Dexora | sort; printf '%s\n' '--- readme ---'; sed -n '1,260p' Dexora/README.md; printf '%s\n' '--- shell files ---'; for f in Dexora/*.sh Dexora/scripts/*.sh; do [ -f "$f" ] && { echo "### $f"; sed -n '1,240p' "$f"; }; done`
>
> <details><summary>Explored 5 files, a search</summary>
>
> - Searched for `^##|^###|Stage|stage|train|teleop|inference|dataset` in `README.md`
> - Read `./Dexora/configs/base_400m.yaml`
> - Read `./Dexora/models/rdt/model.py`
> - Read `./Dexora/models/rdt/blocks.py`
> - Read `./Dexora/train/dataset.py`
> - Read `./Dexora/data/lerobot_vla_dataset.py`
>
> </details>
>
> </details>
>
> 目前主线已经清楚：公开数据是 LeRobot v2.1，策略默认只控制 36 DoF；三阶段中的“阶段 2”实际包含质量预筛、回放验证、log-π 计算和 PU 判别器训练四个子步骤。我继续把这些步骤在 Python 中如何落地，以及部署时各进程的消息格式补齐。
>
> <details><summary>Explored 13 files, a search</summary>
>
> - Searched for `^class |def |compute_loss|forward|action|state|image|lang|scor|weight|logpi` in `rdt_runner.py`
> - Read `./Dexora/deploy/README.md`
> - Read `./Dexora/deploy/dexora_policy.py`
> - Read `./Dexora/deploy/dexora_inference_zmq.py`
> - Read `./Dexora/teleop/README.md`
> - Read `./Dexora/teleop/scripts/record_delete.py`
> - Read `./Dexora/teleop/imitate_all/record_4_rgb_cam.py`
> - Read `./Dexora/teleop/teleop_pkg/receive_from_vision_pro.py`
> - Read `./Dexora/deploy/dexora_policy.py`
> - Read `./Dexora/deploy/dexora_inference_zmq.py`
> - Read `./Dexora/models/rdt_runner.py`
> - Read `./Dexora/models/scoring_model.py`
> - Read `./Dexora/scripts/analyze_episode_quality.py`
> - Read `./Dexora/scripts/compute_logpi.py`
>
> </details>
>
> 这个项目是 Dexora：一个面向双臂、双灵巧手机器人的视觉-语言-动作模型（VLA）。它把四路相机图像、自然语言任务描述和当前机器人状态作为输入，输出未来一段时间的 36 维关节动作。
>
> 核心控制对象是：
>
> ```text
> [left_arm 6]
> [right_arm 6]
> [left_hand 12]
> [right_hand 12]
> = 36 DoF
> ```
>
> 公开 LeRobot 数据中实际保存 39 维，多出的最后 3 维是：
>
> ```text
> head_joint_1, head_joint_2, spine_joint
> ```
>
> 代码默认通过 `state_dim_keep=36` 去掉它们。
>
> ---
>
> ## 一、项目整体流程
>
> 完整流程由 `run_all_stages.sh` 串起来：
>
> ```text
> 数据采集/转换
>     ↓
> LeRobot v2.1 数据集
>     ↓
> Stage 1：训练基础 Diffusion Transformer 策略
>     ↓
> Stage 2a：根据加速度、加加速度筛选轨迹
>     ↓
> Stage 2b：回放验证，筛出高质量轨迹
>     ↓
> Stage 2c-1：用 Stage-1 策略计算 log-π proxy
>     ↓
> Stage 2c-2：训练质量判别器
>     ↓
> Stage 3：用判别器给训练样本加权，继续训练策略
>     ↓
> 离线 open-loop 测试
>     ↓
> 真实机械臂闭环执行
> ```
>
> 对应的 Shell 文件是：
>
> ```text
> s1_pretrain.sh
> s2a_analyze_jerk.sh
> s2b_replay.sh
> s2c_compute_logpi.sh
> s2c_train_scoring.sh
> s3_post_train.sh
> ```
>
> 严格来说，项目所说的“三阶段训练”是：
>
> 1. Stage 1：策略预训练
> 2. Stage 2：质量判别器训练
> 3. Stage 3：质量感知的策略后训练
>
> 其中 Stage 2 又拆成 2a、2b、2c 三个数据质量处理步骤。
>
> ---
>
> # 二、模型架构
>
> ## 1. 输入
>
> 运行时 `deploy/dexora_policy.py` 的 `get_action(obs)` 要求输入：
>
> ```python
> obs = {
>     "state": np.ndarray,       # [36]
>     "images": {
>         "cam_head": ...,
>         "cam_left_wrist": ...,
>         "cam_third_view": ...,
>         "cam_right_wrist": ...
>     },
>     "instruction": str,
>     "ctrl_freq": 20.0
> }
> ```
>
> 具体输入含义：
>
> | 输入 | 原始形式 | 作用 |
> |---|---:|---|
> | 当前状态 | `[36]` | 当前两臂和两手关节位置 |
> | 四路图像 | `[H,W,3]` | 观察环境和物体 |
> | 语言指令 | 字符串 | 例如 “Pick the apple and put it on the plate.” |
> | 控制频率 | 标量，通常 20 Hz | 告知模型控制时间尺度 |
> | 动作监督 | `[32,36]` | 训练时使用，未来 32 帧的真实动作 |
>
> 动作 chunk 长度由 `configs/base_400m.yaml` 指定：
>
> ```yaml
> action_chunk_size: 32
> state_dim: 36
> num_cameras: 4
> img_history_size: 1
> ```
>
> 所以策略每次预测未来 32 个控制周期的动作。
>
> ---
>
> ## 2. 语言编码器：T5
>
> 默认模型：
>
> ```text
> google/t5-v1_1-xxl
> ```
>
> 在 `deploy/dexora_policy.py` 中，语言指令经过 `T5Embedder`，输出：
>
> ```text
> 语言 token：
> [B, language_length, 512]
> ```
>
> 然后经过 `RDTRunner.lang_adaptor`：
>
> ```python
> self.lang_adaptor = ...
> ```
>
> 把 512 维语言 token 映射到 Transformer 的隐藏维度：
>
> ```text
> [B, language_length, 512]
>         ↓ MLP
> [B, language_length, 1024]
> ```
>
> 配置中：
>
> ```yaml
> lang_token_dim: 512
> lang_adaptor: mlp2x_gelu
> tokenizer_max_length: 1024
> ```
>
> 语言编码结果只在任务描述变化时重新计算，部署代码中会缓存：
>
> ```python
> self._cached_instruction
> self._cached_lang_tokens
> self._cached_lang_mask
> ```
>
> ---
>
> ## 3. 图像编码器：SigLIP
>
> 默认模型：
>
> ```text
> google/siglip-so400m-patch14-384
> ```
>
> 每张 384×384 图像被 SigLIP 编码成视觉 patch token。
>
> 配置：
>
> ```yaml
> num_patches_per_view: 729
> img_token_dim: 1152
> num_cameras: 4
> ```
>
> 729 个 patch 等于：
>
> ```text
> 27 × 27 = 729
> ```
>
> 四个相机、每个相机一帧，因此视觉条件大致是：
>
> ```text
> 4 × 729 = 2916 个视觉 token
> [B, 2916, 1152]
> ```
>
> 然后通过：
>
> ```python
> self.img_adaptor
> ```
>
> 投影到 RDT 的隐藏维度：
>
> ```text
> [B, 2916, 1152]
>         ↓ MLP
> [B, 2916, 1024]
> ```
>
> 四路相机的顺序在 `dexora_policy.py` 中固定为：
>
> ```python
> (
>     "cam_head",
>     "cam_left_wrist",
>     "cam_third_view",
>     "cam_right_wrist",
> )
> ```
>
> 数据集中的相机名称则被 `data/lerobot_vla_dataset.py` 映射：
>
> ```text
> top          → cam_high
> wrist_left   → cam_left_wrist
> wrist_right  → cam_right_wrist
> front        → cam_third_view
> ```
>
> ---
>
> ## 4. 状态编码器
>
> 当前状态是 36 维：
>
> ```text
> [左臂 6 | 右臂 6 | 左手 12 | 右手 12]
> ```
>
> 训练时，状态和动作都带一个维度 mask，因此状态适配器输入维度是：
>
> ```python
> state_token_dim * 2
> ```
>
> 也就是：
>
> ```text
> 36 个状态值 + 36 个 mask 值 = 72
> ```
>
> 经过：
>
> ```yaml
> state_adaptor: mlp3x_gelu
> ```
>
> 映射到：
>
> ```text
> [B, 1, 1024]
> ```
>
> 代码位置：
>
> ```python
> RDTRunner.adapt_conditions()
> ```
>
> ---
>
> ## 5. RDT Diffusion Transformer
>
> 核心模型在：
>
> ```text
> models/rdt/model.py
> models/rdt/blocks.py
> models/rdt_runner.py
> ```
>
> 400M 配置为：
>
> ```yaml
> hidden_size: 1024
> depth: 28
> num_heads: 16
> ```
>
> 也就是：
>
> ```text
> 28 层 Transformer block
> 16 个 attention heads
> 隐藏维度 1024
> 约 400M 参数
> ```
>
> RDT 内部输入序列大致是：
>
> ```text
> [timestep token]
> [control frequency token]
> [current state token]
> [action token 1]
> [action token 2]
> ...
> [action token 32]
> ```
>
> 实际在 `RDT.forward()` 中：
>
> ```python
> x = torch.cat([t, freq, x], dim=1)
> ```
>
> 其中：
>
> - `t`：扩散时间步
> - `freq`：控制频率
> - `x`：当前状态和带噪动作序列
>
> 语言 token 和视觉 token 作为条件序列输入 Transformer：
>
> ```python
> conds = [lang_c, img_c]
> ```
>
> 因此模型结构可以概括为：
>
> ```text
> T5 ────────┐
>            ├─> 条件 token ────────┐
> SigLIP ────┘                      │
>                                   ↓
> 状态 ─> State Adapter ───────> RDT Transformer ─> 36维动作噪声
> 带噪动作 ─> State Adapter ────┘
> 时间步 ─> Timestep Embedding ──┘
> 频率 ─> Frequency Embedding ──┘
> ```
>
> ---
>
> ## 6. Diffusion 训练输入输出
>
> 训练时，真实动作记为：
>
> ```text
> a₀: [B, 32, 36]
> ```
>
> 代码随机采样噪声：
>
> ```python
> noise = torch.randn_like(action_gt)
> ```
>
> 然后根据随机 diffusion timestep 加噪：
>
> ```python
> noisy_action = noise_scheduler.add_noise(
>     action_gt, noise, timesteps
> )
> ```
>
> 模型输入：
>
> ```text
> 当前状态
> 带噪动作序列
> 语言条件
> 图像条件
> 扩散时间步
> 控制频率
> ```
>
> 模型输出：
>
> ```text
> pred: [B, 32, 36]
> ```
>
> 默认 `prediction_type: epsilon`，因此输出的是预测噪声：
>
> ```text
> ε̂θ
> ```
>
> 训练目标是实际加入的噪声：
>
> ```text
> ε
> ```
>
> 损失：
>
> ```text
> MSE(ε̂θ, ε)
> ```
>
> 对应 `RDTRunner.compute_loss()`：
>
> ```python
> pred = self.model(...)
> target = noise
> loss = weighted_mse_loss(pred, target, ...)
> ```
>
> 推理时则从纯随机噪声开始，用 DPMSolver++ 逐步去噪，得到：
>
> ```text
> [B, 32, 36]
> ```
>
> 默认只使用 5 个 inference steps：
>
> ```yaml
> num_inference_timesteps: 5
> ```
>
> ---
>
> # 三、数据格式和数据采集
>
> ## 1. 最终训练数据格式
>
> 训练使用 LeRobot v2.1 数据集，典型结构：
>
> ```text
> airbot_pick_and_place/
> ├── data/
> │   └── chunk-000/episode_000000.parquet
> ├── videos/
> │   └── chunk-000/
> │       ├── observation.images.top/
> │       ├── observation.images.wrist_left/
> │       ├── observation.images.wrist_right/
> │       └── observation.images.front/
> └── meta/
>     ├── info.json
>     ├── episodes.jsonl
>     ├── tasks.jsonl
>     └── modality.json
> ```
>
> 每帧主要字段：
>
> ```text
> observation.state
> action
> observation.images.top
> observation.images.wrist_left
> observation.images.wrist_right
> observation.images.front
> episode_index
> frame_index
> ```
>
> `data/lerobot_vla_dataset.py` 负责：
>
> 1. 读取 LeRobot 数据；
> 2. 读取 episode 对应任务指令；
> 3. 读取当前状态；
> 4. 读取未来 32 帧动作；
> 5. 读取四个相机；
> 6. 将 39 维切成 36 维；
> 7. 用 `dataset_statistics.json` 做归一化；
> 8. 映射成训练框架所需的旧式字段名。
>
> 最终训练样本类似：
>
> ```python
> {
>     "meta": {
>         "episode_idx": ...,
>         "step_id": ...,
>         "instruction": ...
>     },
>     "state": [36],
>     "actions": [32, 36],
>     "cam_high": [1, H, W, 3],
>     "cam_left_wrist": [1, H, W, 3],
>     "cam_right_wrist": [1, H, W, 3],
>     "cam_third_view": [1, H, W, 3],
> }
> ```
>
> 动作和状态使用每维 min-max statistics 归一化。统计文件不存在时，`s1_pretrain.sh` 会调用：
>
> ```bash
> python -m data.lerobot_vla_dataset --stat
> ```
>
> 生成统计数据。
>
> ---
>
> ## 2. 真实数据采集
>
> 真实数据采集入口是：
>
> ```text
> teleop/scripts/record_delete.py
> ```
>
> 这个脚本同时启动两个子进程。
>
> ### 机器人和相机进程
>
> ```text
> teleop/imitate_all/record_4_rgb_cam.py
> ```
>
> 它负责：
>
> - 读取 AIRBOT 双臂机器人状态；
> - 读取四个 RGB 相机；
> - 以 20 Hz 采样；
> - 记录机器人 observation；
> - 保存图像；
> - 保存机械臂动作和状态；
> - 输出 BSON 或 raw episode 数据。
>
> 默认参数包括：
>
> ```python
> --fps 20
> --num-frames-per-episode 1000
> --num-episodes 10000
> ```
>
> 四个相机设备：
>
> ```python
> camera_left_wrist  -> /dev/camera_left
> camera_right_wrist -> /dev/camera_right
> camera_third_view  -> /dev/camera_high
> camera_head        -> /dev/camera_head
> ```
>
> 图像由 `ImageSaver` 异步写入，避免相机采集线程被磁盘写入阻塞。
>
> ### Vision Pro 灵巧手进程
>
> ```text
> teleop/teleop_pkg/receive_from_vision_pro.py
> ```
>
> 它负责：
>
> 1. 从 Apple Vision Pro 获取手部骨架；
> 2. 通过 `XHandTeleOps.get_data_from_visionpro()` 读取数据；
> 3. 通过 `retarget_data()` 映射成 XHand 的 12 关节；
> 4. 驱动左右灵巧手；
> 5. 读取真实手部观测；
> 6. 把动作和观测以 20 Hz 写入 `xhand_control_data.bson`。
>
> 其单帧记录类似：
>
> ```python
> {
>     "t": timestamp,
>     "action": {
>         "left_hand": [12],
>         "right_hand": [12]
>     },
>     "observation": {
>         "left_hand": [12],
>         "right_hand": [12]
>     }
> }
> ```
>
> ### episode 归档
>
> 两个进程都结束后，`record_delete.py` 会把：
>
> ```text
> episode_0/
> episode_0.bson
> xhand_control_data.bson
> ```
>
> 复制到：
>
> ```text
> ARCHIVE_ROOT/episode_N/
> ```
>
> 然后删除源文件。
>
> 注意：这个脚本中的路径是机器相关的，例如：
>
> ```python
> ROBOT_PYTHON_PATH = "/home/slam/miniconda3/envs/imitall/bin/python"
> HAND_PYTHON_PATH  = "/home/slam/miniconda3/envs/xhand_tele_env/bin/python"
> ARCHIVE_ROOT = "/media/slam/data/action6"
> ```
>
> 需要在实际机器上修改。
>
> ---
>
> ## 3. BSON 转 LeRobot
>
> 采集结果不是直接训练的，通常要经过：
>
> ```text
> dataprocess/airbot_lerobot.py
> ```
>
> 以及 `dataprocess/README.md` 中的转换流程，将：
>
> ```text
> 机械臂 BSON
> 灵巧手 BSON
> 相机图像
> 任务描述
> ```
>
> 合并成：
>
> ```text
> LeRobot v2.1
> ```
>
> 转换后的字段统一为：
>
> ```text
> observation.state
> action
> observation.images.*
> ```
>
> 这样才能被 `LeRobotVLADataset` 和 `s1_pretrain.sh` 读取。
>
> ---
>
> # 四、Stage 1：策略预训练
>
> 入口：
>
> ```text
> s1_pretrain.sh
> train/main.py
> train/train.py
> ```
>
> 典型启动：
>
> ```bash
> DEXORA_LEROBOT_ROOT=data/Dexora_Real-World_Dataset/airbot_pick_and_place \
> OUTPUT_DIR=checkpoints/dexora-400m-pretrain \
> bash s1_pretrain.sh
> ```
>
> 默认配置：
>
> ```text
> 模型：400M RDT
> 训练步数：100000
> batch size：1
> 梯度累积：1
> 学习率：1e-4
> 精度：bf16
> 动作 horizon：32
> 状态维度：36
> 控制频率：20 Hz
> ```
>
> 训练步骤：
>
> ```text
> LeRobotDataset 取样
>     ↓
> 读取状态、未来 32 帧动作、四路图像、语言
>     ↓
> 归一化状态和动作
>     ↓
> T5 编码语言
>     ↓
> SigLIP 编码四路图像
>     ↓
> 动作加 diffusion noise
>     ↓
> RDT 预测噪声
>     ↓
> 与真实噪声做 MSE
>     ↓
> 反向传播更新 RDT 和 adapter
> ```
>
> 输出 checkpoint：
>
> ```text
> checkpoints/dexora-400m-pretrain/
> ├── checkpoint-*/
> │   ├── pytorch_model.bin
> │   ├── config.json
> │   └── ema/
> ```
>
> Stage 1 训练的是基础行为策略，没有根据演示质量加权。
>
> ---
>
> # 五、Stage 2：质量判别器
>
> Stage 2 的目标不是直接训练动作策略，而是找出高质量演示，并训练一个评分模型。
>
> ## Stage 2a：加速度和 jerk 筛选
>
> 入口：
>
> ```text
> s2a_analyze_jerk.sh
> scripts/analyze_episode_quality.py
> ```
>
> 每个 episode 的状态序列：
>
> ```text
> s₀, s₁, ..., s_T
> ```
>
> 在归一化空间中计算：
>
> ```text
> 加速度：
> a_t ≈ s_{t+1} - 2s_t + s_{t-1}
>
> jerk：
> j_t ≈ 三阶差分
> ```
>
> 然后计算 episode 级别 RMS：
>
> ```text
> A_ep = acceleration RMS
> J_ep = jerk RMS
> ```
>
> 代码默认：
>
> ```bash
> FPS=20
> TARGET_RATIO=0.2
> STATE_DIM_KEEP=36
> ```
>
> 保留集合：
>
> ```text
> S_pre =
> 低 20% 加速度 episode
> ∩
> 低 20% jerk episode
> ```
>
> 结果写入：
>
> ```text
> runs/spre/complete_analysis_results.json
> ```
>
> 这一步主要筛掉：
>
> - 动作抖动明显的轨迹；
> - 采集过程中的跟踪异常；
> - 机器人状态突变；
> - 录制错误的 episode。
>
> ---
>
> ## Stage 2b：回放验证
>
> 入口：
>
> ```text
> s2b_replay.sh
> scripts/replay_validate.py
> ```
>
> 目标是从 `S_pre` 中找出：
>
> ```text
> S_high
> ```
>
> 也就是任务完成且无碰撞的高质量演示。
>
> 代码支持三种 verifier：
>
> ### 1. `trust_spre`
>
> ```bash
> REPLAY_VERIFIER=trust_spre
> ```
>
> 不真正验证，直接把 `S_pre` 当作通过。
>
> 适合 smoke test，但不是真实质量筛选。
>
> ### 2. `energy`
>
> ```bash
> REPLAY_VERIFIER=energy
> ```
>
> 使用启发式检查：
>
> - 状态是否超出范围；
> - 是否有加速度尖峰；
> - 是否出现异常状态。
>
> 默认大致检查归一化状态是否落在：
>
> ```text
> [-3.5, 3.5]
> ```
>
> ### 3. `mujoco`
>
> ```bash
> REPLAY_VERIFIER=mujoco
> REPLAY_TWIN_MODULE=your_module
> ```
>
> 要求用户提供模块，并暴露：
>
> ```python
> replay(states, actions, task_id)
> ```
>
> 返回：
>
> ```python
> {
>     "success": bool,
>     "collision_free": bool
> }
> ```
>
> 结果写入：
>
> ```text
> runs/shigh.json
> ```
>
> 需要注意，仓库内置的默认 `trust_spre` 并不是真正的 MuJoCo 任务完成验证；真正的仿真验证需要用户接入自己的数字孪生环境。
>
> ---
>
> ## Stage 2c-1：计算 log-π proxy
>
> 入口：
>
> ```text
> s2c_compute_logpi.sh
> scripts/compute_logpi.py
> ```
>
> 它加载 Stage-1 policy，对真实动作进行多次随机 diffusion 加噪和预测。
>
> 对每个样本：
>
> ```text
> 真实动作 a
>     ↓
> 随机选 diffusion timestep t
>     ↓
> 加噪得到 a_t
>     ↓
> Stage-1 策略预测 ε̂θ
>     ↓
> 计算残差能量：
> E_t = ||ε̂θ - ε||²
> ```
>
> 默认：
>
> ```bash
> NUM_NOISE_STEPS=4
> FRAME_STRIDE=10
> ```
>
> 也就是每隔 10 帧计算一次，并对 4 个随机噪声时间步求近似能量。
>
> 随后将能量转成：
>
> ```text
> logπ_hat = -zscore(E)
> ```
>
> 直观上：
>
> - 模型更容易解释的动作；
> - 去噪误差更小；
> - residual energy 更低；
> - log-π proxy 更高。
>
> 输出：
>
> ```text
> runs/logpi/logpi.json
> runs/logpi/logpi_raw_E.json
> ```
>
> ---
>
> ## Stage 2c-2：训练质量评分模型
>
> 入口：
>
> ```text
> s2c_train_scoring.sh
> train/main_scoring.py
> train/train_scoring.py
> models/scoring_model.py
> ```
>
> 默认评分模型约 30M 参数。
>
> 它的输入是：
>
> ```text
> 当前状态 s_t              [B,36]
> 动作 chunk a_{t:t+31}     [B,32,36]
> log-π proxy                [B,1]
> 语言条件                   [B,L,512]
> 图像条件                   [B,2916,1152]
> ```
>
> 在 `ScoringModel.forward()` 中：
>
> ```python
> state_tokens = self.state_proj(state)
> action_tokens = self.action_proj(action_chunk)
> logpi_tokens = self.logpi_proj(logpi_encoded)
> x = torch.cat([
>     state_tokens,
>     action_tokens,
>     logpi_tokens
> ], dim=1)
> ```
>
> 形成：
>
> ```text
> [state token]
> [action token 1 ... action token 32]
> [logπ token]
> ```
>
> 然后与语言、图像条件一起送入 Transformer。
>
> 输出：
>
> ```text
> score ∈ (0,1)
> ```
>
> 含义是当前演示片段的质量评分。
>
> 训练采用 Positive-Unlabeled learning：
>
> - 正样本：`S_high`
> - 未标注样本：其他数据
> - `logπ` 作为辅助质量信号
>
> 评分 checkpoint 默认输出到：
>
> ```text
> checkpoints/dexora-scoring/
> ```
>
> ---
>
> # 六、Stage 3：质量感知后训练
>
> 入口：
>
> ```text
> s3_post_train.sh
> train/main_posttrain.py
> train/train_posttrain.py
> ```
>
> 它加载：
>
> ```text
> Stage-1 policy
> Stage-2 scoring model
> ```
>
> 评分模型冻结，只用于计算样本权重。
>
> 对于每个训练片段：
>
> ```text
> d(x_i) = scoring_model(state, action_chunk, logpi, language, images)
> ```
>
> 再通过 DWBC 映射成权重：
>
> ```text
> w_i = DWBC(d(x_i))
> ```
>
> 代码默认参数：
>
> ```bash
> DWBC_ETA=0.5
> DWBC_W_MIN=0.0
> DWBC_W_MAX=5.0
> DWBC_WARMUP_STEPS=1000
> ```
>
> 最终损失为：
>
> ```text
> L = Σ_i w_i ||ε̂θ - ε||² / Σ_i w_i
> ```
>
> 对应 `RDTRunner.compute_loss()` 的：
>
> ```python
> weighted_mse_loss(
>     pred,
>     target,
>     sample_weights=sample_weights
> )
> ```
>
> 因此：
>
> - 高质量片段权重大；
> - 低质量片段权重小；
> - 不需要直接删除所有低质量数据；
> - 策略继续学习整个数据集，但重点学习更可靠的行为。
>
> 如果设置：
>
> ```bash
> EXTRA_FLAGS="--no_quality_weights"
> ```
>
> 则可以复现不使用判别器权重的普通后训练 baseline。
>
> 输出：
>
> ```text
> checkpoints/dexora-400m-posttrain/
> ```
>
> ---
>
> # 七、离线测试
>
> ## 1. Action curve open-loop 测试
>
> 入口：
>
> ```text
> scripts/eval_action_curves.py
> scripts/run_eval_example.sh
> ```
>
> 运行示例：
>
> ```bash
> MODEL_PATH=checkpoints/dexora-400m-posttrain \
> REPO_DIR=data/Dexora_Real-World_Dataset/airbot_pick_and_place \
> EPISODE_IDX=12 \
> bash scripts/run_eval_example.sh
> ```
>
> 测试逻辑：
>
> 1. 读取某个真实 episode；
> 2. 在每个采样点读取真实状态、图像和任务描述；
> 3. 用策略预测 32 帧动作；
> 4. 与真实动作轨迹比较；
> 5. 对 36 个关节画 GT 和 prediction 曲线。
>
> 默认：
>
> ```bash
> INFERENCE_INTERVAL=32
> ```
>
> 也就是每 32 帧重新生成一个动作 chunk。
>
> 这是 open-loop 测试，特点是：
>
> ```text
> 每次都使用数据集中的真实 observation
> 不把策略上一帧预测结果作为下一帧 observation
> ```
>
> 因此它主要检查：
>
> - 模型是否产生合理动作；
> - 动作曲线是否平滑；
> - 各关节幅值是否正确；
> - 预测动作是否和 demonstration 对齐。
>
> 它不能完全代表真实机器人闭环性能，因为没有执行预测动作后再读取真实机器人状态。
>
> 结果默认写入：
>
> ```text
> eval_results/ep{EPISODE_IDX}/
> ```
>
> ---
>
> ## 2. Smoothness 和动作质量测试
>
> 项目还提供：
>
> ```text
> scripts/eval_smoothness.py
> scripts/analyze_episode_quality.py
> tests/test_smoothness_eval.py
> ```
>
> 这些脚本主要检查：
>
> - 加速度；
> - jerk；
> - 动作曲线连续性；
> - 各维度轨迹是否存在尖峰。
>
> 它们适合在训练后比较：
>
> ```text
> Stage-1 checkpoint
> Stage-3 checkpoint
> no-quality-weight baseline
> ```
>
> ---
>
> # 八、真实机械臂测试和部署
>
> 部署入口：
>
> ```text
> deploy/dexora_inference_zmq.py
> deploy/dexora_policy.py
> ```
>
> 整体是三个进程：
>
> ```text
> Dexora policy 进程
>        │
>        ├── ZMQ :5556 ──> mmk_forwarder.py
>        │                    AIRBOT 双臂
>        │
>        └── ZMQ :5557 ──> xhand_forwarder.py
>                             XHand 双手
> ```
>
> 这样拆分是因为：
>
> - 策略需要 GPU、PyTorch、T5、SigLIP；
> - AIRBOT SDK 有自己的 Python 环境；
> - XHand SDK 使用另一套 Python 3.8 环境；
> - 三者依赖版本可能冲突。
>
> ---
>
> ## 1. 机器人观测
>
> `ZMQRobotInterface` 通过 ZMQ 请求：
>
> ```python
> {"command": "get_observations"}
> ```
>
> 从机械臂 forwarder 获得：
>
> ```python
> qpos
> ```
>
> 从灵巧手 forwarder 获得：
>
> ```python
> left_hand
> right_hand
> ```
>
> 组合为：
>
> ```text
> [左臂6 | 右臂6 | 左手12 | 右手12]
> ```
>
> 同时直接从本地读取四路相机。
>
> 图像输入为：
>
> ```python
> {
>     "cam_head": RGB/BGR image,
>     "cam_left_wrist": image,
>     "cam_third_view": image,
>     "cam_right_wrist": image,
> }
> ```
>
> ---
>
> ## 2. 策略输出
>
> `DexoraPolicy.get_action()` 的返回值是：
>
> ```text
> [32,36]
> ```
>
> 顺序为：
>
> ```text
> [左臂 6 | 右臂 6 | 左手 12 | 右手 12]
> ```
>
> 部署代码使用 chunk-and-replay：
>
> ```text
> 每 32 个控制 tick：
>     运行一次 diffusion policy
>     得到 32 步动作
>     按顺序逐步发送
> ```
>
> 也就是：
>
> ```python
> action_buffer[t % L]
> ```
>
> 其中：
>
> ```text
> L = 32
> ```
>
> ---
>
> ## 3. 发送给机械臂和灵巧手
>
> 前 12 维发给双臂：
>
> ```python
> arm_action_12d = action[:12]
> ```
>
> 发送协议：
>
> ```python
> {
>     "command": "execute_action",
>     "action": arm_action_12d.tolist()
> }
> ```
>
> 后 24 维拆成左右手：
>
> ```python
> left_hand  = action[12:24]
> right_hand = action[24:36]
> ```
>
> 发送协议：
>
> ```python
> {
>     "command": "execute_action",
>     "action_data": {
>         "left_hand": left_hand.tolist(),
>         "right_hand": right_hand.tolist()
>     }
> }
> ```
>
> 因此策略直接输出的是机器人关节动作，部署层不再进行额外的动作解码。
>
> ---
>
> ## 4. 实际机械臂启动
>
> 可以使用三个终端分别启动：
>
> ```text
> 1. mmk_forwarder.py
> 2. xhand_forwarder.py
> 3. dexora_inference_zmq.py
> ```
>
> 也可以使用：
>
> ```text
> deploy/inference.sh
> ```
>
> 策略部署时一般使用：
>
> ```bash
> python deploy/dexora_inference_zmq.py \
>     --model-path checkpoints/dexora-400m-posttrain \
>     --config-path deploy/mmk_xhand_config.yaml \
</details>

## 1. 项目整体做什么

Dexora 是一个面向双臂双手灵巧操作的 VLA（Vision-Language-Action）系统：

- 输入：4 路相机图像、当前机械臂/机械手状态、自然语言任务指令、控制频率。
- 输出：未来 32 个控制时刻的 36 维动作序列。
- 机器人形态：双臂 + 双手，共 36 DoF：

```text
[left_arm 6]
[right_arm 6]
[left_hand 12]
[right_hand 12]
```

公开 LeRobot 数据实际保存 39 维状态和动作，最后 3 维是：

```text
head_joint_1, head_joint_2, spine_joint
```

训练默认通过 `state_dim_keep=36` 截断，不建模这 3 个维度。

项目主要包括：

```text
数据采集：teleop/
数据转换：dataprocess/
训练：train/、models/
离线评估：scripts/
真实机器人部署：deploy/
```

README 宣称的数据规模是约：

- 12.2K 个真实演示 episode
- 2.92M 帧
- 40.5 小时
- 100K 条仿真轨迹

---

# 2. 三阶段训练流程

项目脚本表面上是 `Stage 1 / Stage 2 / Stage 3`，但 Stage 2 又细分为 2a、2b、2c。

整体流程：

```text
真实/仿真 LeRobot 数据
          │
          ▼
Stage 1：训练基础 Diffusion Transformer 策略
          │
          ├── Stage 2a：根据运动平滑度预筛选 Spre
          ├── Stage 2b：回放验证，得到 Shigh
          ├── Stage 2c-1：计算 Stage-1 策略的 log-π proxy
          └── Stage 2c-2：训练质量判别器
          │
          ▼
Stage 3：质量加权微调 Stage-1 策略
          │
          ▼
最终机器人策略
```

入口脚本：

```text
s1_pretrain.sh
s2a_analyze_jerk.sh
s2b_replay.sh
s2c_compute_logpi.sh
s2c_train_scoring.sh
s3_post_train.sh
run_all_stages.sh
```

---

## Stage 1：预训练策略

入口：

```bash
bash s1_pretrain.sh
```

实际执行：

```bash
accelerate launch -m train.main
```

关键默认参数：

```text
模型：400M Diffusion Transformer
训练步数：100000
动作 chunk：32
状态/动作维度：36
相机数量：4
batch size：1
学习率：1e-4
噪声 SNR：40
混合精度：bf16
```

训练数据由：

```text
data/lerobot_vla_dataset.py
train/dataset.py
```

读取。

### 每个训练样本包含什么

LeRobot 的单帧数据大致包括：

```text
observation.state
action
observation.images.top
observation.images.wrist_left
observation.images.wrist_right
observation.images.front
episode_index
frame_index
task/instruction
```

`LeRobotVLADataset.get_item()` 会把它转换成内部格式：

```python
{
    "state":       [36],
    "actions":     [32, 36],
    "cam_high":    [1, H, W, 3],
    "cam_left_wrist":  [1, H, W, 3],
    "cam_right_wrist": [1, H, W, 3],
    "cam_third_view":  [1, H, W, 3],
    "meta": {
        "episode_idx": ...,
        "step_id": ...,
        "instruction": ...
    }
}
```

其中：

- `state` 是当前时刻状态，形状约为 `[1, 36]`
- `actions` 是从当前时刻开始的未来 32 帧动作，形状为 `[32, 36]`
- 4 个图像来自 4 个相机
- 动作和状态使用 `dataset_statistics.json` 做逐维 min-max 归一化

默认状态/动作顺序：

```text
0:6    左臂
6:12   右臂
12:24  左手
24:36  右手
```

---

## Stage 2a：运动质量预筛选

入口：

```bash
bash s2a_analyze_jerk.sh
```

实际代码：

```text
scripts/analyze_episode_quality.py
```

它对每个 episode 的状态序列计算：

- 加速度 RMS：`Aep`
- jerk（三阶差分）RMS：`Jep`

代码逻辑是先对状态做逐维归一化，然后按照 20 Hz 计算时间差分：

```text
速度 ≈ 一阶差分
加速度 ≈ 二阶差分
jerk ≈ 三阶差分
```

随后保留：

```text
Spre = 低 20% 加速度 episode
       ∩
       低 20% jerk episode
```

输出：

```text
runs/spre/complete_analysis_results.json
```

这个阶段只分析演示数据本身的运动平滑程度，并不真正运行机器人，也不判断任务是否完成。

---

## Stage 2b：回放验证

入口：

```bash
bash s2b_replay.sh
```

实际代码：

```text
scripts/replay_validate.py
```

它读取 Stage 2a 的 `Spre`，对其中的 episode 做进一步检查，输出：

```text
runs/shigh.json
```

支持三种验证器。

### 1. `trust_spre`

默认值：

```bash
REPLAY_VERIFIER=trust_spre
```

不做真正验证，所有 `Spre` 都直接认为通过。

这适合：

- 测试整个流程
- 当前没有 MuJoCo 环境时运行
- 快速 smoke test

### 2. `energy`

一种便宜的启发式检查：

- 状态是否超出归一化范围，例如 `[-3.5, 3.5]`
- 二阶差分加速度是否出现尖峰

它只能发现运动异常，不能真正判断任务是否完成。

### 3. `mujoco`

真正的仿真回放需要用户提供一个 twin module，并且必须实现：

```python
replay(states, actions, task_id)
```

返回类似：

```python
{
    "success": bool,
    "collision_free": bool
}
```

只有任务成功且无碰撞，才进入 `Shigh`。

因此，仓库中的默认版本并不包含一个完整的 Dexora MuJoCo 任务验证环境；`mujoco` 模式需要额外接入数字孪生。

---

## Stage 2c-1：计算 log-π proxy

入口：

```bash
bash s2c_compute_logpi.sh
```

实际代码：

```text
scripts/compute_logpi.py
```

输入：

- Stage-1 策略
- 数据集中的状态、动作、图像、语言
- 每个动作 chunk 长度 32

代码对每个样本重复若干次扩散噪声计算：

1. 从真实动作 `action_gt` 开始
2. 随机采样扩散 timestep
3. 加噪得到 `noisy_action`
4. 用 Stage-1 策略预测噪声
5. 计算预测噪声与真实噪声之间的误差能量

近似公式：

```text
E_t = || εθ(noisy_action, condition, t) - ε ||²
```

默认每个样本采样 4 个 noise steps：

```bash
NUM_NOISE_STEPS=4
```

最后计算：

```text
logπ_hat_t = - zscore(E_t)
```

直观解释：

- Stage-1 策略越容易解释某段动作
- 去噪误差越小
- `logπ_hat` 越高
- 这段动作越像策略认为的“合理动作”

输出：

```text
runs/logpi/logpi.json
runs/logpi/logpi_raw_E.json
```

默认每 10 帧计算一次：

```bash
FRAME_STRIDE=10
```

---

## Stage 2c-2：训练质量判别器

入口：

```bash
bash s2c_train_scoring.sh
```

实际入口：

```text
train/main_scoring.py
train/train_scoring.py
models/scoring_model.py
```

默认判别器约 30M 参数。

判别器输入：

```text
当前状态 s_t
未来动作块 a[t:t+32]
logπ proxy
语言条件
多视角图像条件
```

模型内部序列是：

```text
[state token]
[action token 1]
...
[action token 32]
[logπ token]
```

代码中：

```python
state_proj(state)
action_proj(action_chunk)
logpi_proj(logpi_chunk)
```

然后送入 Transformer。

输出：

```text
score ∈ (0, 1)
```

表示该动作片段/演示的质量分数。

训练使用 Positive-Unlabeled（PU）学习：

- 正样本：`Shigh`
- 未标记样本：其余数据
- `Spre` 在某些配置中可作为 fallback

输出判别器 checkpoint，例如：

```text
checkpoints/dexora-scoring/final_model/pytorch_model.bin
```

---

## Stage 3：质量感知后训练

入口：

```bash
bash s3_post_train.sh
```

实际入口：

```text
train/main_posttrain.py
train/train_posttrain.py
```

它加载：

```text
Stage-1 policy
Stage-2 scoring discriminator
```

每个训练样本先经过判别器得到质量分数：

```text
d(x_i)
```

然后通过 DWBC 方式转换成训练权重：

```text
w_i = DWBC(d(x_i))
```

训练损失变成：

```text
L = Σ_i w_i || εθ - ε ||² / Σ_i w_i
```

也就是说：

- 高质量片段权重大
- 低质量片段权重小
- 不直接删除低质量数据
- 仍然保留一定的数据覆盖范围

默认权重范围：

```text
w_min = 0
w_max = 5
warmup = 1000 steps
eta = 0.5
```

也可以通过：

```bash
EXTRA_FLAGS="--no_quality_weights" bash s3_post_train.sh
```

关闭质量权重，复现普通微调 baseline。

---

# 3. 策略模型架构

核心模型是：

```text
T5-XXL             语言编码器
SigLIP-SO400M      图像编码器
MLP adaptor        模态维度适配
RDT                Diffusion Transformer
DDPM/DPMSolver++   扩散训练/采样
```

---

## 3.1 语言编码器 T5

默认：

```text
google/t5-v1_1-xxl
```

输入：

```text
instruction，例如：
"Pick up the apple and put it on the plate."
```

输出：

```text
language tokens
[B, language_length, 512]
```

训练配置中：

```yaml
lang_token_dim: 512
tokenizer_max_length: 1024
```

语言 token 经过：

```python
lang_adaptor = mlp2x_gelu
```

映射到 Transformer 的 hidden size：

```text
512 → 1024
```

---

## 3.2 视觉编码器 SigLIP

默认：

```text
google/siglip-so400m-patch14-384
```

输入：

```text
4 张 384×384 左右的 RGB 图像
```

相机：

```text
top/head
left wrist
right wrist
front/third view
```

SigLIP 每张图产生：

```text
27 × 27 = 729 个 patch tokens
```

4 个相机总共：

```text
4 × 729 = 2916 image tokens
```

配置：

```yaml
num_cameras: 4
num_patches_per_view: 729
img_token_dim: 1152
```

图像 token 经过：

```python
img_adaptor = mlp2x_gelu
```

映射：

```text
1152 → 1024
```

---

## 3.3 状态输入

当前机器人状态：

```text
state ∈ R^36
```

代码中状态还会与 action mask 拼接：

```text
[state, state/action mask]
```

因此状态适配器输入维度是：

```text
36 × 2 = 72
```

对应：

```python
state_adaptor = mlp3x_gelu
```

映射到：

```text
72 → 1024
```

模型还接收：

```text
ctrl_freq
```

例如：

```text
20 Hz
```

控制频率经过 `freq_embedder` 变成一个 token。

---

## 3.4 RDT Diffusion Transformer

默认配置：

```yaml
hidden_size: 1024
depth: 28
num_heads: 16
action_chunk_size: 32
```

约 400M 参数。

RDT 的主序列大致是：

```text
[timestep token]
[control frequency token]
[state token]
[action token 1]
...
[action token 32]
```

其中：

- `timestep token`：当前扩散时间步
- `control frequency token`：控制频率
- `state token`：当前状态
- `action tokens`：带噪动作序列

语言 token 和图像 token 是条件序列，作为 Transformer 的条件输入。

`RDT.forward()` 的主要输入：

```python
x       # state + noisy action tokens
freq    # 控制频率
t       # diffusion timestep
lang_c  # 语言条件
img_c   # 图像条件
```

输出：

```text
[B, 32, 36]
```

表示每个动作时刻、每个控制维度的预测值。

---

# 4. Diffusion 训练与推理

## 训练

`RDTRunner.compute_loss()` 的逻辑：

```python
noise = torch.randn_like(action_gt)

noisy_action = noise_scheduler.add_noise(
    action_gt, noise, timesteps
)

pred = model(
    state,
    noisy_action,
    timestep,
    language,
    images
)
```

默认配置：

```text
训练扩散步数：1000
beta schedule：squaredcos_cap_v2
prediction_type：epsilon
```

训练目标：

```text
predicted noise ≈ actual noise
```

即：

```text
|| εθ - ε ||²
```

Stage 3 只是给每个样本的 MSE 加上质量权重。

---

## 推理

`RDTRunner.predict_action()`：

1. 从纯高斯噪声初始化长度为 32 的动作序列
2. 输入当前状态、语言和图像条件
3. 通过扩散采样逐步去噪
4. 输出未来 32 帧动作

推理时使用：

```text
DPMSolver++
num_inference_timesteps = 5
```

因此训练时是 1000 步 DDPM，部署时只进行 5 步快速采样。

输出：

```text
[32, 36]
```

部署代码 `DexoraPolicy.get_action()` 会将其转成 numpy 数组，并反归一化后交给机器人。

---

# 5. 数据采集流程

数据采集主要由两个并行进程完成：

```text
机械臂 + 相机录制
灵巧手 + Vision Pro 遥操作
```

入口：

```text
teleop/scripts/record_delete.py
```

它启动：

```text
imitate_all/record_4_rgb_cam.py
teleop_pkg/receive_from_vision_pro.py
```

并使用两个 Python 环境：

```text
imitall
xhand_tele_env
```

原因是 AIRBOT SDK 和 XHAND SDK 依赖不同的 Python 版本和包。

---

## 5.1 机械臂与相机采集

`record_4_rgb_cam.py` 使用：

```text
左腕相机
右腕相机
第三视角相机
头部相机
```

默认采样频率：

```text
20 Hz
```

机器人每帧记录：

- 机械臂关节状态
- 相机图像
- 时间戳
- episode/frame 编号

图像由异步 `ImageSaver` 线程写盘，减少保存图像对控制循环的影响。

默认参数中：

```text
每 episode 1000 帧
fps 20
```

原始数据保存为类似：

```text
episode_0/
episode_0.bson
```

---

## 5.2 Vision Pro 灵巧手采集

`receive_from_vision_pro.py` 使用：

```python
node = XHandTeleOps("config.yaml")
```

每帧执行：

1. 从 Vision Pro 获取手部骨骼数据
2. `retarget_data()` 将 Vision Pro 手部骨骼映射到 XHAND 的 12 DoF
3. 读取当前左右手真实关节状态
4. 发送目标动作到 XHAND
5. 将动作和观测写入内存中的 `frames`

每一帧的数据类似：

```python
{
    "t": timestamp,
    "action": {
        "left_hand":  [12],
        "right_hand": [12]
    },
    "observation": {
        "left_hand":  [12],
        "right_hand": [12]
    }
}
```

停止后写入：

```text
xhand_control_data.bson
```

---

## 5.3 episode 归档

两个子进程结束后，`record_delete.py` 会把：

```text
episode_0/
episode_0.bson
xhand_control_data.bson
```

复制到：

```text
ARCHIVE_ROOT/episode_N/
```

然后删除原始临时目录。

注意：代码中的 `ARCHIVE_ROOT`、两个 Python 解释器路径、相机设备路径都是机器相关配置，需要修改。

---

# 6. BSON 到 LeRobot

采集数据原始格式主要是 BSON，训练使用 LeRobot v2.1。

转换入口主要在：

```text
dataprocess/airbot_lerobot.py
dataprocess/airbot.py
dataprocess/code/
```

数据转换后典型结构：

```text
dataset/
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet
├── videos/
│   └── chunk-000/
│       ├── observation.images.top/
│       ├── observation.images.wrist_left/
│       ├── observation.images.wrist_right/
│       └── observation.images.front/
└── meta/
    ├── info.json
    ├── episodes.jsonl
    ├── tasks.jsonl
    └── modality.json
```

训练 loader 通过：

```python
LeRobotDataset("", repo_dir, delta_timestamps=...)
```

读取连续的：

```text
当前状态
未来 32 帧动作
当前/历史图像
任务语言
```

如果统计文件不存在，训练脚本会自动执行：

```bash
python -m data.lerobot_vla_dataset --stat
```

生成：

```text
new_lerobot_stats/dataset_statistics.json
```

---

# 7. 离线测试

## 7.1 动作曲线测试

入口：

```bash
bash scripts/run_eval_example.sh
```

实际代码：

```text
scripts/eval_action_curves.py
```

示例：

```bash
MODEL_PATH=checkpoints/dexora-400m-posttrain \
REPO_DIR=data/Dexora_Real-World_Dataset/airbot_pick_and_place \
EPISODE_IDX=12 \
bash scripts/run_eval_example.sh
```

它：

1. 加载一个 LeRobot episode
2. 使用数据集中真实的当前状态和图像
3. 运行策略预测未来 32 步动作
4. 与 ground-truth 动作比较
5. 绘制 36 个关节的预测曲线和真实曲线

这是 open-loop 测试：

```text
每次仍然使用真实观测作为条件
不会把模型上一步的预测动作作为下一步观测
```

所以它适合检查：

- 动作方向是否合理
- 动作是否过于抖动
- 预测是否与演示大致对齐
- 各关节尺度是否正常

但它不能代表真实机器人闭环成功率。

输出通常位于：

```text
eval_results/epN/
```

---

## 7.2 平滑度评估

相关脚本：

```text
scripts/eval_smoothness.py
scripts/analyze_episode_quality.py
```

主要通过动作/状态的：

```text
速度
加速度
jerk
```

评估动作是否平滑。

---

## 7.3 replay_validate 测试

```text
scripts/replay_validate.py
```

可用于：

- 读取 episode 的状态和动作
- 使用启发式 energy verifier
- 或调用外部 MuJoCo twin
- 判断状态是否越界、是否有高加速度尖峰、是否完成任务、是否碰撞

当前默认 `trust_spre` 不是真正的仿真测试，只是直接信任预筛结果。

---

# 8. 真实机械手臂测试/部署

真实部署入口：

```text
deploy/dexora_inference_zmq.py
```

部署被拆成三个进程：

```text
策略进程：GPU、PyTorch、T5、SigLIP
机械臂进程：AIRBOT SDK
灵巧手进程：XHAND SDK
```

通信方式：

```text
ZMQ
```

结构：

```text
dexora_inference_zmq.py
        │
        ├── tcp://localhost:5556
        │       mmk_forwarder.py
        │
        └── tcp://localhost:5557
                xhand_forwarder.py
```

这样可以隔离不同 SDK 的 Python 环境。

---

## 8.1 策略进程获取观测

`ZMQRobotInterface` 分别请求：

```python
{"command": "get_observations"}
```

机械臂返回：

```python
{
    "qpos": [...]
}
```

灵巧手返回：

```python
{
    "left_hand":  [12],
    "right_hand": [12]
}
```

同时本地采集 4 路图像：

```text
cam_head
cam_left_wrist
cam_third_view
cam_right_wrist
```

最终策略输入：

```python
obs = {
    "state": [36],
    "images": {
        "cam_head": ...,
        "cam_left_wrist": ...,
        "cam_third_view": ...,
        "cam_right_wrist": ...
    },
    "instruction": "...",
    "ctrl_freq": 20.0
}
```

---

## 8.2 策略输出和动作拆分

策略输出：

```text
[32, 36]
```

按维度拆分：

```text
action[:, 0:12]   → 双臂
action[:, 12:24]  → 左手
action[:, 24:36]  → 右手
```

发送给机械臂：

```python
{
    "command": "execute_action",
    "action": arm_action_12d.tolist()
}
```

发送给灵巧手：

```python
{
    "command": "execute_action",
    "action_data": {
        "left_hand": left_12d.tolist(),
        "right_hand": right_12d.tolist()
    }
}
```

---

## 8.3 chunk-and-replay 控制方式

代码不会每个控制 tick 都重新运行一次扩散模型，而是：

1. 每 32 个控制周期运行一次策略
2. 一次产生 32 帧动作
3. 后续 32 个周期逐帧执行动作 buffer
4. 到下一个 chunk 时重新采样

伪代码：

```python
if step % chunk_size == 0:
    action_buffer = policy.get_action(obs)

action = action_buffer[step % chunk_size]
send_to_robot(action)
```

这样可以降低 T5、SigLIP 和扩散采样的实时计算压力。

---

# 9. 需要特别注意的代码问题

有几个实现细节值得注意：

1. `s2b_replay.sh` 默认使用 `trust_spre`，因此默认流程并没有真正做 MuJoCo 任务成功验证。

2. `s2c_train_scoring.sh` 中有一处疑似笔误：

```bash
: "${TRAIN_BATCH_SIZE:=1}s"
```

结果默认值可能会变成字符串 `"1s"`，正常运行时建议显式设置：

```bash
TRAIN_BATCH_SIZE=1 bash s2c_train_scoring.sh
```

3. `scripts/replay_validate.py` 中部分代码兼容旧字段名 `states`，而 LeRobot v2.1 使用：

```text
observation.state
```

实际运行不同数据版本时需要确认字段名。

4. 离线 action curve 测试是 open-loop，不等价于真实机器人闭环测试。

5. 真实部署前需要确认：

```text
动作是否已经反归一化
相机顺序是否正确
左右手维度顺序是否正确
AIRBOT/XHAND 的单位是否都是弧度
```

6. 机械臂和灵巧手必须在对应的 SDK 环境中运行，不能简单地把所有代码放到同一个 Python 环境里。
