#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用真实训练样本直接跑 rdt.predict_action(...)，绕开仿真环境和 sim_eval_dexora.py
里的 obs->policy_obs 转换，只测「checkpoint + 采样本身」在训练分布输入下是否正常。

本质是把 train/sample.py::log_sample_res 的核心逻辑抽出来、去掉 accelerator/EMA/
分布式那一层，加上逐样本的原始数值打印，方便对照。

用法:
    cd /home/ubuntu/myh/expirement/Dexora
    python replay_check.py
"""
import sys
import yaml
import torch

sys.path.insert(0, ".")  # 确保能 import models/ train/ data/

from models.rdt_runner import RDTRunner, DPMSolverMultistepScheduler
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder
from train.dataset import DataCollatorForVLAConsumerDataset, VLAConsumerDataset

# ============================================================
# 下面这些值需要跟你实际启动 train.py 用的命令行参数完全对齐
# —— 尤其 STATE_DIM_KEEP / DATASET_TYPE / LOAD_FROM 三项，
#    麻烦对照你训练脚本里的 --state_dim_keep / --dataset_type / --load_from 改。
# ============================================================
MODEL_CONFIG_PATH = "/home/ubuntu/myh/expirement/Dexora/configs/base_400m.yaml"
CHECKPOINT_DIR     = "/home/ubuntu/myh/expirement/Dexora/checkpoints/dexora-400m-pretrain/"
TEXT_ENCODER_PATH  = "/home/ubuntu/myh/expirement/Dexora/google/t5-v1_1-small"
VISION_ENCODER_PATH= "/home/ubuntu/myh/expirement/Dexora/google/siglip-so400m-patch14-384"
LEROBOT_ROOT        = "/home/ubuntu/myh/expirement/Dexora/lerobot_data/two_arm_can_sort_random"
STATS_FILE          = "/home/ubuntu/myh/expirement/Dexora/lerobot_data/new_lerobot_stats/dataset_statistics.json"

DATASET_TYPE       = "finetune"   # <-- 对照 --dataset_type 确认
LOAD_FROM           = "lerobot"    # <-- 对照 --load_from 确认
PRECOMP_LANG_EMBED  = False        # <-- 对照 --precomp_lang_embed 确认
STATE_DIM_KEEP       = 24          # <-- ⚠️ 重点确认：训练时 --state_dim_keep 到底传的是多少
# ============================================================

DEVICE = "cuda"
DTYPE = torch.bfloat16
N_SAMPLES_TO_CHECK = 4


def main():
    with open(MODEL_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    print("Loading T5 text encoder ...")
    text_embedder = T5Embedder(
        from_pretrained=TEXT_ENCODER_PATH,
        model_max_length=config["dataset"]["tokenizer_max_length"],
        device=DEVICE,
    )
    tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
    text_encoder = text_encoder.to(DEVICE, dtype=DTYPE).eval()

    print("Loading SigLIP vision encoder ...")
    vision_encoder = SiglipVisionTower(vision_tower=VISION_ENCODER_PATH, args=None)
    vision_encoder.vision_tower.to(DEVICE, dtype=DTYPE).eval()
    image_processor = vision_encoder.image_processor

    print(f"Loading RDT policy from {CHECKPOINT_DIR} ...")
    rdt = RDTRunner.from_pretrained(CHECKPOINT_DIR).to(DEVICE, dtype=DTYPE).eval()

    dataset_common_kwargs = dict(
        config=config["dataset"],
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_cameras=config["common"]["num_cameras"],
        img_history_size=config["common"]["img_history_size"],
        dataset_type=DATASET_TYPE,
        use_hdf5=LOAD_FROM,
        use_precomp_lang_embed=PRECOMP_LANG_EMBED,
        lerobot_root=LEROBOT_ROOT,
        stats_file=STATS_FILE,
        state_dim_keep=STATE_DIM_KEEP,
    )
    sample_dataset = VLAConsumerDataset(
        image_aug=False,
        cond_mask_prob=0,
        cam_ext_mask_prob=-1,
        state_noise_snr=None,
        **dataset_common_kwargs,
    )
    data_collator = DataCollatorForVLAConsumerDataset(tokenizer)
    dataloader = torch.utils.data.DataLoader(
        sample_dataset, batch_size=1, shuffle=True, collate_fn=data_collator,
    )

    it = iter(dataloader)
    with torch.no_grad():
        for i in range(N_SAMPLES_TO_CHECK):
            batch = next(it)

            images = batch["images"].to(DEVICE, dtype=DTYPE)
            states = batch["states"].to(DEVICE, dtype=DTYPE)[:, -1:, :]
            actions_gt = batch["actions"].to(DEVICE, dtype=torch.float32)
            state_elem_mask = batch["state_elem_mask"].to(DEVICE, dtype=DTYPE)
            ctrl_freqs = batch["ctrl_freqs"].to(DEVICE)
            lang_attn_mask = batch["lang_attn_mask"].to(DEVICE)

            B, T, C, H, W = images.shape
            image_embeds = vision_encoder(images.reshape(-1, C, H, W)).detach()
            image_embeds = image_embeds.reshape((B, -1, vision_encoder.hidden_size))

            if "input_ids" in batch:
                text_embeds = text_encoder(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=lang_attn_mask,
                )["last_hidden_state"].detach()
            else:
                text_embeds = batch["lang_embeds"].to(DEVICE, dtype=DTYPE)

            pred_actions = rdt.predict_action(
                lang_tokens=text_embeds,
                lang_attn_mask=lang_attn_mask,
                img_tokens=image_embeds,
                state_tokens=states,
                action_mask=state_elem_mask.unsqueeze(1),
                ctrl_freqs=ctrl_freqs,
            ).float()

            print(f"\n=== sample {i} ===")
            print("dataset:", [sample_dataset.get_dataset_id2name()[d] for d in batch["data_indices"]])
            print("pred_actions[0,0]  (normalized):", pred_actions[0, 0].cpu().numpy())
            print("actions_gt[0,0]    (normalized):", actions_gt[0, 0].cpu().numpy())
            mse = torch.nn.functional.mse_loss(pred_actions, actions_gt).item()
            print(f"MSE(pred, gt) over full chunk (via predict_action/sampling): {mse:.6f}")

            # ---- 对照测试：直接跑训练时的 compute_loss（单步 noise-prediction MSE）----
            # 这个和训练时 rdt(...) / forward() 走的是完全同一条路径，
            # 用来判断「权重本身是否真的是那个低 loss 的 checkpoint」。
            train_style_loss = rdt.compute_loss(
                lang_tokens=text_embeds,
                lang_attn_mask=lang_attn_mask,
                img_tokens=image_embeds,
                state_tokens=states,
                action_gt=actions_gt.to(dtype=DTYPE),
                action_mask=state_elem_mask.unsqueeze(1),
                ctrl_freqs=ctrl_freqs,
            )
            print(f"compute_loss (training-style, single-step eps MSE): {train_style_loss.item():.6f}")

            # ---- 对照测试：把采样用的 DPMSolverMultistepScheduler 换成训练用的
            # DDPMScheduler（同一个 beta_schedule/prediction_type），多跑几十步，
            # 看是不是 DPM-Solver 多步法在 5 步下没收敛，还是 conditional_sample
            # 这个循环本身有 bug（换了 scheduler 依然炸，就是循环本身的问题）。
            orig_scheduler = rdt.noise_scheduler_sample
            orig_num_steps = rdt.num_inference_timesteps
            # rdt.noise_scheduler_sample = rdt.noise_scheduler  # 换成训练用的 DDPMScheduler
            rdt.num_inference_timesteps = 50                  # 多给点步数，排除少步近似误差
            # rdt.noise_scheduler_sample = DPMSolverMultistepScheduler(
            #     num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon", use_karras_sigmas=True)
            # rdt.num_inference_timesteps = 5
            pred_actions_ddpm = rdt.predict_action(
                lang_tokens=text_embeds,
                lang_attn_mask=lang_attn_mask,
                img_tokens=image_embeds,
                state_tokens=states,
                action_mask=state_elem_mask.unsqueeze(1),
                ctrl_freqs=ctrl_freqs,
            ).float()
            rdt.noise_scheduler_sample = orig_scheduler
            rdt.num_inference_timesteps = orig_num_steps

            print("pred_actions_ddpm[0,0] (用训练用 DDPMScheduler):",
                  pred_actions_ddpm[0, 0].cpu().numpy())
            mse_ddpm = torch.nn.functional.mse_loss(pred_actions_ddpm, actions_gt).item()
            print(f"MSE(pred_ddpm, gt) over full chunk: {mse_ddpm:.6f}")


if __name__ == "__main__":
    main()
