import os
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback, ModelSummary, LearningRateMonitor
from segment_anything import sam_model_registry
from segment_anything.modeling.camosam import CamoSam

import wandb

from dataloaders.camo_dataset import get_loader
from dataloaders.vos_dataset import get_loader as get_loader_moca
from dataloaders.moca_test import get_test_loader
from callbacks import WandB_Logger
from config import cfg
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 配置 wandb 离线模式 - 必须在任何 wandb 操作之前设置
    os.environ["WANDB_MODE"] = "offline"

    # 可选：设置本地保存目录（默认是 ./wandb）
    os.environ["WANDB_DIR"] = "./wandb_logs"  # 自定义保存目录

    # 可选：禁用一些在线功能
    os.environ["WANDB_DISABLE_CODE"] = "false"  # 仍然保存代码
    os.environ["WANDB_SILENT"] = "true"  # 减少输出信息

    L.seed_everything(2023, workers=True)
    torch.set_float32_matmul_precision('high')

    print("WandB 配置为离线模式，日志将保存在本地")

    ckpt = None

    if cfg.model.propagation_ckpt:
        ckpt = torch.load(cfg.model.propagation_ckpt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = sam_model_registry[cfg.model.type](checkpoint=cfg.model.checkpoint, cfg=cfg)
    model = CamoSam(cfg, model, ckpt=ckpt)

    # WandbLogger 会自动使用离线模式
    wandblogger = WandbLogger(
        project="CVPR_Final",
        save_code=True,
        settings=wandb.Settings(
            code_dir="."
        ),
        # 可以指定实验名称和标签
        name="sam_pm_experiment",
        tags=["offline", "sam", "training"]
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    model_weight_callback = WandB_Logger(cfg, wandblogger)

    callbacks = [ModelSummary(max_depth=3), lr_monitor, model_weight_callback]

    trainer = L.Trainer(
        accelerator=device,
        devices=cfg.num_devices,
        callbacks=callbacks,
        precision=cfg.precision,
        logger=wandblogger,
        max_epochs=cfg.num_epochs,
        num_sanity_val_steps=0,
        log_every_n_steps=15,
        enable_checkpointing=True,
        profiler='simple',
    )

    if trainer.global_rank == 0:
        wandblogger.experiment.config.update(dict(cfg))

    if cfg.dataset.stage1:
        train_dataloader = get_loader_moca(cfg.dataset)
    else:
        train_dataloader = get_loader(cfg.dataset)

    print(f"日志将保存在: {wandblogger.experiment.dir}")
    trainer.fit(model, train_dataloader)

    total_losses_for_plotting = model.train_benchmark

    # 训练结束后，从模型中获取收集到的损失数据
    mse_losses_for_plotting = model.train_mse_benchmark  # **新增：获取 MSE_LOSS 数据**
    import datetime
    # --- 绘制 Total Loss ---
    if total_losses_for_plotting:
        print("绘制训练总损失曲线...")
        plt.figure(figsize=(10, 6))
        plt.plot(total_losses_for_plotting, label='Total Loss (per step)')
        plt.title('Training Total Loss Progression')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.grid(True)
        plot_filename_total = f"./loss/total_loss_{datetime.datetime.now()}.png"
        plt.savefig(plot_filename_total)
        print(f"总损失图已保存为：{plot_filename_total}")
        # plt.show() # 如果你想独立显示每个图，可以保留

    # --- 绘制 MSE Loss ---
    if mse_losses_for_plotting:
        print("绘制 MSE 损失曲线...")
        plt.figure(figsize=(10, 6))
        plt.plot(mse_losses_for_plotting, label='MSE Loss (per step)', color='orange')  # 使用不同颜色
        plt.title('Training MSE Loss Progression')
        plt.xlabel('Training Steps')
        plt.ylabel('MSE Loss Value')
        plt.legend()
        plt.grid(True)
        plot_filename_mse = f"./loss/mse_loss_{datetime.datetime.now()}.png"
        plt.savefig(plot_filename_mse)
        print(f"MSE 损失图已保存为：{plot_filename_mse}")
        plt.show()  # 显示 MSE 损失图

    else:
        print("没有收集到 MSE 损失数据，无法绘制图表。请检查 CamoSam 模型的 `train_mse_benchmark` 列表是否被正确填充。")

    print("训练完成！离线日志保存在本地，可以稍后同步到服务器。")


    # for i, batch in enumerate(train_dataloader):
    #     print(f"input image shape: {batch['image'].shape}")
    #     print(f"num_objects: {batch['num_objects'].shape}")
    #     for key in batch:
    #         print(f"Key: {key}, Type: {type(batch[key])}")
    #     break

