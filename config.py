from omegaconf import OmegaConf

cfg = {
    "description": "Final Model",
    "precision": "32",
    "num_devices": 1,
    "num_epochs": 140,
    "save_log_weights_interval": 20,
    "train_metric_interval": 20,
    "inference_epochs": [],
    "model_checkpoint_at": "checkpoints",
    "img_size": 1024,
    "out_dir": "/",
    "focal_wt": 20,
    "num_tokens": 0,
    "result_dir": "",
    "opt": {
        "learning_rate": 5e-4,
        "auto_lr": False,
        "weight_decay": 0.01,
        "decay_factor": 1/2,
        "steps": [],
    },
    "model": {
        "type": "vit_l",
        "checkpoint": "sam_vit_l_0b3195.pth",
        "requires_grad": {
            "image_encoder": False,
            "prompt_encoder": False,
            "mask_decoder": False,
            "propagation_module": True,
        },
        "multimask_output": True,
        "propagation_ckpt": None,
    },
    "dataset": {
        "name": "moca",
        "root_dir": "raw/",
        "stage1": True,
        "train_batch_size": 1,
        "max_num_obj": 2,
        "num_frames": 3,
        "max_jump": 1,
        "num_workers": 4,
        "pin_memory": False,
        "persistent_workers": True,
    },
}
cfg = OmegaConf.create(cfg)
