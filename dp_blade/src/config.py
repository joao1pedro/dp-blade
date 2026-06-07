import os
import torch
import mlflow
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
MLFLOW_EXPERIMENT_NAME = "dp_blade-text-cls_epsilon1.0"

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = f'./dp_blade_experiments/checkpoints/{timestamp}'
os.makedirs(SAVE_DIR, exist_ok=True)

TASK_TYPE = "text" 

DATASET_CONFIG = {
    #"name": "cifar10",
    "name": "qnli",
    "train_subset_size": None,
    "val_subset_size": None,
    "batch_size": 64, # default 64
    "val_batch_size": 128, # default 128
}

MODEL_CONFIG = {
    #"model_id": "google/vit-base-patch16-224-in21k",
    #"num_labels": 10,
    "model_id": "bert-base-uncased",
    "num_labels": 2, # 3 for mnli
    #"num_labels": 3,
}

TRAIN_CONFIG = {
    #"learning_rate": 5e-4,
    "learning_rate": 5e-5,
    "epochs": 30,
    "max_phys_batch": 128, # default 128
    "checkpoint_interval": 1,
    "patience": 15,
}

LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "bias": "none",
    #"target_modules": ["q_proj", "v_proj"],
    "target_modules": ["q_proj", "v_proj", "query", "value"],
    "modules_to_save": ["classifier"]
}

DP_CONFIG = {
    "target_epsilon": 1.0,
    "target_delta": 1e-5,
    "max_grad_norm": 0.1,
    "poisson_sampling": True,
}

IAKF_DEFAULT_CONFIG = {
    "gamma": 0.5,
    "kappa_shift": 0.7,
    "A": 1.0,
    "alpha": 0.05,
    "q_scale": 1e-2,
    "kappa_min": 0.01,
    "kappa_max": 0.99,
}