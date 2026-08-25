import os
import json
import torch
import numpy as np
import mlflow
from contextlib import nullcontext
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModelForImageClassification, AutoImageProcessor
from peft import get_peft_model, LoraConfig
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

from src.config import DEVICE, SAVE_DIR, DATASET_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, LORA_CONFIG, DP_CONFIG, IAKF_DEFAULT_CONFIG, TASK_TYPE, MLFLOW_EXPERIMENT_NAME, FFTKF_DEFAULT_CONFIG
from src.data.make_dataset import build_loaders
from src.models.optimizers import DopplerDPOptimizer, DiSKOptimizer, IAKFBlockwiseDiSKOptimizer, FFTKFOptimizer
from src.evaluation.metrics import evaluate_metrics, compute_grad_snr, save_checkpoint

def run_experiment(exp_name, method, iakf_config=None, fftkf_config=None, seed=42):
    print(f"\n[{exp_name}] Starting Seed: {seed} | Method: {method}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    run_name_structured = f"{TASK_TYPE.upper()}/{DATASET_CONFIG['name']} - {method}"

    with mlflow.start_run(run_name=run_name_structured):
        mlflow.set_tags({
            "project": "dp_blade",
            "task_type": TASK_TYPE,
            "dataset": DATASET_CONFIG["name"],
            "method": method
        })
        
        mlflow.log_params({
            "exp_name": exp_name,
            "seed": seed,
            **DATASET_CONFIG,
            **MODEL_CONFIG,
            **TRAIN_CONFIG,
            **LORA_CONFIG,
            **DP_CONFIG
        })

        if method == "IAKF" and iakf_config:
            mlflow.log_params({f"iakf_{k}": v for k, v in iakf_config.items()})
        if method == "FFTKF" and fftkf_config:
            mlflow.log_params({f"fftkf_{k}": v for k, v in fftkf_config.items()})

        if TASK_TYPE == "image":
            processor_or_tokenizer = AutoImageProcessor.from_pretrained(MODEL_CONFIG["model_id"])
            model_class = AutoModelForImageClassification
        else:
            processor_or_tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["model_id"])
            model_class = AutoModelForSequenceClassification

        train_loader, val_loader = build_loaders(DATASET_CONFIG, processor_or_tokenizer)

        model = model_class.from_pretrained(
            MODEL_CONFIG["model_id"],
            num_labels=MODEL_CONFIG["num_labels"],
            ignore_mismatched_sizes=True
        )

        model = get_peft_model(model, LoraConfig(**LORA_CONFIG))
        model.to(DEVICE)
        model.train()

        base_optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=TRAIN_CONFIG["learning_rate"], betas=(0.9, 0.999))

        expanded_alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)) + [128, 256, 512, 1024]

        if method != "NO_DP":
            privacy_engine = PrivacyEngine()
            model, dp_optimizer, current_train_loader = privacy_engine.make_private_with_epsilon(
                module=model, 
                optimizer=base_optimizer, 
                data_loader=train_loader,
                target_epsilon=DP_CONFIG["target_epsilon"],
                target_delta=DP_CONFIG["target_delta"],
                epochs=TRAIN_CONFIG["epochs"],
                max_grad_norm=DP_CONFIG["max_grad_norm"],
                alphas=expanded_alphas
            )

            if method == "IAKF":
                conf = iakf_config if iakf_config else IAKF_DEFAULT_CONFIG
                optimizer = IAKFBlockwiseDiSKOptimizer(dp_optimizer=dp_optimizer, model=model, **conf)
            elif method == "FFTKF":
                conf = fftkf_config if fftkf_config else FFTKF_DEFAULT_CONFIG
                optimizer = FFTKFOptimizer(dp_optimizer=dp_optimizer, model=model, **conf)
            elif method == "DISK":
                optimizer = DiSKOptimizer(dp_optimizer=dp_optimizer, model=model)
            elif method == "DOPPLER":
                optimizer = DopplerDPOptimizer(dp_optimizer=dp_optimizer, model=model)
            else:
                optimizer = dp_optimizer
        else:
            optimizer, current_train_loader, privacy_engine = base_optimizer, train_loader, None

        history = { 
            "train_loss": [], "val_loss": [], 
            "train_accuracy": [], "val_accuracy": [], 
            "val_f1": [], "val_precision": [], "val_recall": [], 
            "snr": [], "gpu_memory_gb": [], "epsilon": [] 
        }
        if method == "IAKF":
            for k in ['k_t', 'nu_sq_norm', 'p_t', 'q_t', 'C_t']: history[k] = []

        best_val_acc, best_val_f1 = 0.0, 0.0
        epoch_best_acc, epoch_best_f1 = -1, -1
        final_val_acc, final_val_f1, final_eps = 0.0, 0.0, 0.0
        exp_save_dir = os.path.join(SAVE_DIR, exp_name, f"seed_{seed}")

        patience = TRAIN_CONFIG.get("patience", 3)
        epochs_no_improve = 0

        for epoch in range(TRAIN_CONFIG["epochs"]):
            model.train()
            epoch_losses, epoch_snrs, epoch_accs = [], [], []
            torch.cuda.reset_peak_memory_stats()

            ctx = BatchMemoryManager(data_loader=current_train_loader, max_physical_batch_size=TRAIN_CONFIG["max_phys_batch"], optimizer=optimizer) if method != "NO_DP" else nullcontext(current_train_loader)

            with ctx as loader:
                for step, batch in enumerate(loader):
                    if TASK_TYPE == "image":
                        inputs = batch["pixel_values"].to(DEVICE)
                        outputs_kwargs = {"pixel_values": inputs}
                    else:
                        inputs = batch["input_ids"].to(DEVICE)
                        outputs_kwargs = {"input_ids": inputs}
                        
                    labels = batch["label"].to(DEVICE)
                    outputs_kwargs["labels"] = labels

                    acc_val = [0.0]
                    def closure():
                        outputs = model(**outputs_kwargs)
                        loss = outputs.loss
                        preds = torch.argmax(outputs.logits.detach(), dim=-1)
                        acc_val[0] = (preds == labels).sum().item() / max(labels.size(0), 1)
                        loss.backward()
                        return loss

                    if method in ["IAKF", "DISK", "DOPPLER", "FFTKF"]:
                        step_loss = optimizer.step(closure)
                        epoch_losses.append(step_loss.item())
                    else:
                        loss_val = closure()
                        optimizer.step()
                        epoch_losses.append(loss_val.item())

                    epoch_accs.append(acc_val[0])
                    if step % 20 == 0: epoch_snrs.append(compute_grad_snr(model))
                    optimizer.zero_grad()

            torch.cuda.empty_cache()
            max_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            
            val_metrics = evaluate_metrics(model, val_loader)
            eps = privacy_engine.get_epsilon(DP_CONFIG["target_delta"]) if method != "NO_DP" else 0.0

            history["train_loss"].append(np.mean(epoch_losses))
            history["train_accuracy"].append(np.mean(epoch_accs))
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_accuracy"].append(val_metrics["val_accuracy"])
            history["val_f1"].append(val_metrics["val_f1"])
            history["val_precision"].append(val_metrics["val_precision"])
            history["val_recall"].append(val_metrics["val_recall"])
            history["snr"].append(np.mean(epoch_snrs))
            history["gpu_memory_gb"].append(max_mem_gb)
            history["epsilon"].append(eps)

            mlflow.log_metrics({
                "train_loss": np.mean(epoch_losses),
                "val_loss": val_metrics["val_loss"],
                "val_accuracy": val_metrics["val_accuracy"],
                "val_f1": val_metrics["val_f1"],
                "val_precision": val_metrics["val_precision"],
                "val_recall": val_metrics["val_recall"],
                "snr": np.mean(epoch_snrs),
                "epsilon": eps,
                "gpu_memory_gb": max_mem_gb
            }, step=epoch)
            if method == "IAKF":
                iakf_stats = optimizer.get_and_reset_epoch_stats()
                for k, v in iakf_stats.items(): 
                    history[k].append(v)
                    mlflow.log_metric(f"iakf_{k}", v, step=epoch)


            print(f"  -> Epoch {epoch+1:02d} | Loss: {val_metrics['val_loss']:.4f} | Acc: {val_metrics['val_accuracy']:.4f} | F1: {val_metrics['val_f1']:.4f} | Prec: {val_metrics['val_precision']:.4f} | Rec: {val_metrics['val_recall']:.4f} | Eps: {eps:.2f}")

            if val_metrics["val_accuracy"] > best_val_acc:
                best_val_acc = val_metrics["val_accuracy"]
                epoch_best_acc = epoch + 1
                save_checkpoint(model, processor_or_tokenizer, os.path.join(exp_save_dir, "best_acc_model"))

            if val_metrics["val_f1"] > best_val_f1:
                best_val_f1 = val_metrics["val_f1"]
                epoch_best_f1 = epoch + 1
                save_checkpoint(model, processor_or_tokenizer, os.path.join(exp_save_dir, "best_f1_model"))
                epochs_no_improve = 0 
            else:
                epochs_no_improve += 1 

            final_val_acc, final_val_f1, final_eps = val_metrics["val_accuracy"], val_metrics["val_f1"], eps


            if epochs_no_improve >= patience:
                print(f"  -> [Early Stopping] Training interrupt at epoch {epoch+1}. F1-score does not envolve during {patience} epochs.")
                mlflow.set_tag("early_stopped", "true")
                mlflow.log_param("stopped_epoch", epoch + 1)
                break

        save_checkpoint(model, processor_or_tokenizer, os.path.join(exp_save_dir, "final_model"))
        
        history_path = os.path.join(exp_save_dir, "history.json")
        with open(history_path, "w") as f:
            json.dump(history, f)
        #mlflow.log_artifact(history_path)

        summary = {
            "exp_name": exp_name, "method": method, "dataset": DATASET_CONFIG["name"], "seed": seed,
            "best_val_acc": best_val_acc, "epoch_best_acc": epoch_best_acc,
            "best_val_f1": best_val_f1, "epoch_best_f1": epoch_best_f1,
            "final_val_acc": final_val_acc, "final_val_f1": final_val_f1,
            "final_epsilon": final_eps, "max_gpu_memory": max(history["gpu_memory_gb"])
        }

        if method == "IAKF" and iakf_config:
            for k, v in iakf_config.items():
                summary[f"iakf_{k}"] = v
        if method == "FFTKF" and fftkf_config:
            for k, v in fftkf_config.items():
                summary[f"fftkf_{k}"] = v

        mlflow.log_metrics({
            "best_val_acc_overall": best_val_acc,
            "best_val_f1_overall": best_val_f1
        })

        return history, summary