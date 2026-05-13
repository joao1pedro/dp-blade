import os
import torch
import numpy as np
import evaluate
from src.config import DEVICE, TASK_TYPE

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def evaluate_metrics(model, loader):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    
    with torch.no_grad():
        for batch in loader:
            if TASK_TYPE == "image":
                inputs = batch["pixel_values"].to(DEVICE)
                outputs = model(pixel_values=inputs, labels=batch["label"].to(DEVICE))
            else:
                inputs = batch["input_ids"].to(DEVICE)
                outputs = model(input_ids=inputs, labels=batch["label"].to(DEVICE))
                
            labels = batch["label"].to(DEVICE)
            total_loss += outputs.loss.item()
            all_preds.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return (
        total_loss / len(loader),
        accuracy_metric.compute(predictions=all_preds, references=all_labels)["accuracy"],
        f1_metric.compute(predictions=all_preds, references=all_labels, average="weighted")["f1"]
    )

def compute_grad_snr(model):
    snr_list = []
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                g = p.grad.float()
                std_g = g.std().item()
                if std_g > 1e-12:
                    snr_list.append(g.abs().mean().item() / std_g)
    return np.mean(snr_list) if snr_list else 0.0

def save_checkpoint(model, processor_or_tokenizer, path):
    os.makedirs(path, exist_ok=True)
    unwrapped = model._module if hasattr(model, "_module") else model
    unwrapped.save_pretrained(path, safe_serialization=True)
    processor_or_tokenizer.save_pretrained(path)