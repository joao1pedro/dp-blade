import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomCrop
from datasets import load_dataset
from src.config import TASK_TYPE

def get_train_augmentations(dataset_name):
    if dataset_name == "cifar10":
        return Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
        ])
    elif dataset_name == "mnist":
        return Compose([
            RandomCrop(28, padding=2),
        ])
    
    return Compose([])

def build_loaders(dataset_config, processor_or_tokenizer):
    d_name = dataset_config["name"]

    if d_name == "cifar10":
        dataset = load_dataset("cifar10")
        input_col = "img"
        label_col = "label"
        ds_train, ds_val = dataset["train"], dataset["test"]
    elif d_name == "mnist":
        dataset = load_dataset("mnist")
        input_col = "image"
        label_col = "label"
        ds_train, ds_val = dataset["train"], dataset["test"]
    elif d_name == "sst2":
        dataset = load_dataset("glue", "sst2")
        input_col = "sentence"
        label_col = "label"
        ds_train, ds_val = dataset["train"], dataset["validation"]
    elif d_name == "imdb":
        dataset = load_dataset("imdb")
        input_col = "text"
        label_col = "label"
        ds_train, ds_val = dataset["train"], dataset["test"]
    elif d_name == "qnli":
        dataset = load_dataset("glue", "qnli")
        input_col = ["question", "sentence"]
        label_col = "label"
        ds_train, ds_val = dataset["train"], dataset["validation"]
    elif d_name == "mnli":
        dataset = load_dataset("glue", "mnli")
        input_col = ["premise", "hypothesis"]
        label_col = "label"
        ds_train, ds_val = dataset["train"], dataset["validation_matched"]
    else:
        dataset = load_dataset(d_name)
        input_col = "text"
        label_col = "label"
        ds_train, ds_val = dataset["train"], dataset["test"]

    if dataset_config.get("train_subset_size"):
        ds_train = ds_train.shuffle(seed=42).select(range(dataset_config["train_subset_size"]))
    if dataset_config.get("val_subset_size"):
        ds_val = ds_val.shuffle(seed=42).select(range(dataset_config["val_subset_size"]))

    train_aug = get_train_augmentations(d_name)

    def transform_train(examples):
        if TASK_TYPE == "image":
            augmented_images = [train_aug(img.convert("RGB")) for img in examples[input_col]]
            inputs = processor_or_tokenizer(augmented_images, return_tensors="pt")
        else:
            if isinstance(input_col, list):
                inputs = processor_or_tokenizer(examples[input_col[0]], examples[input_col[1]], padding="max_length", truncation=True, return_tensors="pt")
            else:
                inputs = processor_or_tokenizer(examples[input_col], padding="max_length", truncation=True, return_tensors="pt")
            
        inputs["label"] = examples[label_col]
        return inputs

    def transform_val(examples):
        if TASK_TYPE == "image":
            clean_images = [img.convert("RGB") for img in examples[input_col]]
            inputs = processor_or_tokenizer(clean_images, return_tensors="pt")
        else:
            if isinstance(input_col, list):
                inputs = processor_or_tokenizer(examples[input_col[0]], examples[input_col[1]], padding="max_length", truncation=True, return_tensors="pt")
            else:
                inputs = processor_or_tokenizer(examples[input_col], padding="max_length", truncation=True, return_tensors="pt")
            
        inputs["label"] = examples[label_col]
        return inputs

    ds_train.set_transform(transform_train)
    ds_val.set_transform(transform_val)

    def collate_fn(examples):
        if TASK_TYPE == "image":
            input_values = torch.stack([example["pixel_values"] for example in examples])
            key_name = "pixel_values"
        else:
            input_values = torch.stack([example["input_ids"] for example in examples])
            key_name = "input_ids"
            
        labels = torch.tensor([example["label"] for example in examples])
        return {key_name: input_values, "label": labels}

    train_loader = DataLoader(ds_train, batch_size=dataset_config["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(ds_val, batch_size=dataset_config["val_batch_size"], collate_fn=collate_fn)

    return train_loader, val_loader