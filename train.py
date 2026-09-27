import argparse
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader

from datasets.utils import cslr_collate_fn
from datasets.isharah500 import ISharah500Dataset
import torch.nn as nn

from models.model import CSLRModel

from utils.utils import get_config


# ============================================================
# Utils
# ============================================================
def load_config(config_path: str) -> DictConfig:
    """Load YAML config and resolve variable interpolation."""

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    config = OmegaConf.load(config_path)

    # Resolve ${...} references
    OmegaConf.resolve(config)

    return config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(config):
    device = config.training.device

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        return torch.device("cpu")

    return torch.device(device)


# ============================================================
# Dataset
# ============================================================
def build_dataset(config, phase: str):
    dataset_name = config.dataset.benchmark.lower()

    if dataset_name == "isharah500":
        pass

    return ISharah500Dataset(config=config, phase=phase)


# ============================================================
# DataLoader
# ============================================================
def build_dataloader(dataset, config, phase: str, collate_fn=None):
    is_train = phase == "train"

    return DataLoader(
        dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=config.dataloader.shuffle if is_train else False,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        persistent_workers=(
            config.dataloader.persistent_workers and config.dataloader.num_workers > 0
        ),
        collate_fn=collate_fn,
    )


# ============================================================
# Model
# ============================================================


def build_model(config) -> nn.Module:
    model_name = config.model.name.lower()
    if model_name == "mymodel":
        model = CSLRModel(config=config)
        return model
    raise ValueError(f"Unknown model: {model_name}")


# ============================================================
# Loss
# ============================================================


def build_loss(config):
    if config.loss.name.lower() == "cross_entropy":
        return torch.nn.CrossEntropyLoss(
            label_smoothing=config.loss.get("label_smoothing", 0.0)
        )

    raise ValueError(f"Unknown loss: {config.loss.name}")


# ============================================================
# Optimizer
# ============================================================
def build_optimizer(model, config):
    optimizer_config = config.optimizer

    name = optimizer_config.name.lower()

    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay,
            betas=tuple(optimizer_config.get("betas", [0.9, 0.999])),
            eps=optimizer_config.get("eps", 1e-8),
        )

    raise ValueError(f"Unknown optimizer: {optimizer_config.name}")


# ============================================================
# Training
# ============================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_samples = 0
    correct = 0

    for batch in dataloader:
        # ----------------------------------------------------
        # TODO:
        # Điều chỉnh phần này theo output thực tế của Dataset
        # ----------------------------------------------------
        lh_pose = batch["left"].to(device)
        rh_pose = batch["right"].to(device)
        lh_rgb = batch["rgb_left"].to(device) if batch["rgb_left"] is not None else None
        rh_rgb = (
            batch["rgb_right"].to(device) if batch["rgb_right"] is not None else None
        )
        face = batch["face"].to(device) if batch["face"] is not None else None
        body = batch["body"].to(device) if batch["body"] is not None else None
        gloss_ids = batch["gloss_ids"].to(device)
        text_attention_mask = batch["text_attention_mask"].to(device)

        # optimizer.zero_grad()

        outputs = model(
            lh_pose, rh_pose, lh_rgb, rh_rgb, face, body, gloss_ids, text_attention_mask
        )

        break

        # loss = criterion(outputs, labels)

        # loss.backward()
        # optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

    #     batch_size = labels.size(0)

    #     total_loss += loss.item() * batch_size
    #     total_samples += batch_size

    #     predictions = outputs.argmax(dim=1)

    #     correct += (predictions == labels).sum().item()

    # avg_loss = total_loss / total_samples
    # accuracy = correct / total_samples

    # return avg_loss, accuracy


# ============================================================
# Validation
# ============================================================
@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    correct = 0

    for batch in dataloader:
        inputs = batch["pose"]
        labels = batch["label"]

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples

    return avg_loss, accuracy


# ============================================================
# Main
# ============================================================


def main(config_path: str):

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------
    config = load_config(config_path)
    max_frames = config.dataset.max_frames
    seed = config.project.seed

    set_seed(seed)

    device = get_device(config)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    train_dataset = build_dataset(config, phase="train")
    # dev_dataset = build_dataset(config, phase="dev")
    # test_dataset = build_dataset(config, phase="test")

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------
    train_loader = build_dataloader(
        train_dataset,
        config,
        "train",
        lambda batch: cslr_collate_fn(
            batch, vocab=train_dataset.vocab, max_frames=max_frames
        ),
    )
    # dev_loader = build_dataloader(
    #     dev_dataset,
    #     config,
    #     "dev",
    #     lambda batch: cslr_collate_fn(
    #         batch, vocab=train_dataset.vocab, max_frames=max_frames
    #     ),
    # )

    # batch = next(iter(train_loader))

    # print(f"Train dataset size: {len(train_dataset)}")
    # print(f"Train batches: {len(train_loader)}")

    # lh_pose = sample["left"].unsqueeze(0)
    # rh_pose = sample["right"].unsqueeze(0)

    # lh_rgb = sample["rgb_left"].unsqueeze(0)
    # rh_rgb = sample["rgb_right"].unsqueeze(0)

    # face = sample["face"].unsqueeze(0)
    # body = sample["body"].unsqueeze(0)

    model = build_model(config=config)
    model = model.to(device)

    train_one_epoch(model, train_loader, None, None, device)

    # print("=" * 10, output.shape)  # <class 'dict'>

    # print("="*10, type(sample)) # <class 'dict'>
    # print("="*10, sample.keys()) # dict_keys(['id', 'gloss', 'text', 'use_rgb', 'right', 'left', 'face', 'body', 'rgb_left', 'rgb_right'])

    # print("="*10, sample["left"].shape) # (T, H, W, 3)
    # print("="*10, sample["rgb_left"].shape) # (T, H, W, 3)

    # print("="*10, xxxxxx) # xxxxx
    # print("="*10, xxxxxx) # xxxxx
    # print("="*10, xxxxxx) # xxxxx
    # print("="*10, xxxxxx) # xxxxx

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    # train_loader = build_dataloader(train_dataset, config, split="train")

    # val_loader = build_dataloader(val_dataset, config, split="val")

    # print(f"Train batches: {len(train_loader)}")
    # print(f"Val batches  : {len(val_loader)}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    # model = build_model(config)

    # model = model.to(device)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    # criterion = build_loss(config)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    # optimizer = build_optimizer(model, config)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    # num_epochs = config["training"]["epochs"]

    # best_accuracy = 0.0

    # for epoch in range(1, num_epochs + 1):

    #     train_loss, train_acc = train_one_epoch(model=model, dataloader=train_loader, criterion=criterion, optimizer=optimizer, device=device)

    #     val_loss, val_acc = validate(model=model, dataloader=val_loader, criterion=criterion, device=device)

    #     print(
    #         f"Epoch [{epoch:03d}/{num_epochs:03d}] "
    #         f"| Train Loss: {train_loss:.4f} "
    #         f"| Train Acc: {train_acc * 100:.2f}% "
    #         f"| Val Loss: {val_loss:.4f} "
    #         f"| Val Acc: {val_acc * 100:.2f}%"
    #         )

    #     # ----------------------------------------------------
    #     # Save best model
    #     # ----------------------------------------------------

    #     if val_acc > best_accuracy:

    #         best_accuracy = val_acc

    #         save_dir = Path(config["checkpoint"]["save_dir"])

    #         save_dir.mkdir(parents=True, exist_ok=True)

    #         save_path = save_dir / config["checkpoint"]["filename"]

    #         torch.save({
    #                 "epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "val_accuracy": val_acc, "config": config, }, save_path)

    #         print(f"  → Best model saved: {save_path}")


# def test_model(config_path: str):
#     config = load_config(config_path)
#     model = ImageEncoder(config=config)

#     batch_size = 2
#     lh


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="./configs/isharah500.yaml")

    args = parser.parse_args()

    main(args.config)
