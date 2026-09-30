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
from losses.total_loss import TotalLoss


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


def build_dataset(config, phase: str):
    dataset_name = config.dataset.benchmark.lower()

    if dataset_name == "isharah500":
        pass

    return ISharah500Dataset(config=config, phase=phase)


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


def build_model(cfg) -> nn.Module:
    model_name = cfg.model.name.lower()
    if model_name == "mymodel":
        model = CSLRModel(cfg=cfg)
        return model
    raise ValueError(f"Unknown model: {model_name}")


def build_loss(config):
    if config.loss.name.lower() == "cross_entropy":
        return torch.nn.CrossEntropyLoss(
            label_smoothing=config.loss.get("label_smoothing", 0.0)
        )

    raise ValueError(f"Unknown loss: {config.loss.name}")


def build_optimizer(model, cfg):
    optimizer_config = cfg.optimizer

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


def edit_distance(pred, target):
    n = len(target)
    m = len(pred)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if target[i - 1] == pred[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = (
                    min(
                        dp[i - 1][j],  # deletion
                        dp[i][j - 1],  # insertion
                        dp[i - 1][j - 1],  # substitution
                    )
                    + 1
                )

    return dp[n][m]


def train_one_epoch(model, dataloader, criterion, optimizer, device, blank_id):
    model.train()

    total_loss = 0
    total_samples = 0

    total_errors = 0
    total_words = 0

    for batch in dataloader:
        # ----------------------------------------------------
        # Input
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

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------
        outputs = model(
            lh_pose, rh_pose, lh_rgb, rh_rgb, face, body, gloss_ids, text_attention_mask
        )

        # return 0, 0

        # ----------------------------------------------------
        # CTC targets
        # ----------------------------------------------------
        target_ids = torch.cat(batch["target_ids"]).to(device)

        input_lengths = batch["input_lengths"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------
        loss_dict = criterion(outputs, target_ids, input_lengths, target_lengths)

        loss = loss_dict["loss"]

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Loss statistics
        # ----------------------------------------------------
        batch_size = lh_pose.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # ====================================================
        # CTC prediction
        # ====================================================

        # [B, T, C]
        ctc_logits = outputs["ctc_logits"]

        # [B, T]
        predictions = ctc_logits.argmax(dim=-1)

        for i in range(batch_size):
            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            pred = predictions[i]

            # Remove padding timesteps
            pred = pred[: input_lengths[i]]

            # CTC collapse:
            # [A, A, B, B, B, C] -> [A, B, C]
            pred = torch.unique_consecutive(pred)

            # Remove CTC blank
            pred = pred[pred != blank_id]

            pred = pred.cpu().tolist()

            # ------------------------------------------------
            # Ground truth
            # ------------------------------------------------

            start = target_lengths[:i].sum().item()

            end = start + target_lengths[i].item()

            target = target_ids[start:end].cpu().tolist()

            # ------------------------------------------------
            # Edit distance
            # ------------------------------------------------

            errors = edit_distance(pred, target)

            total_errors += errors
            total_words += len(target)

    # --------------------------------------------------------
    # Epoch statistics
    # --------------------------------------------------------

    avg_loss = total_loss / total_samples

    wer = total_errors / total_words if total_words > 0 else 0.0

    return avg_loss, wer


@torch.no_grad()
def validate(model, dataloader, criterion, device, blank_id):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    total_errors = 0
    total_words = 0

    for batch in dataloader:
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

        outputs = model(
            lh_pose, rh_pose, lh_rgb, rh_rgb, face, body, gloss_ids, text_attention_mask
        )

        target_ids = torch.cat(batch["target_ids"]).to(device)
        input_lengths = batch["input_lengths"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        loss_dict = criterion(outputs, target_ids, input_lengths, target_lengths)

        loss = loss_dict["loss"]

        batch_size = lh_pose.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        ctc_logits = outputs["ctc_logits"]

        # [B, T, C]
        predictions = ctc_logits.argmax(dim=-1)

        for i in range(batch_size):
            pred = predictions[i]

            # Only use valid timesteps
            pred = pred[: input_lengths[i]]

            pred = torch.unique_consecutive(pred)

            # Remove blank
            pred = pred[pred != blank_id]

            pred = pred.cpu().tolist()

            # ---------------------------------
            # Ground truth
            # ---------------------------------
            start = target_lengths[:i].sum().item()
            end = start + target_lengths[i].item()

            target = target_ids[start:end].cpu().tolist()

            # ---------------------------------
            # Edit distance
            # ---------------------------------
            errors = edit_distance(pred, target)

            total_errors += errors
            total_words += len(target)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    wer = total_errors / total_words if total_words > 0 else 0.0

    return avg_loss, wer


def main(config_path: str):
    cfg = load_config(config_path)
    max_frames = cfg.dataset.max_frames
    seed = cfg.project.seed

    set_seed(seed)

    device = get_device(cfg)

    train_dataset = build_dataset(cfg, phase="train")
    dev_dataset = build_dataset(cfg, phase="dev")
    # test_dataset = build_dataset(cfg, phase="test")

    train_loader = build_dataloader(
        train_dataset,
        cfg,
        "train",
        lambda batch: cslr_collate_fn(
            batch, vocab=train_dataset.vocab, max_frames=max_frames
        ),
    )
    dev_loader = build_dataloader(
        dev_dataset,
        cfg,
        "dev",
        lambda batch: cslr_collate_fn(
            batch, vocab=train_dataset.vocab, max_frames=max_frames
        ),
    )

    model = build_model(cfg=cfg)
    model = model.to(device)

    criterion = TotalLoss(cfg)

    optimizer = build_optimizer(model, cfg)

    num_epochs = cfg.training.epochs

    for epoch in range(1, num_epochs + 1):
        train_loss, train_wer = train_one_epoch(
            model, train_loader, criterion, optimizer, device, train_dataset.blank_id
        )

        dev_loss, dev_wer = validate(
            model, dev_loader, criterion, device, train_dataset.blank_id
        )
        print(
            f"Epoch [{epoch:03d}/{num_epochs:03d}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train WER: {train_wer * 100:.2f}% "
            f"| Val Loss: {dev_loss:.4f} "
            f"| Val WER: {dev_wer * 100:.2f}%"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------
        # if dev_acc > best_accuracy:
        #     best_accuracy = dev_acc

        #     save_dir = Path(cfg.checkpoint.save_dir)

        #     save_dir.mkdir(parents=True, exist_ok=True)

        #     save_path = save_dir / cfg.checkpoint.filename

        #     torch.save(
        #         {
        #             "epoch": epoch,
        #             "model_state_dict": model.state_dict(),
        #             "optimizer_state_dict": optimizer.state_dict(),
        #             "val_accuracy": dev_acc,
        #             "config": cfg,
        #         },
        #         save_path,
        #     )

        #     print(f"  → Best model saved: {save_path}")

        # break


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
