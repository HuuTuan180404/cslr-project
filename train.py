import argparse
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader

from datasets.isharah500 import ISharah500Dataset
from utils.visualization import visualize_pose_sequence, visualize_rgb_sequence

from models.image_encoder import ImageEncoder

# ============================================================
# Utils
# ============================================================
def load_config(config_path: str) -> DictConfig:
    """Load YAML config and resolve variable interpolation."""

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file does not exist: {config_path}"
        )

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
    device = config["training"]["device"]

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        return torch.device("cpu")

    return torch.device(device)


# ============================================================
# Dataset
# ============================================================
def build_dataset(config, phase: str):
    dataset_config = config["dataset"]

    dataset_name = dataset_config["benchmark"].lower()

    if dataset_name == "isharah500":
        pass

    return ISharah500Dataset(args=config, phase=phase)


# ============================================================
# DataLoader
# ============================================================

# def build_dataloader(dataset, config, split: str):
#     dataloader_config = config["dataloader"]

#     is_train = split == "train"

#     return DataLoader(dataset, batch_size=dataloader_config["batch_size"], shuffle=dataloader_config["shuffle"] if is_train else False, num_workers=dataloader_config["num_workers"], pin_memory=dataloader_config["pin_memory"], drop_last=dataloader_config["drop_last"] if is_train else False, persistent_workers=(dataloader_config["persistent_workers"]
#             and dataloader_config["num_workers"] > 0
# ))


# ============================================================
# Model
# ============================================================

# def build_model(config):
#     model_config = config["model"]

#     model_name = model_config["name"].lower()

#     if model_name == "siformer":
#         from models.siformer import Siformer

#         model = Siformer(num_classes=model_config["num_classes"], num_hid=model_config["input_dim"])

#         return model

#     raise ValueError(f"Unknown model: {model_name}")


# ============================================================
# Loss
# ============================================================

# def build_loss(config):
#     loss_config = config["loss"]

#     if loss_config["name"].lower() == "cross_entropy":
#         return torch.nn.CrossEntropyLoss(label_smoothing=loss_config.get("label_smoothing", 0.0))

#     raise ValueError(f"Unknown loss: {loss_config['name']}")


# ============================================================
# Optimizer
# ============================================================

# def build_optimizer(model, config):
#     optimizer_config = config["optimizer"]

#     name = optimizer_config["name"].lower()

#     if name == "adamw":
#         return torch.optim.AdamW(model.parameters(), 
#                                  lr=optimizer_config["learning_rate"], 
#                                  weight_decay=optimizer_config["weight_decay"], 
#                                  betas=tuple(optimizer_config.get("betas", [0.9, 0.999])), 
#                                  eps=optimizer_config.get("eps", 1e-8))

#     raise ValueError(f"Unknown optimizer: {optimizer_config['name']}")


# ============================================================
# Training
# ============================================================

# def train_one_epoch(model, dataloader, criterion, optimizer, device):
#     model.train()

#     total_loss = 0.0
#     total_samples = 0
#     correct = 0

#     for batch in dataloader:

#         # ----------------------------------------------------
#         # TODO:
#         # Điều chỉnh phần này theo output thực tế của Dataset
#         # ----------------------------------------------------

#         inputs = batch["pose"]
#         labels = batch["label"]

#         inputs = inputs.to(device)
#         labels = labels.to(device)

#         optimizer.zero_grad()

#         outputs = model(inputs)

#         loss = criterion(outputs, labels)

#         loss.backward()
#         optimizer.step()

#         # ----------------------------------------------------
#         # Statistics
#         # ----------------------------------------------------

#         batch_size = labels.size(0)

#         total_loss += loss.item() * batch_size
#         total_samples += batch_size

#         predictions = outputs.argmax(dim=1)

#         correct += (predictions == labels).sum().item()

#     avg_loss = total_loss / total_samples
#     accuracy = correct / total_samples

#     return avg_loss, accuracy


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

    seed = config.project.seed
    set_seed(seed)

    # device = get_device(config)

    print("=" * 60, "Training configuration", "=" * 60)

    print("=" * 60, f"Dataset : {config['dataset']['benchmark']}", "=" * 60)

    # print(f"Model   : {config['model']['name']}")
    # print(f"Device  : {device}")
    # print(f"Seed    : {seed}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = build_dataset(config, phase="train")
    # dev_dataset = build_dataset(config, phase="dev")
    # test_dataset = build_dataset(config, phase="test")

    sample = train_dataset[0]
    lh_pose = sample["left"]
    rh_pose = sample["right"]

    lh_rgb = sample["rgb_left"]
    rh_rgb = sample["rgb_right"]

    face = sample["face"]
    body = sample["body"]

    model = ImageEncoder(config=config)
    
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