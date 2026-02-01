import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.transforms import Compose, ToTensord
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics.classification import MulticlassAUROC, MulticlassAveragePrecision

from renal_vision.end2end.dataset import create_smart_dataset
from renal_vision.end2end.model import build_model
from renal_vision.features.base_extractor import ZeroExtractor
from renal_vision.features.base_preprocessor import BasePreprocessor
from renal_vision.features.dataset import FeatureDatasetProcessor
from renal_vision.features.preprocessing import FMCIBPreprocessor
from renal_vision.shared.utils import describe_data, generate_patient_fold_mapping


def stable_hash(message) -> str:
    """Generates a md5 hash of the message"""
    key = str(message).encode("utf-8")
    return hashlib.md5(key).hexdigest()


def get_file_hash(filepath) -> str:
    """Generates a SHA256 hash of the file content."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read in chunks to avoid loading huge files into memory just to hash
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


class LesionClassifier(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        class_weights: torch.Tensor,
        use_pretrained: bool = False,
        max_epochs: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = build_model(num_classes, fmcib_pretrained=use_pretrained)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        self.num_classes = num_classes
        self.max_epochs = max_epochs

        # Metrics that accumulate across batches
        self.train_auroc = MulticlassAUROC(num_classes=num_classes, average=None)
        self.train_ap = MulticlassAveragePrecision(num_classes=num_classes, average=None)
        self.val_auroc = MulticlassAUROC(num_classes=num_classes, average=None)
        self.val_ap = MulticlassAveragePrecision(num_classes=num_classes, average=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.unsqueeze(1)
        return self.model(x)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        inputs, labels = batch["image"], batch["label"]
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels.long())
        probs = F.softmax(outputs, dim=1)

        self.log("train/loss", loss, on_step=True, on_epoch=True, batch_size=labels.size(0))

        # Log training metrics (compute at epoch end)
        self.train_auroc.update(probs, labels)
        self.train_ap.update(probs, labels)

        # Log learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train/lr", current_lr, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        # Compute AUROC on full training set (same as validation)
        auroc_per_class = self.train_auroc.compute()
        ap_per_class = self.train_ap.compute()

        # Log per-class metrics
        for i, (auc, ap) in enumerate(zip(auroc_per_class, ap_per_class)):
            self.log(f"train/AUC_{i}", auc)
            self.log(f"train/AP_{i}", ap)

        # Log macro average
        self.log("train/mean_AUC", auroc_per_class.mean())
        self.log("train/mean_AP", ap_per_class.mean())

        # Reset for next epoch
        self.train_auroc.reset()
        self.train_ap.reset()

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        inputs, labels = batch["image"], batch["label"]
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels.long())
        probs = F.softmax(outputs, dim=1)

        self.log(
            "val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(labels)
        )

        # Accumulate predictions across batches
        self.val_auroc.update(probs, labels)
        self.val_ap.update(probs, labels)

    def on_validation_epoch_end(self) -> None:
        # Compute metrics on full validation set
        auroc_per_class = self.val_auroc.compute()
        ap_per_class = self.val_ap.compute()

        # Log per-class metrics
        for i, (auc, ap) in enumerate(zip(auroc_per_class, ap_per_class)):
            self.log(f"val/AUC_{i}", auc)
            self.log(f"val/AP_{i}", ap)

        # Log macro averages
        self.log("val/mean_AUC", auroc_per_class.mean(), prog_bar=True)
        self.log("val/mAP", ap_per_class.mean(), prog_bar=True)

        # Reset metrics for next epoch
        self.val_auroc.reset()
        self.val_ap.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # AdamW with weight decay for better regularization
        optimizer = torch.optim.AdamW(self.parameters())

        # Warmup scheduler
        warmup_epochs = 5
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0 / warmup_epochs, end_factor=1.0, total_iters=warmup_epochs
        )

        # Cosine annealing scheduler
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs - warmup_epochs,
            eta_min=1e-6,
        )

        # Combine warmup + cosine
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
        )

        return {  # type: ignore [return-value]
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "warmup_cosine",
            },
        }


def create_dataloader(
    df: pd.DataFrame,
    preprocessor: BasePreprocessor,
    min_volume: int,
    batch_size: int,
    class_counts: List[int],
    is_train: bool = False,
    cache_dir: Optional[Path] = None,
    cache_rate: float = 0,
    num_workers: int = 4,
):
    required_keys = ["image_path", "seg_path", "lesion_id", "class_id", "case"]
    data = df[required_keys].to_dict("records")

    transform = Compose(
        [
            preprocessor.LoadCase(),  # ← Cached
            preprocessor.SelectComponent(min_volume=min_volume, augment=is_train),  # ← NOT cached
            ToTensord(keys=["image"]),
        ]
    )

    if cache_dir:
        cache_dir = cache_dir / "preprocessed_images"
    dataset = create_smart_dataset(
        data=data, transform=transform, cache_dir=cache_dir, cache_rate=cache_rate, verbose=True
    )

    weights = [
        sum(class_counts) / (len(class_counts) * class_counts[int(l)]) for l in df["class_id"]
    ]

    if is_train:
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        loader = DataLoader(
            dataset,
            batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loader


def train(
    csv_path: Path,
    output_dir: Path,
    cache_dir: Path,
    preprocessor: BasePreprocessor = FMCIBPreprocessor(),
    min_volume: int = 400,
    num_workers: int = 4,
    batch_size: int = 8,
    epochs: int = 10,
    lr: float = 1e-4,
    pretrained: bool = False,
    cache_rate: float = 0,
    no_image_caching: bool = False,
    validate: bool = True,
):
    if not csv_path.exists():
        raise Exception(f"File not found {csv_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        torch.set_float32_matmul_precision("high")
    except Exception as e:
        print("Warning: couldnt set float32 matmul precision", e)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #### Scan Data ####
    print(f"Scanning {csv_path} before training...")
    file_hash = get_file_hash(csv_path)[:10]
    parquet_path = cache_dir / f"dataset_{file_hash}.parquet"

    if parquet_path.exists():
        print(f"Cache hit! Loading {parquet_path.name}")
    else:
        print("Cache miss. Processing CSV...")

        processor = FeatureDatasetProcessor(ZeroExtractor(preprocessor, min_volume=400))
        processor.process_dataset(
            input_df=pd.read_csv(csv_path),
            output_path=parquet_path,
            augment_count=0,
            num_jobs=max(1, num_workers // 2),
            cores_per_job=min(num_workers, 2),
        )
        print(f"Saved cache to {parquet_path}")
    df = pd.read_parquet(parquet_path)

    # Split data into train and validation folds
    if validate:
        fold_map = generate_patient_fold_mapping(
            df, group_col="case", stratify_col="class_id", n_folds=10
        )
        df["fold"] = df["case"].astype(str).map(fold_map)

        val_fold = 0
        train = df[df["fold"] != val_fold].reset_index(drop=True)
        val = df[df["fold"] == val_fold].reset_index(drop=True)

        print("Train Data:")
        describe_data(train)
        print("\nValidation Data:")
        describe_data(val)
    else:
        train = df
        print("Train Data (No Validation):")
        describe_data(train)

    # DEBUG
    # print("DEBUG: shorten dataset")
    # train = df[df["fold"] == 1].reset_index(drop=True)

    total_lesions = len(train)
    classes = list(df["class_id"].unique())
    classes.sort()
    num_classes = len(classes)
    class_counts = [(train["class_id"] == cid).sum() for cid in classes]
    class_weights = torch.tensor(
        [total_lesions / (num_classes * count) for count in class_counts], dtype=torch.float32
    ).to(device)

    # Initialize training
    model = LesionClassifier(
        num_classes=num_classes,
        class_weights=class_weights,
        use_pretrained=pretrained,
        max_epochs=epochs,
    )

    train_loader = create_dataloader(
        df=train,
        preprocessor=preprocessor,
        min_volume=min_volume,
        batch_size=batch_size,
        class_counts=class_counts,
        is_train=True,
        cache_dir=cache_dir if not no_image_caching else None,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )
    val_loader = None
    if validate:
        val_loader = create_dataloader(
            df=val,
            preprocessor=preprocessor,
            min_volume=min_volume,
            batch_size=batch_size,
            class_counts=class_counts,
            is_train=False,
            cache_dir=cache_dir if not no_image_caching else None,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

    # Logger
    wandb_logger = WandbLogger(
        project="lesion-classifier",
        config={
            "batch_size": batch_size,
            "learning_rate": lr,
            "epochs": epochs,
            "model": "ResNet50-3D",
            "total_lesions": total_lesions,
            "validation_enabled": validate,
            "pretrained": pretrained,
            "cache_rate": cache_rate,
        },
    )

    checkpoint_last = ModelCheckpoint(
        save_last=True,
        filename="last-{epoch}",
        dirpath=output_dir,
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_last],
        accelerator=device,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    print(f"Last model saved at: {checkpoint_last.last_model_path}")
