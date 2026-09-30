"""Обучение ResNet-18 на CIFAR-10 с MLflow tracking."""
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torchmetrics

class CIFARClassifier(pl.LightningModule):
    def __init__(self, lr=1e-3, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x): return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        loss = self.criterion(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.acc(logits, y), prog_bar=True)
        self.log("val_f1", self.f1(logits, y), prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
        return {"optimizer": opt, "lr_scheduler": sch}

def build_loaders(batch=256):
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    tf_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    train = datasets.ImageFolder("/data/cifar10/train", tf_train)
    val   = datasets.ImageFolder("/data/cifar10/val", tf_val)
    return (DataLoader(train, batch, shuffle=True, num_workers=4, pin_memory=True),
            DataLoader(val, batch, num_workers=4, pin_memory=True))

def main():
    mlf_logger = MLFlowLogger(
        experiment_name="deepvision",
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
    )
    ckpt = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1, filename="best-{epoch}-{val_acc:.4f}")
    early = EarlyStopping(monitor="val_acc", patience=5, mode="max")

    model = CIFARClassifier()
    train_loader, val_loader = build_loaders()

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="auto",
        devices=1,
        logger=mlf_logger,
        callbacks=[ckpt, early],
        log_every_n_steps=20,
    )
    trainer.fit(model, train_loader, val_loader)

    # логируем модель в MLflow Registry
    import mlflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.pytorch.log_model(
        model.model, "model",
        registered_model_name="deepvision-classifier",
    )
    print(f"✅ best checkpoint: {ckpt.best_model_path}")

if __name__ == "__main__":
    main()
