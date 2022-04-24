from argparse import ArgumentParser
from sched import scheduler
from typing import Any
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from utils.focal_loss import FocalLoss
from transformers import get_cosine_schedule_with_warmup
import nni
import torchmetrics
import pytorch_lightning as pl
import torch.nn as nn
import torch


class VideoHw1Cnn(pl.LightningModule):
    def __init__(self, num_classes=8, **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 8, (3, 3), stride=1, padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, (3, 3), stride=1, padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, (3, 3), stride=1, padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (3, 3), stride=1, padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, (3, 3), stride=1, padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(128 * 8 * 8, 64)
        self.batch_norm = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(self.hparams.dropout)
        self.act = nn.ReLU()
        self.classifier = nn.Linear(64, 8)
        
        self.criterion = nn.BCEWithLogitsLoss()

        self.train_auroc = torchmetrics.AUROC(
            num_classes=self.hparams.num_classes)
        self.val_auroc = torchmetrics.AUROC(
            num_classes=self.hparams.num_classes)
        self.train_f1_score = torchmetrics.F1Score(self.hparams.num_classes)
        self.val_f1_score = torchmetrics.F1Score(self.hparams.num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.batch_norm(x)
        x = self.act(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels.to(torch.float))

        self.train_auroc(logits, labels)
        self.train_f1_score(logits, labels)
        self.log_dict({'train_loss': loss}, on_step=True, on_epoch=True)
        self.log_dict({'train_auroc': self.train_auroc,
                       'train_f1_score': self.train_f1_score},
                      on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels.to(torch.float))
        self.val_auroc(logits, labels)
        self.val_f1_score(logits, labels)
        self.log_dict(
            {'val_loss': loss, 'val_auroc': self.val_auroc,
             'val_f1_score': self.val_f1_score}, on_step=False, on_epoch=True)

    def validation_epoch_end(self, output) -> None:
        if self.hparams.use_nni:
            nni.report_intermediate_result(self.val_f1_score.compute().item())

    def configure_optimizers(self):
        max_step = (int(0.9 * 139) // self.hparams.batch_size) * self.hparams.max_epochs
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            0.1 * max_step,
            max_step
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--dropout', type=float, default=0.5)
        return parser


if __name__ == '__main__':
    x = torch.randn(64, 3, 224, 224)
    model = VideoHw1Cnn()
    out = model(x)
