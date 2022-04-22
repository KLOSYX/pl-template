from argparse import ArgumentParser
from click import progressbar
from matplotlib.pyplot import text

import pytorch_lightning as pl
import torch
from torch.functional import F
from torch import nn
from transformers import (
    BertForPreTraining,
    BertConfig,
    BertTokenizer,
    get_cosine_schedule_with_warmup,
)

from utils.focal_loss import FocalLoss


class MLP(pl.LightningDataModule):
    def __init__(self,
                 mlp_dropout=0.5,
                 **args):
        super().__init__()
        self.save_hyperparameters()
        self.mlp = nn.Sequential(
            nn.Linear(2048, 4 * 2048),
            F.dropout(mlp_dropout),
            nn.Linear(4 * 2048, 2048),
        )

    def forward(self, img_features):
        out = self.mlp(img_features)
        return out


class GaiicBertClassifer(pl.LightningModule):
    def __init__(
        self,
        model_name,
        vocab_size,
        padding_idx,
        train_data_size,
        lr=1e-4,
        dropout=0.2,
        image_token_size=1,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        # bert_config = BertConfig.from_pretrained(
        #     model_name, cache_dir="~/.cache")
        # hidden_size = bert_config.hidden_size
        # self.text_embed = nn.Embedding(vocab_size, hidden_size, padding_idx)
        # self.img_proj = nn.Linear(2048, self.hparams.image_token_size * hidden_size)

        self.transformer = BertForPreTraining.from_pretrained(
            model_name, cache_dir="~/.cache")
        hidden_size = self.transformer.bert.pool
        self.mlp = MLP(mlp_dropout=self.hparams.mlp_dropout)
        self.classifer = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(dropout)

        self.criterion = FocalLoss(alpha=[1, 1], num_classes=2)

    def forward(self, tokens, img_features, attention_mask, token_type_ids):
        # img_features = img_features.unsqueeze(1)  # [N, 1, 2048]
        img_features = self.mlp(img_features)
        text_features = self.transformer(
            input_ids=tokens,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).pooler_output
        # try pooler output
        # x = outputs.pooler_output

        x = torch.cat([text_features, img_features], dim=-1)
        x = self.dropout(x)
        x = self.classifer(x)

        # try get average of all hidden state
        # x = outputs[0]
        # x = (x * attention_mask.unsqueeze(2)).sum(dim=1) / attention_mask.sum(
        #     dim=1
        # )[:, None]

        # x = self.dropout(x)
        # x = self.classifer(x)
        return x

    def training_step(self, batch):
        encodings, targets = batch
        outputs = self(*encodings)
        loss = self.criterion(outputs, targets)

        with torch.no_grad():
            predicts = torch.argmax(outputs, dim=1)
            # lr = self.optimizer.param_groups[0]["lr"]
            train_loss = loss.cpu().item()
            train_acc = predicts.cpu().eq(targets.cpu()).sum().item() / targets.size(0)

        self.log_dict(
            {"train_loss": train_loss, "train_acc": train_acc, },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def training_epoch_end(self, outputs) -> None:
        print("")

    def configure_optimizers(self):
        step_per_epoch = self.hparams.train_data_size // self.hparams.batch_size
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            5 * step_per_epoch,
            self.hparams.max_epochs * step_per_epoch,
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def validation_step(self, batch, batch_idx):
        encodings, targets = batch
        encodings = [x.to(self.device) for x in encodings]

        targets = targets.to(self.device).reshape(-1)
        outputs = self(*encodings)
        loss = self.criterion(outputs, targets)
        predicts = torch.argmax(outputs, dim=1)
        val_acc = predicts.cpu().eq(targets.cpu()).sum().item() / targets.size(0)
        val_loss = loss.cpu().item()

        self.log_dict(
            {"val_loss": val_loss, "val_acc": val_acc},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def validation_step_end(self, outputs):
        print("")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--model_name", type=str,
                            default="bert-base-chinese")
        parser.add_argument("--dropout", type=float, default=0.2)
        parser.add_argument("--image_token_size", type=int, default=1)
        return parser
