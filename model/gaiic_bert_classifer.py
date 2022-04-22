from argparse import ArgumentParser
from click import progressbar
from matplotlib.pyplot import text

import pytorch_lightning as pl
import torch
from torch import nn
from transformers import (
    BertModel,
    BertConfig,
    BertTokenizer,
    get_cosine_schedule_with_warmup,
)

from utils.focal_loss import FocalLoss


class GaiicBertClassifer(pl.LightningModule):
    def __init__(
        self,
        model_name,
        vocab_size,
        padding_idx,
        train_data_size,
        lr=1e-4,
        dropout=0.2,
        image_token_size=2,
        focal_loss=False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        bert_config = BertConfig.from_pretrained(
            model_name, cache_dir="~/.cache")
        hidden_size = bert_config.hidden_size
        self.text_embed = nn.Embedding(vocab_size, hidden_size, padding_idx)
        self.img_proj = nn.Linear(
            2048, self.hparams.image_token_size * hidden_size)
        self.transformer = BertModel(bert_config)
        self.classifer = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(dropout)

        self.criterion = FocalLoss(
            alpha=[1, 1], num_classes=2) if self.hparams.focal_loss else nn.CrossEntropyLoss()

    def forward(self, tokens, img_features, attention_mask, token_type_ids):
        # img_features = img_features.unsqueeze(1)  # [N, 1, 2048]
        B = img_features.shape[0]
        img_features = self.img_proj(img_features)  # [N, n * hidden_size]
        img_features = img_features.reshape(
            B, self.hparams.image_token_size, -1)  # [N, n, hidden_size]
        text_features = self.text_embed(tokens)  # [N, L, hidden_size]
        # [N, L-n, hidden_size]
        text_features = text_features[:, :-self.hparams.image_token_size, :]
        attention_mask[:, -self.hparams.image_token_size:] = 1.
        token_type_ids[:, -self.hparams.image_token_size:] = 1
        joint_features = torch.cat(
            [text_features, img_features], dim=1
        )  # [N, L, hidden_size]
        outputs = self.transformer(
            inputs_embeds=joint_features,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # try pooler output
        x = outputs.pooler_output

        # try get average of all hidden state
        # x = outputs[0]
        # x = (x * attention_mask.unsqueeze(2)).sum(dim=1) / attention_mask.sum(
        #     dim=1
        # )[:, None]

        x = self.dropout(x)
        x = self.classifer(x)
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
        parser.add_argument("--focal_loss", action='store_true')
        return parser
