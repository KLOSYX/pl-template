from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch import nn
from transformers import (
    BertModel,
    BertForSequenceClassification,
    BertConfig,
    BertTokenizer,
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
import torchmetrics
import nni

from utils.focal_loss import FocalLoss
import utils.r_drop


class GaiicBertClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name,
        vocab_size,
        padding_idx,
        train_data_size,
        lr=1e-4,
        dropout=0.2,
        image_token_size=1,
        focal_loss=False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        bert_config = BertConfig.from_pretrained(
            model_name, cache_dir="/data/contest/pytorch-lightning/~/.cache")
        hidden_size = bert_config.hidden_size
        self.transformer = BertModel.from_pretrained(
            model_name, cache_dir="/data/contest/pytorch-lightning/~/.cache")
        # self.text_embed = nn.Embedding(vocab_size, hidden_size, padding_idx)
        self.text_embed = self.transformer.get_input_embeddings()
        # self.img_proj = nn.Linear(
        #     2048, self.hparams.image_token_size * hidden_size)
        self.img_proj = nn.Sequential(
            nn.Linear(2048, 2048 * 4),
            # nn.BatchNorm1d(4 * 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048 * 4, hidden_size),
        )
        self.classifier = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(dropout)

        self.criterion = FocalLoss(
            alpha=[1, 1], num_classes=2) if self.hparams.focal_loss else nn.CrossEntropyLoss()

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

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
        x = self.classifier(x)
        return x

    def training_step(self, batch):
        encodings, targets = batch
        if not self.hparams.r_drop:
            logits = self(*encodings)
            loss = self.criterion(logits, targets)

        else:
            # encodings = [torch.cat([x, x.clone().detach()], dim=0) for x in encodings]
            encodings = [x.repeat(2, 1) for x in encodings]
            logits = self(*encodings)  # [2 * N, 2]
            logits1, logits2 = torch.chunk(logits, 2, dim=0)  # 2 * [N, 2]
            loss1 = self.criterion(logits1, targets)
            loss2 = self.criterion(logits2, targets)
            ce_loss = 0.5 * loss1 + 0.5 * loss2
            kl_loss = utils.r_drop.compute_kl_loss(logits1, logits2)
            loss = ce_loss + self.hparams.r_drop_alpha * kl_loss
            self.log_dict(
                {'ce_loss': ce_loss, 'kl_loss': kl_loss}, on_step=True)

        if self.hparams.r_drop:
            logits = (logits1 + logits2) / 2
        preds = torch.argmax(logits, dim=1)

        return {'loss': loss, 'preds': preds, 'targets': targets}

    def training_step_end(self, step_output):
        self.train_accuracy(step_output['preds'], step_output['targets'])
        losses = step_output['loss']
        loss = (losses[0] + losses[1]) / 2
        self.log_dict(
            {"train_loss": loss,
            "train_acc": self.train_accuracy, },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def training_epoch_end(self, outputs) -> None:
        print("")

    def configure_optimizers(self):
        step_per_epoch = self.hparams.train_data_size // self.hparams.batch_size
        max_step = self.hparams.max_epochs * step_per_epoch

        trm_params = list(map(id, self.transformer.parameters()))
        base_params = filter(lambda p: id(p) not in trm_params,
                             self.parameters())
        optimizer = torch.optim.AdamW([{'params': base_params}, {'params': self.transformer.parameters(),
                                                                 'lr': self.hparams.ft_lr}],
                                      lr=self.hparams.lr,
                                      eps=1e-6,
                                      weight_decay=0.02)
        if self.hparams.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                0.1 * max_step,
                max_step,
            )
        elif self.hparams.scheduler == 'polynomial':
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                0.1 * max_step,
                max_step,
            )
        else:
            raise ValueError('Wrong scheduler name')
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def validation_step(self, batch, batch_idx):
        encodings, targets = batch

        outputs = self(*encodings)
        loss = self.criterion(outputs, targets)
        preds = torch.argmax(outputs, dim=1)

        return {'loss': loss, 'preds': preds, 'targets': targets}

    def validation_step_end(self, outputs):
        self.val_accuracy(outputs['preds'], outputs['targets'])

        losses = outputs['loss']
        loss = (losses[0] + losses[1]) / 2
        self.log_dict(
            {"val_loss": loss, "val_acc": self.val_accuracy},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def validation_epoch_end(self, outputs) -> None:
        if self.hparams.use_nni:
            nni.report_intermediate_result(self.val_accuracy.compute().item())
        print("")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--model_name", type=str,
                            default="bert-base-chinese")
        parser.add_argument("--dropout", type=float, default=0.2)
        parser.add_argument("--image_token_size", type=int, default=1)
        parser.add_argument("--focal_loss", action='store_true')
        parser.add_argument("--ft_lr", type=float, default=2e-5)
        parser.add_argument("--r_drop", action='store_true')
        parser.add_argument("--r_drop_alpha", type=float, default=5.)
        parser.add_argument("--scheduler", type=str, default='cosine')
        return parser
