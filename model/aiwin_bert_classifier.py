from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch import logit, nn
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


class AiwinBertClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name,
        train_data_size,
        lr=1e-4,
        dropout=0.2,
        focal_loss=True,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        bert_config = BertConfig.from_pretrained(
            model_name, cache_dir=Path.home() / '.cache')
        hidden_size = bert_config.hidden_size
        self.transformer = BertModel.from_pretrained(
            model_name, cache_dir=Path.home() / '.cache')
        # self.text_embed = nn.Embedding(vocab_size, hidden_size, padding_idx)
        self.text_embed = self.transformer.get_input_embeddings()
        # self.img_proj = nn.Linear(
        #     2048, self.hparams.image_token_size * hidden_size)
        self.num_proj = nn.Sequential(
            nn.Linear(3708, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, hidden_size),
        )
        self.classifier = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(dropout)

        self.criterion = FocalLoss(
            alpha=[1, 1], num_classes=2) if self.hparams.focal_loss else nn.CrossEntropyLoss()

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.train_auc = torchmetrics.AUC(reorder=True)
        self.val_auc = torchmetrics.AUC(reorder=True)

    def forward(self, num_features, text_encodings):
        # [N, 1, 3708]
        num_features = num_features.unsqueeze(1)
        B = num_features.shape[0]
        L = num_features.shape[1]
        # print(num_features.shape)
        num_features = self.num_proj(num_features)  # [N, n * hidden_size]
        text_features = self.text_embed(text_encodings.input_ids)  # [N, L, hidden_size]
        num_attention_mask = torch.ones((B, L), dtype=torch.float).to(self.device)
        num_token_type_ids = torch.ones((B, L), dtype=torch.long).to(self.device)
        
        text_attention_mask = text_encodings.attention_mask
        text_token_type_ids = text_encodings.token_type_ids
        attention_mask = torch.cat([num_attention_mask, text_attention_mask], dim=-1)
        token_type_ids = torch.cat([text_token_type_ids, num_token_type_ids], dim=-1)
        joint_features = torch.cat(
            [text_features, num_features], dim=1
        )
        outputs = self.transformer(
            inputs_embeds=joint_features,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # try pooler output
        x = outputs.pooler_output

        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch):
        num_features = batch['nums']
        text_encodings = batch['text']
        targets = batch['label']
        logits = self(num_features, text_encodings)
        loss = self.criterion(logits, targets)
        
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, targets)
        self.train_auc(preds, targets)
        
        self.log_dict(
            {"train_loss": loss,
            "train_acc": self.train_accuracy, 
            "train_auc": self.train_auc},
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
        num_features = batch['nums']
        text_encodings = batch['text']
        targets = batch['label']
        logits = self(num_features, text_encodings)
        loss = self.criterion(logits, targets)
        
        preds = torch.argmax(logits, dim=1)
        
        self.val_accuracy(preds, targets)
        self.val_auc(preds, targets)
        self.log_dict(
            {"val_loss": loss, "val_acc": self.val_accuracy, "val_auc": self.val_auc},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def validation_epoch_end(self, outputs) -> None:
        if self.hparams.use_nni:
            nni.report_intermediate_result(self.val_auc.compute().item())
        print("")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--model_name", type=str,
                            default="julien-c/bert-xsmall-dummy")
        parser.add_argument("--dropout", type=float, default=0.2)
        parser.add_argument("--focal_loss", action='store_true')
        parser.add_argument("--ft_lr", type=float, default=2e-5)
        parser.add_argument("--scheduler", type=str, default='cosine')
        return parser
