from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch import nn
from transformers import (
    VisualBertModel,
    VisualBertConfig,
    BertTokenizer,
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
import torchmetrics
import nni
from scipy.stats import spearmanr

from utils.get_simcse_sup_loss import get_simcse_sup_loss


class GaiicVisualbertClassifierCl(pl.LightningModule):
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
        bert_config = VisualBertConfig.from_pretrained(
            model_name, cache_dir="~/.cache")
        hidden_size = bert_config.hidden_size
        visual_embedding_dim = bert_config.visual_embedding_dim
        self.transformer = VisualBertModel.from_pretrained(
            model_name, cache_dir="~/.cache") if self.hparams.use_pre_trained else VisualBertModel(bert_config)
        # self.text_embed = nn.Embedding(vocab_size, hidden_size, padding_idx)
        # self.text_embed = self.transformer.get_input_embeddings()
        # self.img_proj = nn.Linear(
        #     2048, self.hparams.image_token_size * hidden_size)
        self.img_proj = nn.Sequential(
            nn.Linear(2048, 2048 * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048 * 4, visual_embedding_dim),
        )
        self.classifer = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, tokens, img_features, attention_mask, token_type_ids):
        # img_features = img_features.unsqueeze(1)  # [N, 1, 2048]
        B = img_features.shape[0]
        visual_embeds = self.img_proj(img_features)  # [N, n * hidden_size]
        visual_embeds = visual_embeds.reshape(
            B, self.hparams.image_token_size, -1)  # [N, n, hidden_size]
        # [N, L-n, hidden_size]
        visual_token_type_ids = torch.ones(
            visual_embeds.shape[:-1], dtype=torch.long).to(self.device)
        visual_attention_mask = torch.ones(
            visual_embeds.shape[:-1], dtype=torch.float).to(self.device)
        outputs = self.transformer(
            input_ids=tokens,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
            visual_token_type_ids=visual_token_type_ids,
        )
        # try pooler output
        hiddens = outputs.pooler_output

        # try get average of all hidden state
        # x = outputs[0]
        # x = (x * attention_mask.unsqueeze(2)).sum(dim=1) / attention_mask.sum(
        #     dim=1
        # )[:, None]

        # x = self.dropout(hiddens)
        # x = self.classifer(x)
        return hiddens

    def training_step(self, batch):
        encodings = batch
        hiddens = self(*encodings)
        loss = get_simcse_sup_loss(hiddens)

        return {'loss': loss}

    def training_step_end(self, step_output):
        losses = step_output['loss']
        loss = (losses[0] + losses[1]) / 2
        self.log_dict(
            {"train_loss": loss},
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
        encodings = batch

        hiddens = self(*encodings)
        loss = get_simcse_sup_loss(hiddens)

        return {'loss': loss}

    def validation_step_end(self, outputs):

        losses = outputs['loss']
        loss = (losses[0] + losses[1]) / 2
        self.log_dict(
            {"val_loss": loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def validation_epoch_end(self, outputs) -> None:
        print("")
        if self.hparams.use_nni:
            nni.report_intermediate_result(self.val_accuracy.compute().item())

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
