from argparse import ArgumentParser
import sys
from pathlib2 import Path
f_path = Path(__file__)
sys.path.append(str(f_path.parent.parent.absolute()))

from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl

from utils.processor import Processor
from data.gaiic_data import JsonDataset2


class JsonDatasetCL(JsonDataset2):
    def __init__(self, input_filename, attrval_info_file, attr_to_attrvals_file, positive_prob=0.5, title_prob=0.3):
        super().__init__(input_filename, attrval_info_file,
                         attr_to_attrvals_file, positive_prob, title_prob)

    def __getitem__(self, idx):
        image = np.array(self.items[idx]["feature"]).astype(np.float32)
        text = np.random.choice(self.items[idx]['texts'], 1)[0]
        text_pos, _ = self._gen_pos_sample(text)
        text_neg, _ = self._gen_neg_sample(text)
        return torch.from_numpy(image), text_pos, text_pos, text_neg


class GaiicDataCl(pl.LightningDataModule):
    def __init__(
        self,
        model_name: str,
        train_file: str,
        dev_file: str,
        attrval_info_file: str,
        attr_to_attrvals_file: str,
        title_prob: float = 0.3,
        batch_size: int = 256,
        num_workers: int = 8,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()
        """@nni.variable(nni.choice(32, 64, 128), name=self.hparams.batch_size)"""
        self.processor = Processor(
            self.hparams.tokenizer_name, False)
        if self.hparams.debug:
            self.hparams.train_file = self.hparams.dev_file
        self.train_dataset = JsonDatasetCL(
            self.hparams.train_file,
            self.hparams.attrval_info_file,
            self.hparams.attr_to_attrvals_file,
            self.hparams.title_prob,
        )
        self.dev_dataset = JsonDatasetCL(
            self.hparams.dev_file,
            self.hparams.attrval_info_file,
            self.hparams.attr_to_attrvals_file,
        )

    def setup(self, stage: str) -> None:
        pass

    @staticmethod
    def _collate_fn(data: list, processor):
        assert isinstance(data, list)
        features = []
        texts_1 = []
        texts_2 = []
        texts_neg = []
        for feature, text_1, text_2, text_neg in data:
            features.append(feature.unsqueeze(0))
            texts_1.append(text_1)
            texts_2.append(text_2)
            texts_neg.append(text_neg)
        N = len(features)
        features = torch.cat(features, dim=0)
        features = features.expand(3 * N, features.shape[-1])
        encodings = processor(features, texts_1 + texts_2 + texts_neg)
        return encodings

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            collate_fn=lambda x: GaiicDataCl._collate_fn(x, self.processor),
        )

    def val_dataloader(self):
        return DataLoader(
            self.dev_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=lambda x: GaiicDataCl._collate_fn(x, self.processor),
        )

    def test_dataloader(self):
        return DataLoader(
            self.dev_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=lambda x: GaiicDataCl._collate_fn(x, self.processor),
        )

    def get_train_size(self):
        return len(self.train_dataset)

    def get_vocab_size(self):
        return self.processor.get_vocab_size()

    def get_padding_idx(self):
        return self.processor.get_padding_idx()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--train_file', type=str,
                            default='/data/contest/train/train_fine.txt.00,/data/contest/train/train_coarse.txt')
        parser.add_argument('--dev_file', type=str,
                            default='/data/contest/train/train_fine.txt.01')
        parser.add_argument('--attrval_info_file', type=str,
                            default='/data/contest/attrval_info.json')
        parser.add_argument('--attr_to_attrvals_file',
                            type=str, default='/data/contest/train/attr_to_attrvals.json')
        parser.add_argument('--title_prob', type=float, default=.5)
        parser.add_argument('--tokenizer_name', type=str,
                            default='bert-base-uncased')
        return parser

    def add_data_info(self):
        params = {
            "vocab_size": self.get_vocab_size(),
            "padding_idx": self.get_padding_idx(),
            "train_data_size": self.get_train_size(),
        }
        return params


if __name__ == '__main__':
    data_module = JsonDatasetCL(
        '../train/train_fine.txt.01',
        '../attrval_info.json',
        '../train/attr_to_attrvals.json',
    )
    processor = Processor('bert-base-chinese', False)
    data = DataLoader(data_module, 1, True, collate_fn=lambda x : GaiicDataCl._collate_fn(x, processor))
    it = iter(data)
    while True:
        item = next(it)
    