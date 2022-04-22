import logging
import json
import itertools
import re
from argparse import ArgumentParser
import sys
from pathlib2 import Path
f_path = Path(__file__)
sys.path.append(str(f_path.parent.parent.absolute()))

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.processor import Processor


class GaiicData(pl.LightningDataModule):
    def __init__(
        self,
        model_name: str,
        local: bool,
        train_file: str,
        dev_file: str,
        attrval_info_file: str,
        attr_to_attrvals_file: str,
        positive_prob: float = 0.5,
        title_prob: float = 0.3,
        batch_size: int = 256,
        num_workers: int = 8,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()
        self.processor = Processor(
            self.hparams.tokenizer_name, local=self.hparams.local)
        # if self.hparams.dataset_version == 1:
        #     MyDataset = JsonDataset
        # else:
        #     MyDataset = JsonDataset2
        if self.hparams.debug:
            self.hparams.train_file = self.hparams.dev_file

    def setup(self, stage: str) -> None:
        if stage == 'fit' or stage is None:
            self.train_dataset = JsonDataset2(
                self.hparams.train_file,
                self.hparams.attrval_info_file,
                self.hparams.attr_to_attrvals_file,
                self.hparams.positive_prob,
                self.hparams.title_prob
            )
            self.dev_dataset = JsonDataset(
                self.hparams.dev_file,
                self.hparams.attrval_info_file,
                self.hparams.attr_to_attrvals_file,
                0.5,
                0.3,
            )

    @staticmethod
    def _collate_fn(data: list, processor):
        assert isinstance(data, list)
        features = []
        texts = []
        labels = []
        for feature, text, label in data:
            features.append(feature.unsqueeze(0))
            texts.append(text)
            labels.append(label)
        features = torch.cat(features, dim=0)
        encodings = processor(features, texts)
        labels = torch.tensor(labels, dtype=torch.long)
        return encodings, labels

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            collate_fn=lambda x: GaiicData._collate_fn(x, self.processor),
        )

    def val_dataloader(self):
        return DataLoader(
            self.dev_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=lambda x: GaiicData._collate_fn(x, self.processor),
        )

    def test_dataloader(self):
        return DataLoader(
            self.dev_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=lambda x: GaiicData._collate_fn(x, self.processor),
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
        # parser.add_argument('--dataset_version', type=int, default=1)
        parser.add_argument('--local', action="store_true")
        parser.add_argument('--train_file', type=str,
                            default='/data/contest/train/train_fine.txt.00,/data/contest/train/train_coarse.txt')
        parser.add_argument('--dev_file', type=str,
                            default='/data/contest/train/train_fine.txt.01')
        parser.add_argument('--attrval_info_file', type=str,
                            default='/data/contest/attrval_info.json')
        parser.add_argument('--attr_to_attrvals_file',
                            type=str, default='/data/contest/train/attr_to_attrvals.json')
        parser.add_argument('--positive_prob', type=float, default=.5)
        parser.add_argument('--title_prob', type=float, default=.15)
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


class JsonDataset(Dataset):
    def __init__(
        self,
        input_filename,
        attrval_info_file,
        attr_to_attrvals_file,
        positive_prob=0.5,
        title_prob=0.3,
    ):
        super().__init__()
        logging.debug(f"Loading json data from {input_filename}.")

        self.attr_dict = {}
        with open(attrval_info_file, "r") as f:
            self.attrval_info = json.load(f)
        with open(attr_to_attrvals_file, "r") as f:
            for attr, attrval_list in json.load(f).items():
                attrval_list = list(map(lambda x: x.split("="), attrval_list))
                self.attr_dict[attr] = list(
                    itertools.chain.from_iterable(attrval_list))

        self.items = []
        self.positive_prob = positive_prob
        self.title_prob = title_prob
        for file in input_filename.split(","):
            with open(file, "r") as f:
                for line in tqdm(f):
                    item = json.loads(line)
                    if item["match"]["图文"]:
                        expand_attr = []
                        if item["key_attr"]:
                            for attr in item["key_attr"]:
                                ret = self._match_attrval(item["title"], attr)
                                if ret is not None:
                                    expand_attr.append(ret)
                        item["expand_attr"] = expand_attr
                        # pop unnecessary items
                        item.pop("key_attr")
                        item.pop("img_name")
                        item.pop("match")
                        self.items.append(item)

        logging.debug("Done loading data.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image = np.array(self.items[idx]["feature"]).astype(np.float32)
        if np.random.uniform() < self.positive_prob:  # get a positive sample
            # if key exists
            if self.items[idx]["expand_attr"] and np.random.uniform() > self.title_prob:
                text, label = self._gen_pos_sample(
                    np.random.choice(self.items[idx]["expand_attr"], 1)[0])
            else:
                text, label = self._gen_pos_sample(
                    self.items[idx]["title"])
        else:  # get a negative sample
            if self.items[idx]["expand_attr"] and np.random.uniform() > self.title_prob:
                text, label = self._gen_neg_sample(
                    np.random.choice(self.items[idx]["expand_attr"], 1)[0])
            else:
                text, label = self._gen_neg_sample(
                    self.items[idx]["title"])
        return torch.from_numpy(image), text, label

    def _gen_pos_sample(self, text: str):
        # 随机替换同义属性产生一个正例
        if not isinstance(text, str):
            text = text[0]
        attrvals = "|".join(self.attrval_info.keys())
        hit_attrvals = re.findall(attrvals, text)
        label = torch.ones(1, dtype=torch.long)
        if hit_attrvals:
            random_attrval = np.random.choice(hit_attrvals, 1)[0]
            match = np.random.choice(
                self.attrval_info[random_attrval]["match_attr"], 1
            )[0]
            text = text.replace(random_attrval, match)
        return text, label

    def _gen_neg_sample(self, text: str):
        # 随机替换不匹配的属性值产生一个负例
        if not isinstance(text, str):
            text = text[0]
        attrvals = "|".join(self.attrval_info.keys())
        hit_attrvals = re.findall(attrvals, text)
        label = torch.zeros(1, dtype=torch.long)
        if hit_attrvals:
            random_attrval = np.random.choice(hit_attrvals, 1)[0]
            mismatch = np.random.choice(
                self.attrval_info[random_attrval]["mismatch_attr"], 1
            )[0]
            text = text.replace(random_attrval, mismatch)
        else:
            text = np.random.choice(self.items, 1)[0]["title"]
        return text, label

    def _match_attrval(self, title, attr):
        # 在title中匹配属性值
        attrvals = "|".join(self.attr_dict[attr])
        ret = re.findall(attrvals, title)
        return "{}{}".format(attr, "".join(ret)) if ret else None


class JsonDataset2(Dataset):
    def __init__(
        self,
        input_filename,
        attrval_info_file,
        attr_to_attrvals_file,
        positive_prob=0.5,
        title_prob=0.3,
    ):
        super().__init__()
        logging.debug(f"Loading json data from {input_filename}.")

        self.attr_dict = {}
        with open(attrval_info_file, "r") as f:
            self.attrval_info = json.load(f)
        with open(attr_to_attrvals_file, "r") as f:
            for attr, attrval_list in json.load(f).items():
                attrval_list = list(map(lambda x: x.split("="), attrval_list))
                self.attr_dict[attr] = list(
                    itertools.chain.from_iterable(attrval_list))

        self.items = []
        self.positive_prob = positive_prob
        self.title_prob = title_prob
        for file in input_filename.split(","):
            with open(file, "r") as f:
                for line in tqdm(f):
                    item = json.loads(line)
                    if item["match"]["图文"]:
                        item["texts"] = [item["title"]]
                        if item["key_attr"]:
                            for attr in item["key_attr"]:
                                ret = self._match_attrval(item["title"], attr)
                                if ret is not None:
                                    item["texts"].append(ret)
                        # pop unnecessary items
                        item.pop("key_attr")
                        item.pop("img_name")
                        item.pop("match")
                        self.items.append(item)

        logging.debug("Done loading data.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image = np.array(self.items[idx]["feature"]).astype(np.float32)
        text = np.random.choice(self.items[idx]["texts"], 1)[0]
        if np.random.uniform() < self.positive_prob:  # get a positive sample
            text, label = self._gen_pos_sample(text)
        else:  # get a negative sample
            text, label = self._gen_neg_sample(text)
        return torch.from_numpy(image), text, label

    def _gen_pos_sample(self, text: str):
        # 随机替换同义属性产生一个正例
        if not isinstance(text, str):
            text = text[0]
        attrvals = "|".join(self.attrval_info.keys())
        hit_attrvals = re.findall(attrvals, text)
        label = torch.ones(1, dtype=torch.long)
        if hit_attrvals:
            random_attrval = np.random.choice(hit_attrvals, 1)[0]
            match = np.random.choice(
                self.attrval_info[random_attrval]["match_attr"], 1
            )[0]
            text = text.replace(random_attrval, match)
        return text, label

    def _gen_neg_sample(self, text: str):
        # 随机替换不匹配的属性值产生一个负例
        if not isinstance(text, str):
            text = text[0]
        attrvals = "|".join(self.attrval_info.keys())
        hit_attrvals = re.findall(attrvals, text)
        label = torch.zeros(1, dtype=torch.long)
        if hit_attrvals and np.random.uniform() > self.title_prob:
            random_attrval = np.random.choice(hit_attrvals, 1)[0]
            mismatch = np.random.choice(
                self.attrval_info[random_attrval]["mismatch_attr"], 1
            )[0]
            text = text.replace(random_attrval, mismatch)
        else:
            text = np.random.choice(self.items, 1)[0]["title"]
        return text, label

    def _match_attrval(self, title, attr):
        # 在title中匹配属性值
        attrvals = "|".join(self.attr_dict[attr])
        ret = re.findall(attrvals, title)
        return "{}{}".format(attr, "".join(ret)) if ret else None


if __name__ == '__main__':
    dev_dataset = JsonDataset2(
        '../train/train_fine.txt.01',
        '../attrval_info.json',
        '../train/attr_to_attrvals.json',
        0.5,
        0.15,
    )
    data = DataLoader(dev_dataset, shuffle=True, batch_size=1)
    data_iter = iter(data)
    while True:
        item = next(data_iter)
