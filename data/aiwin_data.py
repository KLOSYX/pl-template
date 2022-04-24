from typing import Optional, List, Dict
from argparse import ArgumentParser
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.absolute()))

from transformers import BertTokenizer
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split


class CollectFn(object):
    def __init__(self, tokenizer: 'BertTokenizer'):
        super().__init__()
        self.tokenizer = tokenizer

    def __call__(self, data: List[Dict]):
        ret = {k: [] for k in data[0].keys()}
        for line in data:
            for k, v in line.items():
                if k == 'ent_id' or k == 'text':
                    ret[k].append(v)
                elif k == 'nums':
                    ret[k].append(torch.tensor(v, dtype=torch.float).unsqueeze(0))
                else:
                    ret[k].append(torch.tensor(v, dtype=torch.long).unsqueeze(0))
        ret['text'] = {k: v for k, v in self.tokenizer(ret['text'],
                                                truncation=True,
                                                max_length=256,
                                                padding="max_length",
                                                return_tensors="pt").items()}
        ret['nums'] = torch.cat(ret['nums'], dim=0)
        if ret.get('label') is not None:
            ret['label'] = torch.cat(ret['label'], dim=0)
        return ret.get('nums'), ret.get('text'), ret.get('label', None)

class AiwinDataset(Dataset):
    def __init__(self, data_path: str, stage: str = None) -> None:
        super().__init__()
        data_path = Path(data_path)
        data = pd.read_feather(data_path)
        if stage is None or stage == 'fit':
            self.data = data[data.iloc[:, -1].notnull()].values
        else:
            self.data = data[data.iloc[:, -1].isnull()].values
        self.stage = stage

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        line = self.data[index]
        if self.stage == None or self.stage == 'fit':
            return {'ent_id': line[0], 'nums': np.array(line[1:-2], dtype=np.float), 'text': line[-2], 'label': np.array(line[-1], dtype=np.int)}
        else:
            return {'ent_id': line[0], 'nums': np.array(line[1:-2], dtype=np.float), 'text': line[-2]}


class AiwinData(pl.LightningDataModule):
    def __init__(self,
                 data_path='data/raw/aiwin/df_data.feather',
                 tokenizer_name='bert-base-chinese',
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        if hasattr(self, 'train_dataset') or hasattr(self, 'test_dataset'):
            return 
        self.tokenizer = BertTokenizer.from_pretrained(
            self.hparams.tokenizer_name, cache=Path.home() / '.cache')
        if stage is None or stage == 'fit':
            dataset = AiwinDataset(self.hparams.data_path, stage)
            train_size = int(len(dataset) * 0.9)
            val_size = len(dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(
                dataset, [train_size, val_size])

        else:
            self.test_dataset = AiwinDataset(self.hparams.data_path, stage)

    @staticmethod
    def _collect_fn(data: List[Dict], tokenizer: 'BertTokenizer'):
        ret = {k: [] for k in data[0].keys()}
        for line in data:
            for k, v in line.items():
                if k == 'ent_id' or k == 'text':
                    ret[k].append(v)
                elif k == 'nums':
                    ret[k].append(torch.tensor(v, dtype=torch.float).unsqueeze(0))
                else:
                    ret[k].append(torch.tensor(v, dtype=torch.long).unsqueeze(0))
        ret['text'] = tokenizer(ret['text'],
                                truncation=True,
                                max_length=256,
                                padding="max_length",
                                return_tensors="pt",)
        ret['nums'] = torch.cat(ret['nums'], dim=0)
        if ret.get('label') is not None:
            ret['label'] = torch.cat(ret['label'], dim=0)
        return ret

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          shuffle=True,
                          collate_fn=CollectFn(self.tokenizer),)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          collate_fn=CollectFn(self.tokenizer),)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          collate_fn=CollectFn(self.tokenizer),)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--tokenizer_name", type=str,
                           default='prajjwal1/bert-small')
        parser.add_argument("--data_path", type=str, default="data/raw/aiwin/df_data.feather")
        return parser
    
    def add_data_info(self):
        self.setup()
        return {'train_data_size': len(self.train_dataset)}


if __name__ == '__main__':
    dm = AiwinData(batch_size=8, num_workers=0)
    dm.setup('fit')
    train_data = dm.train_dataloader()
    it = iter(train_data)
    while True:
        item = next(it)