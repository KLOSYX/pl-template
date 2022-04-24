from argparse import ArgumentParser
from pathlib import Path
import json
import sys
sys.path.append(str(Path(__file__).parent.absolute()))
from typing import Optional

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import torch


class VideoHw1Datset(Dataset):
    def __init__(self, data_path: str, transformer: transforms = None) -> None:
        super().__init__()
        data_path = Path(data_path)
        if not data_path.is_dir():
            raise NotADirectoryError(f"{str(data_path)}: Path Error!")
        self.file_list = list(data_path.glob('*'))
        self.transformer = transformer if transformer is not None else transforms.Compose(
            [
                transforms.Resize((256, 256)),  # 缩放图片(Image)，保持长宽比不变，最短边为224像素
                transforms.RandomCrop(224),  # 从图片中间切出224*224的图片
                # 将图片(Image)转成Tensor，归一化至[0, 1]（直接除以255）
                transforms.ToTensor(),
                # 标准化至[-1, 1]，规定均值和标准差
                transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
            ]
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        label_path = file_path / (file_path.name + '.json')
        image_path = file_path / 'BireView.png'

        image = Image.open(image_path)
        image = self.transformer(image)

        with open(label_path, 'r', encoding='utf-8') as f:
            label = [x['type'] for x in json.load(f)['themes']]
        label = torch.tensor(label, dtype=torch.long)

        return image, label


class VideoHw1Data(pl.LightningDataModule):
    def __init__(self, data_path: str, 
                 batch_size: int = 8,
                 num_workers: int = 4,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.data_path = data_path

    def setup(self, stage: Optional[str] = None) -> None:
        data = VideoHw1Datset(self.data_path)
        data_size= len(data)
        train_size = int(0.8 * data_size)
        val_size = data_size - train_size
        self.train_data, self.val_data = random_split(data, [train_size, val_size])
        
    @staticmethod
    def _collect_fn(data: list):
        assert isinstance(data, list)
        batch_size = len(data)
        labels = torch.zeros((batch_size, 8), dtype=torch.long)
        images = []
        for i, (image, label) in enumerate(data):
            labels[i, label] = 1
            images.append(image.unsqueeze(0))
        images = torch.cat(images, dim=0)
        return images, labels
        

    def train_dataloader(self):
        return DataLoader(self.train_data, collate_fn=self._collect_fn, batch_size=self.hparams.batch_size, 
                          num_workers=self.hparams.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, collate_fn=self._collect_fn, batch_size=self.hparams.batch_size, 
                          num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.val_data, collate_fn=self._collect_fn, batch_size=self.hparams.batch_size, 
                          num_workers=self.hparams.num_workers)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data_path', type=str, default='data/raw/hw1_data')
        return parser


if __name__ == '__main__':
    dataset = VideoHw1Datset('data/raw/hw1_data')
    dm = VideoHw1Data('data/raw/hw1_data')
    dm.setup()
    train_data = dm.train_dataloader()
    it = iter(train_data)
    while True:
        item = next(it)