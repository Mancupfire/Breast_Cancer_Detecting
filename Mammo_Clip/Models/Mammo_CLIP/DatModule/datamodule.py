import os
from pathlib import Path
from typing import Optional, Dict

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

def get_Paths(args):
    from pathlib import Path
    base = Path(args.root)
    checkpoint_path = base / "checkpoints"
    output_path     = base / "outputs"
    tb_logs_path    = base / "logs"

    return checkpoint_path, output_path, tb_logs_path

class DataModule:
    def __init__(
        self,
        data_config: Dict,
        dataloader_config: Dict,
        tokenizer_config: Optional[Dict] = None,
        loss_config: Optional[Dict] = None,
        transform_config: Optional[Dict] = None,
        mean: Optional[list] = None,
        std: Optional[list] = None,
        image_encoder_type: Optional[str] = None,
        cur_fold: Optional[int] = None,
    ):
        # Tokenizer
        self.tokenizer = None
        if tokenizer_config and "pretrained_model_name_or_path" in tokenizer_config:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_config["pretrained_model_name_or_path"]
            )

        # Transforms
        self.train_transform = None
        self.val_transform = None
        if transform_config:
            self.train_transform = get_transforms(transform_config, mean, std, train=True)
            self.val_transform = get_transforms(transform_config, mean, std, train=False)

    def setup(self):
        # Placeholder for dataset setup if needed
        pass

    def train_dataloader(self, distributed):
        assert self.dataloader_config is not None

        if self.train_loader is None:
            dataset = ConcatDataset(self.datasets["train"])
            shuffle = self.dataloader_config["train"]["shuffle"]

            # DDP
            if distributed:
                self.dataloader_config["train"]["shuffle"] = False  # Disable shuffle if using DistributedSampler

                # Create DistributedSampler
                self.train_sampler = DistributedSampler(dataset=dataset, num_replicas=util.GlobalEnv.get().world_size,
                                                        rank=util.GlobalEnv.get().world_rank)

            else:
                self.train_sampler = None

            self.train_loader = DataLoader(
                dataset,
                collate_fn=getattr(self.datasets["train"][0], "collate_fn", None),
                sampler=self.train_sampler,
                **self.dataloader_config["train"],
            )

        return self.train_loader, self.train_sampler

    def valid_dataloader(self, distributed=False):
        assert self.dataloader_config is not None

        if self.valid_loader_dict is None:
            self.valid_loader_dict = dict()

            for val_dataset in self.datasets["valid"]:
                # DDP
                if distributed:
                    sampler = DistributedSampler(dataset=val_dataset, num_replicas=util.GlobalEnv.get().world_size,
                                                 rank=util.GlobalEnv.get().world_rank)
                else:
                    sampler = None
                    # sampler.set_epoch(0)   # This ensures shuffling will be the same across epochs.

                dataloader = DataLoader(
                    val_dataset, collate_fn=getattr(val_dataset, "collate_fn", None), sampler=sampler,
                    **self.dataloader_config["valid"]
                )
                self.valid_loader_dict[val_dataset.dataset] = dataloader

        return self.valid_loader_dict


def get_transforms(config_or_args, mean: Optional[list] = None, std: Optional[list] = None, train: bool = True):
    # Determine image size and normalization parameters
    if isinstance(config_or_args, dict):
        cfg = config_or_args
        size = cfg.get("resize", 224)
        mean = cfg.get("mean", mean if mean is not None else [0.485, 0.456, 0.406])
        std = cfg.get("std", std if std is not None else [0.229, 0.224, 0.225])
    else:
        # assume args namespace
        args = config_or_args
        size = getattr(args, "img_size", 224)
        mean = getattr(args, "mean", mean if mean is not None else [0.485, 0.456, 0.406])
        std = getattr(args, "std", std if std is not None else [0.229, 0.224, 0.225])

    tfms = [transforms.Resize((size, size))]
    if train:
        tfms += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ]
    tfms += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(tfms)


class MammoDataset_Mapper(Dataset):
    def __init__(self, args, df, transform=None):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.data_dir = Path(args.data_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Determine image path
        if "image_path" in row:
            img_path = self.data_dir / row["image_path"]
        elif "image_id" in row:
            img_path = self.data_dir / f"{row['image_id']}.png"
        else:
            raise KeyError("DataFrame must have 'image_path' or 'image_id' column")

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # Labels for attributes: Mass and Suspicious_Calcification
        mass = row.get("Mass", row.get("mass", 0))
        calc = row.get(
            "Suspicious_Calcification", row.get("suspicious_calcification", 0)
        )
        labels = torch.tensor([mass, calc], dtype=torch.int64)

        return {"img": img, "labels": labels}
