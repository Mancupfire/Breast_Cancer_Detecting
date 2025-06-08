import logging
import os
from pathlib import Path
from typing import Optional, Dict
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from .support_util import load_tokenizer

from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

log = logging.getLogger(__name__)

def load_dataset(
    df: pd.DataFrame,
    split: str,
    dataset: str,
    data_dir: str,
    image_dir: str,
    data_type: str,
    tokenizer=None,
    transform_config: dict = None,
    loss_config=None,
    text_max_length: int = None,
    mean=None,
    std=None,
    image_encoder_type: str = None,
    label_col: str = None,
    label_text: str = None,
) -> Dataset:
    data_type = data_type.lower()
    if data_type == "image":
        tfm = get_transforms(transform_config, mean, std, train=(split == "train"))
        class A: pass
        args = A()
        args.data_dir = os.path.join(data_dir, image_dir)
        return MammoDataset_Mapper(args, df, transform=tfm)

    elif data_type == "text":
        class TextDataset(Dataset):
            def __init__(self, texts, tokenizer, max_len):
                self.texts = texts.reset_index(drop=True)
                self.tokenizer = tokenizer
                self.max_len = max_len

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                txt = str(self.texts.iloc[idx])
                out = self.tokenizer(
                    txt,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors="pt",
                )
                return {k: v.squeeze(0) for k, v in out.items()}

        col = label_text if label_text and label_text in df.columns else "text1"
        return TextDataset(df[col], tokenizer, text_max_length)

    else:
        raise ValueError(f"Unsupported data_type '{data_type}'")
    

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
            dataloader_config: Dict = None,
            tokenizer_config: Dict = None,
            loss_config: Dict = None,
            transform_config: Dict = None,
            cur_fold: int = 0,
            mean: float = 0,
            std: float = 0,
            image_encoder_type: str = "swin"
    ):
        self.train_sampler = None
        dtype_options = {
            "patient_id": str,
            "image_id": str,
            "laterality": str,
            "view": str,
            "text1": str,
            "text_aug": str,
            "fold": int
        }

        self.data_config = data_config
        self.dataloader_config = dataloader_config
        self.tokenizer_config = tokenizer_config
        self.loss_config = loss_config
        self.tokenizer = load_tokenizer(**self.tokenizer_config) if self.tokenizer_config is not None else None
        self.datasets = {"train": [], "valid": [], "test": []}
        self.image_encoder_type = image_encoder_type

        self.train_loader = None
        self.valid_loader_dict = None
        self.test_loader = None

        for dataset in data_config:
            df = pd.read_csv(Path(
                data_config[dataset]["data_dir"]) / data_config[dataset]["data_path"], dtype=dtype_options)
            df = df.fillna(0)
            if data_config[dataset]["name"].lower() == "vindr":
                train_df = df[df['split'] == "training"].reset_index(drop=True)
                valid_df = df[df['split'] == "test"].reset_index(drop=True)
            else:
                train_df = df[df['fold'] != cur_fold].reset_index(drop=True)
                valid_df = df[df['fold'] == cur_fold].reset_index(drop=True)

            train_dataset = load_dataset(
                df=train_df,
                split="train",
                dataset=data_config[dataset]["name"],
                data_dir=data_config[dataset]["data_dir"],
                image_dir=data_config[dataset]["img_dir"],
                data_type=data_config[dataset]["data_type"],
                tokenizer=self.tokenizer,
                transform_config=transform_config,
                loss_config=self.loss_config,
                text_max_length=data_config[dataset]["text_max_length"],
                mean=mean,
                std=std,
                image_encoder_type=self.image_encoder_type,
                label_col=data_config[dataset]["label_col"] if "label_col" in data_config[dataset] else None,
                label_text=data_config[dataset]["label_text"] if "label_text" in data_config[dataset] else None,
            )
            valid_dataset = load_dataset(
                df=valid_df,
                split="valid",
                dataset=data_config[dataset]["name"],
                data_dir=data_config[dataset]["data_dir"],
                image_dir=data_config[dataset]["img_dir"],
                data_type=data_config[dataset]["data_type"],
                tokenizer=self.tokenizer,
                transform_config=transform_config,
                loss_config=self.loss_config,
                text_max_length=data_config[dataset]["text_max_length"],
                mean=mean,
                std=std,
                image_encoder_type=self.image_encoder_type,
                label_col=data_config[dataset]["label_col"] if "label_col" in data_config[dataset] else None,
                label_text=data_config[dataset]["label_text"] if "label_text" in data_config[dataset] else None,
            )

            self.datasets["train"].append(train_dataset)
            self.datasets["valid"].append(valid_dataset)

            log.info(f"Loading fold: {cur_fold}")
            log.info(f"train_df length: {train_df.shape}")
            log.info(f"valid_df length: {valid_df.shape}")
            log.info(f"Length of train_dataset: {len(train_dataset)}")
            log.info(f"Length of valid_dataset: {len(valid_dataset)}")

            log.info(f"Dataset: {dataset} is loaded")

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
