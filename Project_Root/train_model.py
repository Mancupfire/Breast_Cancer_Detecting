import os
import random
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path

from Mammo_CLIP.Project_Root.clip_model import CLIPCustom
from DatModule.datamodule import DataModule, get_transforms, MammoDataset_Mapper
from loss_functions import BreastClipLoss
from transformers import get_cosine_schedule_with_warmup


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    seed = cfg.base.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running on device: {device}")
    print(OmegaConf.to_yaml(cfg))

    # Model
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model = CLIPCustom(model_cfg).to(device)

    # Resume if requested
    if cfg.base.resume_training and cfg.base.checkpoint_to_start:
        ckpt_path = to_absolute_path(cfg.base.checkpoint_to_start)
        last_epoch = model.load(ckpt_path, optimizer=None, scheduler=None)
        print(f"[INFO] Resumed model from {ckpt_path}, last_epoch = {last_epoch}")

    #DataModule & Datasets
    dm = DataModule(
        data_config       = OmegaConf.to_container(cfg.data_train, resolve=True),
        dataloader_config = OmegaConf.to_container(cfg.dataloader, resolve=True),
        tokenizer_config  = OmegaConf.to_container(cfg.tokenizer, resolve=True),
        loss_config       = OmegaConf.to_container(cfg.loss, resolve=True),
        transform_config  = OmegaConf.to_container(cfg.transform, resolve=True),
        mean              = cfg.base.mean,
        std               = cfg.base.std,
        image_encoder_type= cfg.model.image_encoder.model_type,
        cur_fold          = cfg.base.fold,
    )

    # Read metadata CSV and filter
    df = dm.read_metadata_csv()
    df = df[(df["Mass"] == 1) | (df["Suspicious_Calcification"] == 1)]
    train_df = df[df["split"] == "training"].reset_index(drop=True)
    valid_df = df[df["split"] == "test"].reset_index(drop=True)

    # Build Dataset objects
    train_ds = MammoDataset_Mapper(dm, train_df, transform=get_transforms(dm, split="train"))
    valid_ds = MammoDataset_Mapper(dm, valid_df, transform=get_transforms(dm, split="valid"))

    # DataLoaders
    if cfg.base.balanced_dataloader:
        # Load or compute sampler weights
        weights_path = os.path.join(cfg.base.output.checkpoint, f"weights_fold{cfg.base.fold}.pkl")
        if os.path.exists(weights_path):
            with open(weights_path, "rb") as f:
                weights = pickle.load(f)
        else:
            # cfg.base.sampler_weights should be path to JSON or dict in config
            sw = cfg.base.sampler_weights
            # if it's a path
            if isinstance(sw, str) and os.path.exists(to_absolute_path(sw)):
                import json
                with open(to_absolute_path(sw)) as f:
                    wdict = json.load(f)
                pos_w = wdict[f"fold{cfg.base.fold}"]["pos_wt"]
                neg_w = wdict[f"fold{cfg.base.fold}"]["neg_wt"]
            else:
                pos_w = cfg.base.sampler_weights[f"fold{cfg.base.fold}"]["pos_wt"]
                neg_w = cfg.base.sampler_weights[f"fold{cfg.base.fold}"]["neg_wt"]

            train_df["w"] = train_df["cancer"].map({1: pos_w, 0: neg_w})
            weights = train_df["w"].values

            # Validate weights
            arr = np.array(weights, dtype=np.float32)
            assert arr.shape[0] == len(train_df), "Sampler weights length mismatch!"
            assert np.all(arr >= 0), "Negative weights found!"
            total = arr.sum()
            assert total > 0, "Sum of sampler weights must be positive!"
            zeros = int((arr == 0).sum())
            if zeros > 0:
                print(f"⚠️ Warning: {zeros}/{len(arr)} samples have zero weight.")

            # Save for next time
            with open(weights_path, "wb") as f:
                pickle.dump(weights, f)

        sampler = WeightedRandomSampler(weights.tolist(), len(weights), replacement=True)
        train_loader = DataLoader(
            train_ds,
            batch_size   = cfg.base.batch_size,
            sampler      = sampler,
            num_workers  = cfg.base.num_workers,
            pin_memory   = True,
            drop_last    = True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size   = cfg.base.batch_size,
            shuffle      = True,
            num_workers  = cfg.base.num_workers,
            pin_memory   = True,
            drop_last    = True,
        )

    valid_loader = DataLoader(
        valid_ds,
        batch_size   = cfg.base.batch_size,
        shuffle      = False,
        num_workers  = cfg.base.num_workers,
        pin_memory   = True,
        drop_last    = False,
    )

    print(f"[INFO] Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")

    # TensorBoard Writer
    writer = SummaryWriter(log_dir=to_absolute_path(cfg.base.output.tensorboard))

    # Optimizer
    if cfg.optimizer.name.lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr           = cfg.optimizer.config.lr,
            weight_decay = cfg.optimizer.config.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer.name}")

    # Scheduler 
    if cfg.scheduler.name.lower() == "cosine":
        total_steps  = len(train_loader) * cfg.base.epochs
        warmup_steps = len(train_loader) * cfg.scheduler.config.warmup_epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps   = warmup_steps,
            num_training_steps = total_steps,
            num_cycles         = 0.5
        )
    else:
        scheduler = None

    bc_cfg = OmegaConf.to_container(cfg.loss.breast_clip, resolve=True)
    criterion = BreastClipLoss(
        label_smoothing = bc_cfg["label_smoothing"],
        i2i_weight      = bc_cfg["i2i_weight"],
        t2t_weight      = bc_cfg["t2t_weight"],
        loss_ratio      = bc_cfg["loss_ratio"]
    ).to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.base.amp)

    for epoch in range(cfg.base.epoch_to_start + 1, cfg.base.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            imgs = batch["img"].to(device)
            ids  = batch["input_ids"].to(device)
            am   = batch["attention_mask"].to(device)
            tt   = batch.get("token_type_ids", None)
            if tt is not None:
                tt = tt.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=cfg.base.amp):
                out = model(
                    image=imgs,
                    input_ids=ids,
                    attention_mask=am,
                    token_type_ids=tt
                )
                logits_i2t = out["logits_per_image"]
                logits_t2i = out["logits_per_text"]
                loss_c     = criterion(logits_i2t, logits_t2i)

            scaler.scale(loss_c).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            running_loss += loss_c.item()

        avg_train_loss = running_loss / len(train_loader)
        lr_now = optimizer.param_groups[0]['lr']
        print(f"[Epoch {epoch}] Train loss: {avg_train_loss:.4f}, LR: {lr_now:.2e}")
        writer.add_scalar("train/loss", avg_train_loss, epoch)
        writer.add_scalar("train/lr",   lr_now,           epoch)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                imgs = batch["img"].to(device)
                ids  = batch["input_ids"].to(device)
                am   = batch["attention_mask"].to(device)
                tt   = batch.get("token_type_ids", None)
                if tt is not None:
                    tt = tt.to(device)

                with torch.cuda.amp.autocast(enabled=cfg.base.amp):
                    out = model(
                        image=imgs,
                        input_ids=ids,
                        attention_mask=am,
                        token_type_ids=tt
                    )
                    logits_i2t = out["logits_per_image"]
                    logits_t2i = out["logits_per_text"]
                    loss_v     = criterion(logits_i2t, logits_t2i)

                val_loss += loss_v.item()

        avg_val_loss = val_loss / len(valid_loader)
        print(f"[Epoch {epoch}] Valid loss: {avg_val_loss:.4f}")
        writer.add_scalar("valid/loss", avg_val_loss, epoch)

        ckpt_dir = to_absolute_path(cfg.base.output.checkpoint)
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pth")
        model.save(ckpt_path, optimizer=optimizer, scheduler=scheduler, epoch=epoch)
        print(f"[INFO] Saved checkpoint: {ckpt_path}")

    writer.close()
    print("[INFO] Training complete.")


if __name__ == "__main__":
    main()
