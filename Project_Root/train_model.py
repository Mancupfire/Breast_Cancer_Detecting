import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path
from Mammo_CLIP.Project_Root.clip_model import CLIPCustom
from DatModule.datamodule import DataModule, get_transforms, MammoDataset_Mapper
from loss_functions import BreastClipLoss
from transformers import get_cosine_schedule_with_warmup
import random, numpy as np
import os

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

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model = CLIPCustom(model_cfg).to(device)

    if cfg.base.resume_training and cfg.base.checkpoint_to_start:
        ckpt = to_absolute_path(cfg.base.checkpoint_to_start)
        last_epoch = model.load(ckpt, optimizer=None, scheduler=None)
        print(f"[INFO] Resumed model from {ckpt}, last epoch = {last_epoch}")

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

    df = dm.read_metadata_csv()  # assume DataModule has this helper
    df = df[(df["Mass"] == 1) | (df["Suspicious_Calcification"] == 1)]
    train_df = df[df["split"] == "training"].reset_index(drop=True)
    valid_df = df[df["split"] == "test"].reset_index(drop=True)

    train_ds = MammoDataset_Mapper(dm, train_df, transform=get_transforms(dm, split="train"))
    valid_ds = MammoDataset_Mapper(dm, valid_df, transform=get_transforms(dm, split="valid"))

    # Data Loaders
    if cfg.dataloader.get("balanced", False):
        weights_path = os.path.join(cfg.base.output.checkpoint, f"weights_fold{cfg.base.fold}.pkl")
        if os.path.exists(weights_path):
            import pickle
            with open(weights_path, "rb") as f:
                weights = pickle.load(f)
        else:
            pos_w = cfg.dataloader.sampler_weights[f"fold{cfg.base.fold}"]["pos_wt"]
            neg_w = cfg.dataloader.sampler_weights[f"fold{cfg.base.fold}"]["neg_wt"]
            train_df["w"] = train_df["cancer"].map({1: pos_w, 0: neg_w})
            weights = train_df["w"].values
            import pickle
            with open(weights_path, "wb") as f:
                pickle.dump(weights, f)
        sampler = torch.utils.data.WeightedRandomSampler(weights.tolist(), len(weights), replacement=True)
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size   = cfg.dataloader.batch_size,
            sampler      = sampler,
            num_workers  = cfg.dataloader.num_workers,
            pin_memory   = cfg.dataloader.pin_memory,
            drop_last    = cfg.dataloader.drop_last
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size   = cfg.dataloader.batch_size,
            shuffle      = cfg.dataloader.shuffle,
            num_workers  = cfg.dataloader.num_workers,
            pin_memory   = cfg.dataloader.pin_memory,
            drop_last    = cfg.dataloader.drop_last
        )

    valid_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size   = cfg.dataloader.batch_size,
        shuffle      = False,
        num_workers  = cfg.dataloader.num_workers,
        pin_memory   = cfg.dataloader.pin_memory,
        drop_last    = cfg.dataloader.drop_last
    )

    print(f"[INFO] Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")

    # Optimizer
    if cfg.optimizer.name.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr           = cfg.optimizer.config.lr,
            weight_decay = cfg.optimizer.config.weight_decay
        )
    else:
        raise ValueError("Unsupported optimizer: {}".format(cfg.optimizer.name))

    # Scheduler
    if cfg.scheduler.name.lower() == "cosine":
        total_steps  = len(train_loader) * cfg.scheduler.config.total_epochs
        warmup_steps = len(train_loader) * cfg.scheduler.config.warmup_epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps   = warmup_steps,
            num_training_steps = total_steps,
            num_cycles         = 0.5
        )
    else:
        scheduler = None

    # Loss
    bc_cfg = OmegaConf.to_container(cfg.loss.breast_clip, resolve=True)
    criterion = BreastClipLoss(
        label_smoothing = bc_cfg["label_smoothing"],
        i2i_weight      = bc_cfg["i2i_weight"],
        t2t_weight      = bc_cfg["t2t_weight"],
        loss_ratio      = bc_cfg["loss_ratio"]
    ).to(device)

    # Training loop skeleton
    for epoch in range(cfg.base.epoch_to_start + 1, cfg.base.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch["img"].to(device)             # (B,3,H,W)
            input_ids = batch["input_ids"].to(device)    # (B,seq_len)
            att_mask = batch["attention_mask"].to(device)
            tok_type = batch.get("token_type_ids", None)
            if tok_type is not None:
                tok_type = tok_type.to(device)

            outputs = model(
                image=images,
                input_ids=input_ids,
                attention_mask=att_mask,
                token_type_ids=tok_type
            )
            logits_i2t = outputs["logits_per_image"]
            logits_t2i = outputs["logits_per_text"]

            loss_c = CLIPCustom.compute_contrastive_loss(logits_i2t, logits_t2i)

            optimizer.zero_grad()
            loss_c.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            running_loss += loss_c.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch}] train_loss = {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                images = batch["img"].to(device)
                input_ids = batch["input_ids"].to(device)
                att_mask = batch["attention_mask"].to(device)
                tok_type = batch.get("token_type_ids", None)
                if tok_type is not None:
                    tok_type = tok_type.to(device)

                outputs = model(
                    image=images,
                    input_ids=input_ids,
                    attention_mask=att_mask,
                    token_type_ids=tok_type
                )
                logits_i2t = outputs["logits_per_image"]
                logits_t2i = outputs["logits_per_text"]
                loss_v = CLIPCustom.compute_contrastive_loss(logits_i2t, logits_t2i)
                val_loss += loss_v.item()

            avg_val_loss = val_loss / len(valid_loader)
            print(f"[Epoch {epoch}] valid_loss = {avg_val_loss:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(cfg.base.output.checkpoint, f"epoch_{epoch}.pth")
        model.save(ckpt_path, optimizer=optimizer, scheduler=scheduler, epoch=epoch)
        print(f"[INFO] Saved checkpoint: {ckpt_path}")

        # Early stopping or “best” logic can be added here…

    print("[INFO] Training complete.")


if __name__ == "__main__":
    main()
