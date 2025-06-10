import os
import random
import pickle

import numpy as np
import torch
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


def build_sampler(train_df, cfg, fold):
    weights_path = os.path.join(cfg.base.output.checkpoint, f"weights_fold{fold}.pkl")
    if os.path.exists(weights_path):
        with open(weights_path, "rb") as f:
            weights = pickle.load(f)
    else:
        sw = cfg.base.sampler_weights
        # load from JSON file or direct dict
        if isinstance(sw, str) and os.path.exists(to_absolute_path(sw)):
            import json
            with open(to_absolute_path(sw)) as f:
                wdict = json.load(f)
            pos_w = wdict[f"fold{fold}"]["pos_wt"]
            neg_w = wdict[f"fold{fold}"]["neg_wt"]
        else:
            pos_w = sw[f"fold{fold}"]["pos_wt"]
            neg_w = sw[f"fold{fold}"]["neg_wt"]
        train_df["w"] = train_df["cancer"].map({1: pos_w, 0: neg_w})
        weights = train_df["w"].values
        arr = np.array(weights, dtype=np.float32)
        assert arr.shape[0] == len(train_df), "Sampler weights length mismatch!"
        assert np.all(arr >= 0), "Negative weights found!"
        total = arr.sum()
        assert total > 0, "Sum of sampler weights must be positive!"
        zeros = int((arr == 0).sum())
        if zeros > 0:
            print(f"⚠️ {zeros}/{len(arr)} samples have zero weight.")
        with open(weights_path, "wb") as f:
            pickle.dump(weights, f)
    return WeightedRandomSampler(weights.tolist(), len(weights), replacement=True)


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    # reproducibility
    torch.manual_seed(cfg.base.seed)
    np.random.seed(cfg.base.seed)
    random.seed(cfg.base.seed)
    torch.cuda.manual_seed_all(cfg.base.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)
    print(OmegaConf.to_yaml(cfg))

    # model
    model = CLIPCustom(OmegaConf.to_container(cfg.model, resolve=True)).to(device)
    if cfg.base.resume_training and cfg.base.checkpoint_to_start:
        ckpt = to_absolute_path(cfg.base.checkpoint_to_start)
        last_epoch = model.load(ckpt, optimizer=None, scheduler=None)
        print(f"[INFO] Resumed from {ckpt}, epoch={last_epoch}")

    # datamodule
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

    df = dm.read_metadata_csv()
    df = df[(df['Mass']==1)|(df['Suspicious_Calcification']==1)]
    train_df = df[df['split']=='training'].reset_index(drop=True)
    valid_df = df[df['split']=='test'].reset_index(drop=True)

    train_ds = MammoDataset_Mapper(dm, train_df,
                   transform=get_transforms(dm, split='train'))
    valid_ds = MammoDataset_Mapper(dm, valid_df,
                   transform=get_transforms(dm, split='valid'))

    # dataloaders
    if cfg.base.balanced_dataloader:
        sampler = build_sampler(train_df, cfg, cfg.base.fold)
        train_loader = DataLoader(train_ds,
                            batch_size=cfg.base.batch_size,
                            sampler=sampler,
                            num_workers=cfg.base.num_workers,
                            pin_memory=True,
                            drop_last=True)
    else:
        train_loader = DataLoader(train_ds,
                            batch_size=cfg.base.batch_size,
                            shuffle=True,
                            num_workers=cfg.base.num_workers,
                            pin_memory=True,
                            drop_last=True)
    valid_loader = DataLoader(valid_ds,
                            batch_size=cfg.base.batch_size,
                            shuffle=False,
                            num_workers=cfg.base.num_workers,
                            pin_memory=True,
                            drop_last=False)

    print(f"Train batches: {len(train_loader)}, Valid: {len(valid_loader)}")

    # writer
    writer = SummaryWriter(to_absolute_path(cfg.base.output.tensorboard))

    # optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.config.lr,
        weight_decay=cfg.optimizer.config.weight_decay
    )
    # scheduler
    total_steps = len(train_loader)*cfg.base.epochs
    warmup_steps= len(train_loader)*cfg.scheduler.config.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5
    )

    # loss
    bc = OmegaConf.to_container(cfg.loss.breast_clip, resolve=True)
    criterion = BreastClipLoss(
        label_smoothing=bc['label_smoothing'],
        i2i_weight=bc['i2i_weight'],
        t2t_weight=bc['t2t_weight'],
        loss_ratio=bc['loss_ratio'],
        multi_view=cfg.base.multi_view,
        view_weights=tuple(cfg.loss.multi_view.view_weight)
    ).to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.base.amp)

    # train loop
    for epoch in range(cfg.base.epoch_to_start+1, cfg.base.epochs+1):
        model.train()
        run_loss=0.0
        for batch in train_loader:
            imgs = batch['img']  # shape (B,2,C,H,W) if multi_view
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=cfg.base.amp):
                if cfg.base.multi_view:
                    I1 = model.encode_image(imgs[:,0].to(device))
                    I2 = model.encode_image(imgs[:,1].to(device))
                    T  = model.encode_text(input_ids=ids, attention_mask=mask)
                    L1_i2t, L1_t2i = model.compute_logits(I1, T)
                    L2_i2t, L2_t2i = model.compute_logits(I2, T)
                    loss = criterion(L1_i2t, L1_t2i, L2_i2t, L2_t2i)
                else:
                    out = model(images=imgs.to(device), input_ids=ids, attention_mask=mask)
                    loss = criterion(out['logits_per_image'], out['logits_per_text'])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            run_loss += loss.item()

        avg_train=run_loss/len(train_loader)
        writer.add_scalar('train/loss', avg_train, epoch)

        # validation
        model.eval()
        val_loss=0.0
        with torch.no_grad():
            for batch in valid_loader:
                imgs = batch['img']
                ids  = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                with torch.cuda.amp.autocast(enabled=cfg.base.amp):
                    if cfg.base.multi_view:
                        I1 = model.encode_image(imgs[:,0].to(device))
                        I2 = model.encode_image(imgs[:,1].to(device))
                        T  = model.encode_text(input_ids=ids, attention_mask=mask)
                        L1_i2t, L1_t2i = model.compute_logits(I1, T)
                        L2_i2t, L2_t2i = model.compute_logits(I2, T)
                        loss_v = criterion(L1_i2t, L1_t2i, L2_i2t, L2_t2i)
                    else:
                        out = model(images=imgs.to(device), input_ids=ids, attention_mask=mask)
                        loss_v = criterion(out['logits_per_image'], out['logits_per_text'])
                val_loss += loss_v.item()
        avg_val=val_loss/len(valid_loader)
        writer.add_scalar('valid/loss', avg_val, epoch)

        # checkpoint
        ckpt_dir = to_absolute_path(cfg.base.output.checkpoint)
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_pt  = os.path.join(ckpt_dir, f'epoch_{epoch}.pth')
        model.save(ckpt_pt, optimizer=optimizer, scheduler=scheduler, epoch=epoch)
        print(f"[Epoch {epoch}] train={avg_train:.4f}, val={avg_val:.4f}")

    writer.close()


if __name__ == "__main__":
    main()
