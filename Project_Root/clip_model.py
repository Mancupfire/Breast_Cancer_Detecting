import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoConfig, AutoModel, AutoModelForImageClassification
import os


class CLIPCustom(nn.Module):
    def __init__(self, model_cfg: dict):
        super().__init__()
        assert model_cfg["name"] == "clip_custom", "cfg.model.name must be 'clip_custom'"

        self.cfg = model_cfg

        self.temperature = nn.Parameter(
            torch.tensor(model_cfg["temperature"], dtype=torch.float32),
            requires_grad=True
        )

        ie = model_cfg["image_encoder"]
        source = ie["source"].lower()
        img_name = ie["name"]
        pretrained = ie.get("pretrained", True)
        freeze_img = ie.get("freeze", False)
        drop_features = ie.get("drop_features", False)

        if source == "cnn":
            self.image_backbone = timm.create_model(
                img_name,
                pretrained=pretrained,
                features_only=False   # we want the final pooled vector
            )
            image_feature_dim = self.image_backbone.num_features

        elif source == "huggingface":
            # Use a HuggingFace ViT or similar
            hf_img_cfg = AutoConfig.from_pretrained(
                img_name,
                cache_dir=ie.get("cache_dir", None),
                trust_remote_code=ie.get("trust_remote_code", False),
                add_pooling_layer=False
            )
            # Using AutoModelForImageClassification gives us a 'classifier' head we can drop
            self.image_backbone = AutoModelForImageClassification.from_pretrained(
                img_name,
                config=hf_img_cfg,
                cache_dir=ie.get("cache_dir", None),
                trust_remote_code=ie.get("trust_remote_code", False)
            )
            # Many HF image models have a “classifier” on top—strip it off by taking .base_model
            # but for simplicity, assume we want last hidden state + global_pool
            image_feature_dim = self.image_backbone.config.hidden_size

        else:
            raise ValueError(f"Unknown image_encoder.source: {source}")

        if freeze_img:
            for p in self.image_backbone.parameters():
                p.requires_grad = False

        self.drop_image_features = drop_features

        te = model_cfg["text_encoder"]
        source_te = te["source"].lower()
        txt_name = te["name"]
        pretrained_te = te.get("pretrained", True)
        grad_ckpt = te.get("gradient_checkpointing", False)
        pooling = te.get("pooling", "eos").lower()
        cache_dir = te.get("cache_dir", None)
        trust_code = te.get("trust_remote_code", False)
        mlm_head = te.get("mlm_head", False)
        freeze_txt = te.get("freeze", False)

        if source_te != "huggingface":
            raise ValueError(f"Unsupported text_encoder.source: {source_te}")

        hf_txt_cfg = AutoConfig.from_pretrained(
            txt_name,
            cache_dir=cache_dir,
            trust_remote_code=trust_code,
            gradient_checkpointing=grad_ckpt,
            add_pooling_layer=False
        )
        self.text_backbone = AutoModel.from_pretrained(
            txt_name,
            config=hf_txt_cfg,
            cache_dir=cache_dir,
            trust_remote_code=trust_code
        )
        text_hidden_size = self.text_backbone.config.hidden_size

        assert pooling in {"eos", "bos", "mean"}, "pooling must be 'eos', 'bos' or 'mean'"
        self.text_pooling = pooling

        if freeze_txt:
            for p in self.text_backbone.parameters():
                p.requires_grad = False

        # Projection Head for text
        ph = model_cfg["projection_head"]
        proj_dim = ph["proj_dim"]
        dropout = ph["dropout"]
        ph_name = ph["name"].lower()
        mlp_hidden = ph.get("mlp_hidden_dim", None)
        num_layers = ph.get("num_layers", 1)

        # Build image projection
        if ph_name == "linear":
            self.image_proj = nn.Sequential(
                nn.Linear(image_feature_dim, proj_dim),
                nn.Dropout(dropout),
                nn.LayerNorm(proj_dim),
            )
        elif ph_name == "mlp":
            assert mlp_hidden is not None, "You must specify mlp_hidden_dim for MLP"
            # e.g. stack if num_layers > 1
            layers = []
            in_dim = image_feature_dim
            for i in range(num_layers - 1):
                layers.append(nn.Linear(in_dim, mlp_hidden))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_dim = mlp_hidden
            # final layer → proj_dim
            layers.append(nn.Linear(in_dim, proj_dim))
            layers.append(nn.LayerNorm(proj_dim))
            self.image_proj = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unknown projection_head.name {ph_name} for image")

        # Build text projection
        if ph_name == "linear":
            self.text_proj = nn.Sequential(
                nn.Linear(text_hidden_size, proj_dim),
                nn.Dropout(dropout),
                nn.LayerNorm(proj_dim),
            )
        elif ph_name == "mlp":
            assert mlp_hidden is not None, "You must specify mlp_hidden_dim for MLP"
            layers = []
            in_dim = text_hidden_size
            for i in range(num_layers - 1):
                layers.append(nn.Linear(in_dim, mlp_hidden))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_dim = mlp_hidden
            layers.append(nn.Linear(in_dim, proj_dim))
            layers.append(nn.LayerNorm(proj_dim))
            self.text_proj = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unknown projection_head.name {ph_name} for text")

        # Classification Head
        ch_cfg = model_cfg.get("classification_head", None)
        if ch_cfg is not None and ch_cfg.get("enabled", False):
            num_classes = ch_cfg["num_classes"]
            ch_dropout = ch_cfg["dropout"]
            ch_hidden = ch_cfg["hidden_dim"]
            # e.g. two‐layer MLP on top of proj_dim → class logits
            self.classifier = nn.Sequential(
                nn.Linear(proj_dim, ch_hidden),
                nn.ReLU(),
                nn.Dropout(ch_dropout),
                nn.Linear(ch_hidden, num_classes)
            )
            self.has_classifier = True
        else:
            self.has_classifier = False

        self.mixed_precision = model_cfg.get("mixed_precision", False)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        # Pass through backbone
        img_feats = self.image_backbone(image)  # (B, feature_dim)
        if self.drop_image_features:
            return img_feats

        proj = self.image_proj(img_feats)       # (B, proj_dim)
        emb  = F.normalize(proj, dim=-1)        # (B, proj_dim)
        return emb

    def encode_text(self,
                    input_ids: torch.LongTensor,
                    attention_mask: torch.LongTensor = None,
                    token_type_ids: torch.LongTensor = None) -> torch.Tensor:
        out = self.text_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        last_hidden = out.last_hidden_state  # (B, seq_len, hidden_size)
        B, seq_len, hidden_size = last_hidden.size()

        if self.text_pooling == "eos":
            eos_id = self.text_backbone.config.eos_token_id
            seq_lens = attention_mask.sum(dim=1)          # (B,)
            last_idxs = seq_lens - 1                      # fallback: last non‐masked
            eos_positions = (input_ids == eos_id).nonzero(as_tuple=False)
            for b_idx, pos in eos_positions:
                last_idxs[b_idx] = pos
            gather_idxs = last_idxs.to(image.device)      # (B,)
            txt_feats = last_hidden[torch.arange(B), gather_idxs]  # (B, hidden_size)

        elif self.text_pooling == "bos":
            txt_feats = last_hidden[:, 0, :]              # (B, hidden_size)

        elif self.text_pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()   # (B, seq_len, 1)
            summed = (last_hidden * mask).sum(dim=1)      # (B, hidden_size)
            counts = mask.sum(dim=1)                      # (B, 1)
            txt_feats = summed / (counts + 1e-8)          # (B, hidden_size)

        else:
            raise ValueError(f"Unknown pooling: {self.text_pooling}")

        proj = self.text_proj(txt_feats)           # (B, proj_dim)
        emb  = F.normalize(proj, dim=-1)           # (B, proj_dim)
        return emb

    def forward(self, image: torch.Tensor, input_ids: torch.LongTensor, attention_mask: torch.LongTensor = None,
                token_type_ids: torch.LongTensor = None, return_embeddings_only: bool = False):
        B = image.size(0)

        # Image Branch
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            if self.drop_image_features:
                img_feats = self.image_backbone(image)        # (B, feature_dim)
                image_embeds = img_feats
            else:
                img_feats = self.image_backbone(image)        # (B, feature_dim)
                img_proj = self.image_proj(img_feats)         # (B, proj_dim)
                image_embeds = F.normalize(img_proj, dim=-1)  # (B, proj_dim)

            # Text Branch
            out = self.text_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True
            )
            last_hidden = out.last_hidden_state             # (B, seq_len, hidden_size)

            if self.text_pooling == "eos":
                eos_id = self.text_backbone.config.eos_token_id
                seq_lens = attention_mask.sum(dim=1)
                last_idxs = seq_lens - 1
                eos_pos = (input_ids == eos_id).nonzero(as_tuple=False)
                for b_idx, pos in eos_pos:
                    last_idxs[b_idx] = pos
                gather_idxs = last_idxs.to(image.device)
                txt_feats = last_hidden[torch.arange(B), gather_idxs]

            elif self.text_pooling == "bos":
                txt_feats = last_hidden[:, 0, :]

            elif self.text_pooling == "mean":
                mask = attention_mask.unsqueeze(-1).float()
                summed = (last_hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1)
                txt_feats = summed / (counts + 1e-8)

            else:
                raise ValueError(f"Unknown pooling: {self.text_pooling}")

            txt_proj = self.text_proj(txt_feats)           # (B, proj_dim)
            text_embeds = F.normalize(txt_proj, dim=-1)    # (B, proj_dim)

        if return_embeddings_only:
            return image_embeds, text_embeds

        # Cosine Smilarity Logits
        logits_per_image = (image_embeds @ text_embeds.t()) / self.temperature
        logits_per_text  = logits_per_image.t()

        out_dict = {
            "image_embeds": image_embeds,           # (B, proj_dim)
            "text_embeds": text_embeds,             # (B, proj_dim)
            "logits_per_image": logits_per_image,   # (B, B)
            "logits_per_text": logits_per_text      # (B, B)
        }

        # Classification Head
        if self.has_classifier:
            # Example: run classifier on the image‐text “joint” embedding by concatenation
            joint = torch.cat([image_embeds, text_embeds], dim=1)  # (B, 2*proj_dim)
            cls_logits = self.classifier(joint)                    # (B, num_classes)
            out_dict["classifier_logits"] = cls_logits

        return out_dict

    @staticmethod
    def compute_contrastive_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor):
        device = logits_per_image.device
        B = logits_per_image.size(0)
        target = torch.arange(B, dtype=torch.long, device=device)
        loss_i = F.cross_entropy(logits_per_image, target)
        loss_t = F.cross_entropy(logits_per_text,  target)
        return 0.5 * (loss_i + loss_t)

    def save(self, path: str, optimizer=None, scheduler=None, epoch=None):
        state = {
            "model_state_dict": self.state_dict(),
            "temperature": self.temperature.detach().cpu().item()
        }
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()
        if epoch is not None:
            state["epoch"] = epoch
        torch.save(state, path)

    def load(self, path: str, optimizer=None, scheduler=None):

        checkpoint = torch.load(path, map_location="cpu")
        self.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint.get("epoch", None)
