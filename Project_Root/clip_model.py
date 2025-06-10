import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoConfig, AutoModel, AutoModelForImageClassification
from loss_functions import BreastClipLoss

class CLIPCustom(nn.Module):
    def __init__(self,
                 model_cfg: dict,
                 loss_cfg: dict = None,
                 multi_view: bool = False,
                 view_weights: tuple = (0.5, 0.5)):
        super().__init__()
        assert model_cfg["name"] == "clip_custom", "cfg.model.name must be 'clip_custom'"

        self.cfg = model_cfg
        self.temperature = nn.Parameter(
            torch.tensor(model_cfg.get("temperature", 1.0), dtype=torch.float32),
            requires_grad=True
        )

        # Image encoder setup
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
                features_only=False
            )
            image_feature_dim = self.image_backbone.num_features

        elif source == "huggingface":
            hf_img_cfg = AutoConfig.from_pretrained(
                img_name,
                cache_dir=ie.get("cache_dir", None),
                trust_remote_code=ie.get("trust_remote_code", False),
                add_pooling_layer=False
            )
            self.image_backbone = AutoModelForImageClassification.from_pretrained(
                img_name,
                config=hf_img_cfg,
                cache_dir=ie.get("cache_dir", None),
                trust_remote_code=ie.get("trust_remote_code", False)
            )
            image_feature_dim = self.image_backbone.config.hidden_size
        else:
            raise ValueError(f"Unknown image_encoder.source: {source}")

        if freeze_img:
            for p in self.image_backbone.parameters():
                p.requires_grad = False

        self.drop_image_features = drop_features

        # Text encoder setup
        te = model_cfg["text_encoder"]
        source_te = te["source"].lower()
        txt_name = te["name"]
        grad_ckpt = te.get("gradient_checkpointing", False)
        pooling = te.get("pooling", "eos").lower()

        if source_te != "huggingface":
            raise ValueError(f"Unsupported text_encoder.source: {source_te}")

        hf_txt_cfg = AutoConfig.from_pretrained(
            txt_name,
            cache_dir=te.get("cache_dir", None),
            trust_remote_code=te.get("trust_remote_code", False),
            gradient_checkpointing=grad_ckpt,
            add_pooling_layer=False
        )
        self.text_backbone = AutoModel.from_pretrained(
            txt_name,
            config=hf_txt_cfg,
            cache_dir=te.get("cache_dir", None),
            trust_remote_code=te.get("trust_remote_code", False)
        )
        text_hidden_size = self.text_backbone.config.hidden_size
        assert pooling in {"eos", "bos", "mean"}, "pooling must be 'eos', 'bos' or 'mean'"
        self.text_pooling = pooling

        # Projection heads
        ph = model_cfg["projection_head"]
        proj_dim = ph["proj_dim"]
        dropout = ph["dropout"]
        ph_name = ph["name"].lower()
        mlp_hidden = ph.get("mlp_hidden_dim", None)
        num_layers = ph.get("num_layers", 1)

        # Image projection
        if ph_name == "linear":
            self.image_proj = nn.Sequential(
                nn.Linear(image_feature_dim, proj_dim),
                nn.Dropout(dropout),
                nn.LayerNorm(proj_dim),
            )
        elif ph_name == "mlp":
            layers = []
            in_dim = image_feature_dim
            for _ in range(num_layers - 1):
                layers += [nn.Linear(in_dim, mlp_hidden), nn.ReLU(), nn.Dropout(dropout)]
                in_dim = mlp_hidden
            layers += [nn.Linear(in_dim, proj_dim), nn.LayerNorm(proj_dim)]
            self.image_proj = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unknown projection_head.name {ph_name} for image")

        # Text projection
        if ph_name == "linear":
            self.text_proj = nn.Sequential(
                nn.Linear(text_hidden_size, proj_dim),
                nn.Dropout(dropout),
                nn.LayerNorm(proj_dim),
            )
        elif ph_name == "mlp":
            layers = []
            in_dim = text_hidden_size
            for _ in range(num_layers - 1):
                layers += [nn.Linear(in_dim, mlp_hidden), nn.ReLU(), nn.Dropout(dropout)]
                in_dim = mlp_hidden
            layers += [nn.Linear(in_dim, proj_dim), nn.LayerNorm(proj_dim)]
            self.text_proj = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unknown projection_head.name {ph_name} for text")

        # Optional classification head
        ch_cfg = model_cfg.get("classification_head", None)
        if ch_cfg and ch_cfg.get("enabled", False):
            num_classes = ch_cfg["num_classes"]
            self.classifier = nn.Sequential(
                nn.Linear(proj_dim * 2, ch_cfg["hidden_dim"]),
                nn.ReLU(),
                nn.Dropout(ch_cfg["dropout"]),
                nn.Linear(ch_cfg["hidden_dim"], num_classes)
            )
            self.has_classifier = True
        else:
            self.has_classifier = False

        self.mixed_precision = model_cfg.get("mixed_precision", False)

        # Initialize loss
        if loss_cfg:
            self.loss_fn = BreastClipLoss(
                label_smoothing=loss_cfg.get("label_smoothing", 0.0),
                i2i_weight=loss_cfg.get("i2i_weight", 1.0),
                t2t_weight=loss_cfg.get("t2t_weight", 1.0),
                loss_ratio=loss_cfg.get("loss_ratio", 1.0),
                multi_view=multi_view,
                view_weights=view_weights
            )
        else:
            self.loss_fn = None

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        feats = self.image_backbone(image)
        if self.drop_image_features:
            return feats
        proj = self.image_proj(feats)
        return F.normalize(proj, dim=-1)

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
        hidden = out.last_hidden_state
        if self.text_pooling == "eos":
            eos_id = self.text_backbone.config.eos_token_id
            seq_lens = attention_mask.sum(dim=1)
            last_idxs = seq_lens - 1
            eos_pos = (input_ids == eos_id).nonzero(as_tuple=False)
            for b_idx, pos in eos_pos:
                last_idxs[b_idx] = pos
            feats = hidden[torch.arange(hidden.size(0)), last_idxs]
        elif self.text_pooling == "bos":
            feats = hidden[:, 0, :]
        else:  # mean
            mask = attention_mask.unsqueeze(-1).float()
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1)
            feats = summed / (counts + 1e-8)
        proj = self.text_proj(feats)
        return F.normalize(proj, dim=-1)

    def compute_logits(self, image_embeds: torch.Tensor,text_embeds: torch.Tensor):
        logits_i2t = image_embeds @ text_embeds.t() / self.temperature
        logits_t2i = logits_i2t.t()
        return logits_i2t, logits_t2i

    def forward(self,
                images,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                return_loss: bool = False,
                return_embeddings_only: bool = False):
        # Multi-view path
        if self.loss_fn and self.loss_fn.multi_view and isinstance(images, (list, tuple)) and len(images) == 2:
            img1, img2 = images
            emb1 = self.encode_image(img1)
            emb2 = self.encode_image(img2)
            txt_emb = self.encode_text(input_ids, attention_mask, token_type_ids)

            logits1, logits1_t = self.compute_logits(emb1, txt_emb)
            logits2, logits2_t = self.compute_logits(emb2, txt_emb)

            out = {
                "image_embeds": (emb1, emb2),
                "text_embeds": txt_emb,
                "logits_per_image": (logits1, logits2),
                "logits_per_text": (logits1_t, logits2_t)
            }
            if return_loss:
                out["loss"] = self.loss_fn(logits1, logits1_t, logits2, logits2_t)
            if return_embeddings_only:
                return emb1, emb2, txt_emb
            return out

        # Single-view path
        image = images if not isinstance(images, (list, tuple)) else images[0]
        emb_img = self.encode_image(image)
        emb_txt = self.encode_text(input_ids, attention_mask, token_type_ids)
        logits_i2t, logits_t2i = self.compute_logits(emb_img, emb_txt)

        out = {
            "image_embeds": emb_img,
            "text_embeds": emb_txt,
            "logits_per_image": logits_i2t,
            "logits_per_text": logits_t2i
        }
        if self.loss_fn and return_loss:
            out["loss"] = self.loss_fn(logits_i2t, logits_t2i)
        if return_embeddings_only:
            return emb_img, emb_txt
        return out
