import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPConfig
from torchvision import models

import torch
from transformers import CLIPModel
from torchvision import models


def load_image_encoder(encoder_config: dict) -> torch.nn.Module:
    model_type = encoder_config.get("model_type", "vit").lower()
    if model_type == "vit":
        # use CLIP's ViT vision encoder
        hf_ckpt = encoder_config.get(
            "pretrained_checkpoint", "openai/clip-vit-base-patch32"
        )
        clip = CLIPModel.from_pretrained(hf_ckpt)
        vision = clip.vision_model  # nn.Module
    elif model_type.startswith("efficientnet"):
        # use torchvision EfficientNet
        arch = encoder_config.get("efficientnet_arch", model_type)
        if not hasattr(models, arch):
            raise ValueError(f"EfficientNet arch '{arch}' not found in torchvision.models")
        vision = getattr(models, arch)(pretrained=True)
        # if it has a classifier attribute, remove it
        # users can attach their own head later
    else:
        raise ValueError(f"Unsupported image encoder type: {model_type}")

    return vision


class BreastClip(nn.Module):
    def __init__(self, model_config: dict, loss_config: dict, tokenizer=None):
        super().__init__()
        # Load a pretrained CLIP model
        pretrained = model_config.get(
            "pretrained_checkpoint", "openai/clip-vit-base-patch32"
        )
        self.clip = CLIPModel.from_pretrained(pretrained)

        # Whether to apply a learned projection on text embeddings
        self.projection = model_config.get("projection", False)
        if self.projection:
            text_dim = self.clip.config.projection_dim
            self.text_projection = nn.Linear(text_dim, text_dim)

    def encode_text(self, tokens: dict) -> torch.Tensor:
        # HuggingFace CLIPModel's get_text_features handles projection internally
        txt_feats = self.clip.get_text_features(**tokens)
        if self.projection:
            txt_feats = self.text_projection(txt_feats)
        return txt_feats

    def encode_image(self, inputs: dict) -> torch.Tensor:
        # CLIPModel.get_image_features returns projected features
        img_feats = self.clip.get_image_features(**inputs)
        # raw_features from the vision encoder (before pooling)
        vision_outputs = self.clip.vision_model(**inputs)
        raw = vision_outputs.last_hidden_state  # shape (B, C, H, W)
        return img_feats, raw


class MammoClassification(nn.Module):
    def __init__(self, model_config: dict, model_type: str = "vit"):  # add other types as needed
        super().__init__()
        if model_type.lower() == "vit":
            # Use CLIP's vision encoder for ViT
            clip = CLIPModel.from_pretrained(
                model_config.get(
                    "pretrained_checkpoint", "openai/clip-vit-base-patch32"
                )
            )
            self.encoder = clip.vision_model
            hidden = clip.config.vision_config.hidden_size
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # Classification head
        num_classes = model_config.get("num_classes", 2)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: tensor (B, C, H, W)
        outputs = self.encoder(pixel_values=images)
        cls_token = outputs.last_hidden_state[:, 0]  # [CLS]
        logits = self.classifier(cls_token)
        return logits


class MammoEfficientNet(nn.Module):
    def __init__(self, model_config: dict):
        super().__init__()
        arch = model_config.get("efficientnet_arch", "efficientnet_b0")
        # load a torchvision EfficientNet
        if not hasattr(models, arch):
            raise ValueError(f"EfficientNet arch '{arch}' not found in torchvision.models")
        self.backbone = getattr(models, arch)(pretrained=True)
        in_feats = self.backbone.classifier.in_features
        num_classes = model_config.get("num_classes", 2)
        # replace the classifier
        self.backbone.classifier = nn.Linear(in_feats, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)
