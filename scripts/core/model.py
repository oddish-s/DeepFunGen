"""Flexible model factory for FunTorch5 sequence regression."""
from __future__ import annotations

import copy
import math
import warnings
from functools import lru_cache
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_SEQUENCE_LENGTH = 32


# ---------------------------------------------------------------------------
# Frame encoders
# ---------------------------------------------------------------------------
class FrameEncoderBase(nn.Module):
    """Base interface for modules that embed per-frame content."""

    def __init__(self) -> None:
        super().__init__()
        self.output_dim: int = 0

    def forward(self, frames: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface definition
        raise NotImplementedError


class ConvFrameEncoder(FrameEncoderBase):
    """Lightweight CNN that downsamples spatially and projects to embeddings."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        hidden_channels: Optional[Sequence[int]] = None,
        embed_dim: int = 256,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [32, 64, 128, 256]
        channels = [in_channels] + list(hidden_channels)
        layers: list[nn.Module] = []
        for idx in range(len(channels) - 1):
            layers.append(
                nn.Conv2d(
                    channels[idx],
                    channels[idx + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(norm_layer(channels[idx + 1]))
            layers.append(nn.ReLU(inplace=True))
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(channels[-1], embed_dim)
        self.output_dim = embed_dim

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch, seq, channels, height, width = frames.shape
        flat = frames.view(batch * seq, channels, height, width)
        feats = self.features(flat)
        feats = self.avgpool(feats).flatten(1)
        feats = self.proj(feats)
        return feats.view(batch, seq, -1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)


class ResidualFrameEncoder(FrameEncoderBase):
    """Deeper CNN with residual connections."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        stem_channels: int = 32,
        block_channels: Sequence[int] = (64, 128, 256),
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        blocks: list[nn.Module] = []
        in_ch = stem_channels
        for idx, out_ch in enumerate(block_channels):
            stride = 1 if idx == 0 else 2
            blocks.append(ResidualBlock(in_ch, out_ch, stride))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(in_ch, embed_dim)
        self.output_dim = embed_dim

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch, seq, channels, height, width = frames.shape
        flat = frames.view(batch * seq, channels, height, width)
        feats = self.stem(flat)
        feats = self.blocks(feats)
        feats = self.avgpool(feats).flatten(1)
        feats = self.proj(feats)
        return feats.view(batch, seq, -1)


class ConvNeXtFrameEncoder(FrameEncoderBase):
    """Frame encoder backed by torchvision ConvNeXt variants."""

    def __init__(
        self,
        *,
        variant: str = "convnext_tiny",
        trainable: bool = False,
        projection_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        from torchvision.models import convnext_base, convnext_large, convnext_small, convnext_tiny

        builders: Dict[str, callable] = {
            "convnext_tiny": convnext_tiny,
            "convnext_small": convnext_small,
            "convnext_base": convnext_base,
            "convnext_large": convnext_large,
        }
        if variant not in builders:
            raise ValueError(f"Unsupported ConvNeXt variant '{variant}'. Options: {sorted(builders)}")
        backbone = builders[variant](weights=None)
        self.body = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_features = backbone.classifier[2].in_features
        if projection_dim:
            self.proj = nn.Linear(in_features, int(projection_dim))
            self.output_dim = int(projection_dim)
        else:
            self.proj = nn.Identity()
            self.output_dim = int(in_features)
        if not trainable:
            for param in self.body.parameters():
                param.requires_grad = False
            if isinstance(self.proj, nn.Identity):
                pass
            else:
                for param in self.proj.parameters():
                    param.requires_grad = False
        self._trainable = trainable

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch, seq, channels, height, width = frames.shape
        flat = frames.view(batch * seq, channels, height, width)
        if not self._trainable:
            with torch.no_grad():
                feats = self.body(flat)
                pooled = self.pool(feats).flatten(1)
        else:
            feats = self.body(flat)
            pooled = self.pool(feats).flatten(1)
        projected = self.proj(pooled)
        return projected.view(batch, seq, -1)


# ---------------------------------------------------------------------------
# DINOv3-backed frame encoders
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _load_dinov3_backbone(backbone_name: str) -> nn.Module:
    try:
        model = torch.hub.load("facebookresearch/dinov3", backbone_name, pretrained=False)
    except Exception as exc:  # pragma: no cover - relies on external dependency
        raise RuntimeError(
            "Failed to load DINOv3 backbone. Ensure internet access for torch.hub or install dinov3 package."
        ) from exc
    return model


def _create_dinov3_backbone(backbone_name: str) -> nn.Module:
    base = _load_dinov3_backbone(backbone_name)
    return copy.deepcopy(base)


class Dinov3BaseFrameEncoder(FrameEncoderBase):
    def __init__(
        self,
        *,
        backbone_name: str = "dinov3_vits16",
        pretrained_path: Optional[str] = None,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = _create_dinov3_backbone(backbone_name)

        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu")
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            if missing or unexpected:
                warnings.warn(
                    f"DINOv3 weights loaded with missing keys={missing} unexpected keys={unexpected}",
                    RuntimeWarning,
                )
        if not trainable:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self._trainable = trainable
        self.embed_dim = getattr(self.backbone, "embed_dim", None) or getattr(self.backbone, "num_features")
        if self.embed_dim is None:
            raise AttributeError("Unable to determine DINOv3 embedding dimension")
        self.backbone.eval()

    def _forward_backbone(self, images: torch.Tensor) -> torch.Tensor:
        if self._trainable:
            return self.backbone(images)
        with torch.no_grad():
            return self.backbone(images)


class Dinov3TwoTowerFrameEncoder(Dinov3BaseFrameEncoder):
    """Encodes previous/current frames separately and combines their representations."""

    def __init__(
        self,
        *,
        backbone_name: str = "dinov3_vits16",
        pretrained_path: Optional[str] = None,
        projection_dim: Optional[int] = 512,
        normalize: bool = False,
        trainable: bool = False,
    ) -> None:
        super().__init__(
            backbone_name=backbone_name,
            pretrained_path=pretrained_path,
            trainable=trainable,
        )
        combined_dim = self.embed_dim * 3
        if projection_dim:
            self.project = nn.Linear(combined_dim, int(projection_dim))
            self.output_dim = int(projection_dim)
        else:
            self.project = nn.Identity()
            self.output_dim = combined_dim
        self.normalize = normalize

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch, seq, channels, height, width = frames.shape
        prev = torch.roll(frames, shifts=1, dims=1)
        prev[:, 0] = frames[:, 0]
        flat_prev = prev.view(batch * seq, channels, height, width)
        flat_curr = frames.view(batch * seq, channels, height, width)
        prev_feat = self._forward_backbone(flat_prev)
        curr_feat = self._forward_backbone(flat_curr)
        combined = torch.cat([curr_feat, prev_feat, curr_feat - prev_feat], dim=1)
        combined = combined.view(batch, seq, -1)
        projected = self.project(combined)
        if self.normalize:
            projected = F.normalize(projected, dim=-1)
        return projected


class Dinov3StackFrameEncoder(Dinov3BaseFrameEncoder):
    """Stacks resized previous/current frames vertically before encoding."""

    def __init__(
        self,
        *,
        backbone_name: str = "dinov3_vits16",
        pretrained_path: Optional[str] = None,
        target_size: Sequence[int] = (224, 224),
        projection_dim: Optional[int] = None,
        normalize: bool = False,
        trainable: bool = False,
    ) -> None:
        super().__init__(
            backbone_name=backbone_name,
            pretrained_path=pretrained_path,
            trainable=trainable,
        )
        height, width = int(target_size[0]), int(target_size[1])
        self.half_height = max(1, height // 2)
        self.width = width
        if projection_dim:
            self.project = nn.Linear(self.embed_dim, int(projection_dim))
            self.output_dim = int(projection_dim)
        else:
            self.project = nn.Identity()
            self.output_dim = self.embed_dim
        self.normalize = normalize

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch, seq, channels, height, width = frames.shape
        prev = torch.roll(frames, shifts=1, dims=1)
        prev[:, 0] = frames[:, 0]
        flat_prev = prev.view(batch * seq, channels, height, width)
        flat_curr = frames.view(batch * seq, channels, height, width)
        prev_resized = F.interpolate(flat_prev, size=(self.half_height, self.width), mode="bilinear", align_corners=False)
        curr_resized = F.interpolate(flat_curr, size=(self.half_height, self.width), mode="bilinear", align_corners=False)
        stacked = torch.cat([prev_resized, curr_resized], dim=2)
        features = self._forward_backbone(stacked)
        features = features.view(batch, seq, -1)
        projected = self.project(features)
        if self.normalize:
            projected = F.normalize(projected, dim=-1)
        return projected


# ---------------------------------------------------------------------------
# Sequence encoders
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        return x + self.pe[:, :length]


class TransformerSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        *,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.pos_encoder = PositionalEncoding(input_dim, max_len=sequence_length + 10)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation=activation,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoder(x)
        return self.encoder(x)


class LSTMSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
            bidirectional=bidirectional,
        )
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return output


class TemporalConvBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        *,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.causal = bool(causal)
        if self.causal:
            self.left_padding = (kernel_size - 1) * dilation
            conv_padding = 0
        else:
            self.left_padding = ((kernel_size - 1) // 2) * dilation
            conv_padding = self.left_padding
        self.conv1 = nn.Conv1d(in_dim, out_dim, kernel_size, padding=conv_padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size, padding=conv_padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_dim, out_dim, kernel_size=1) if in_dim != out_dim else nn.Identity()
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def _pad_left(tensor: torch.Tensor) -> torch.Tensor:
            if self.causal and self.left_padding > 0:
                return F.pad(tensor, (self.left_padding, 0))
            return tensor

        out = self.conv1(_pad_left(x))
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(_pad_left(out))
        out = self.dropout(out)
        res = self.residual(x)
        return self.activation(out + res)


class TCNSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 256,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        dilation_base: int = 2,
        causal: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for layer_idx in range(num_layers):
            dilation = dilation_base ** layer_idx
            layers.append(
                TemporalConvBlock(
                    in_dim,
                    hidden_dim,
                    kernel_size,
                    dilation,
                    dropout,
                    causal=causal,
                )
            )
            in_dim = hidden_dim
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, dim)
        x = x.transpose(1, 2)  # (batch, dim, seq)
        out = self.network(x)
        return out.transpose(1, 2)


# ---------------------------------------------------------------------------
# Prediction head and full model
# ---------------------------------------------------------------------------
class PredictionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: Optional[Sequence[int]] = None, dropout: float = 0.1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        hidden_layers = [int(h) for h in hidden_layers] if hidden_layers else []
        for hidden in hidden_layers:
            layers.append(nn.LayerNorm(current_dim))
            layers.append(nn.Linear(current_dim, hidden))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden
        layers.append(nn.LayerNorm(current_dim))
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SequenceRegressor(nn.Module):
    """Flexible sequence-to-regression model composed from modular encoders."""

    def __init__(
        self,
        *,
        sequence_length: int,
        frame_encoder: FrameEncoderBase,
        temporal_encoder: nn.Module,
        head: nn.Module,
        summary: str = "last",
    ) -> None:
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.frame_encoder = frame_encoder
        self.temporal_encoder = temporal_encoder
        self.head = head
        mode = str(summary).lower()
        if mode not in {"last", "mean", "center", "full"}:
            raise ValueError("summary must be one of {'last', 'mean', 'center', 'full'}")
        self.summary_mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected input with shape (batch, seq, C, H, W); got {tuple(x.shape)}")
        frame_features = self.frame_encoder(x)
        sequence_features = self.temporal_encoder(frame_features)
        if self.summary_mode == "full":
            summary = sequence_features
        elif sequence_features.ndim == 3:
            summary = self._summarize(sequence_features)
        else:
            summary = sequence_features
        output = self.head(summary)
        return output.squeeze(-1)

    def _summarize(self, features: torch.Tensor) -> torch.Tensor:
        if self.summary_mode == "full":
            return features
        if self.summary_mode == "last":
            return features[:, -1, :]
        if self.summary_mode == "mean":
            return features.mean(dim=1)
        # center
        length = features.size(1)
        center_idx = max(0, min(length - 1, length // 2))
        return features[:, center_idx, :]


class FrameClassifier(nn.Module):
    """Classify ignore frames using the existing frame encoder stack."""

    def __init__(self, frame_encoder: FrameEncoderBase, head: nn.Module) -> None:
        super().__init__()
        self.frame_encoder = frame_encoder
        self.head = head

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim == 4:
            batch_input = frames.unsqueeze(1)
        elif frames.ndim == 5 and frames.size(1) == 1:
            batch_input = frames
        else:
            raise ValueError(
                f"Expected input with shape (batch, C, H, W) or (batch, 1, C, H, W); got {tuple(frames.shape)}"
            )
        features = self.frame_encoder(batch_input)
        if features.ndim != 3 or features.size(1) == 0:
            raise RuntimeError("Frame encoder did not return expected (batch, seq, dim) features.")
        logits = self.head(features[:, 0, :])
        return logits.squeeze(-1)


# ---------------------------------------------------------------------------
# Builders and public factory
# ---------------------------------------------------------------------------
def _build_conv_frame_encoder(cfg: Mapping[str, Any]) -> FrameEncoderBase:
    embed_dim = int(cfg.get("embed_dim", 256))
    in_channels = int(cfg.get("in_channels", 3))
    channels = cfg.get("channels")
    if channels is not None:
        channels = [int(c) for c in channels]
    return ConvFrameEncoder(in_channels=in_channels, hidden_channels=channels, embed_dim=embed_dim)


def _build_residual_frame_encoder(cfg: Mapping[str, Any]) -> FrameEncoderBase:
    in_channels = int(cfg.get("in_channels", 3))
    stem_channels = int(cfg.get("stem_channels", 32))
    block_channels = cfg.get("block_channels", (64, 128, 256))
    block_channels = [int(c) for c in block_channels]
    embed_dim = int(cfg.get("embed_dim", block_channels[-1]))
    return ResidualFrameEncoder(
        in_channels=in_channels,
        stem_channels=stem_channels,
        block_channels=block_channels,
        embed_dim=embed_dim,
    )


def _build_convnext_frame_encoder(cfg: Mapping[str, Any]) -> FrameEncoderBase:
    variant = str(cfg.get("variant", "convnext_tiny"))
    projection_dim = cfg.get("projection_dim")
    projection_dim = int(projection_dim) if projection_dim is not None else None
    trainable = bool(cfg.get("trainable", False))
    return ConvNeXtFrameEncoder(variant=variant, projection_dim=projection_dim, trainable=trainable)


def _build_dinov3_stack_encoder(cfg: Mapping[str, Any]) -> FrameEncoderBase:
    backbone = str(cfg.get("backbone", "dinov3_vits16"))
    pretrained_path = cfg.get("pretrained_path")
    target_size = cfg.get("target_size", (224, 224))
    projection_dim = cfg.get("projection_dim")
    projection_dim = int(projection_dim) if projection_dim is not None else None
    normalize = bool(cfg.get("normalize", False))
    trainable = bool(cfg.get("trainable", False))
    return Dinov3StackFrameEncoder(
        backbone_name=backbone,
        pretrained_path=pretrained_path,
        target_size=target_size,
        projection_dim=projection_dim,
        normalize=normalize,
        trainable=trainable,
    )


def _build_dinov3_two_tower_encoder(cfg: Mapping[str, Any]) -> FrameEncoderBase:
    backbone = str(cfg.get("backbone", "dinov3_vits16"))
    pretrained_path = cfg.get("pretrained_path" )
    projection_dim = cfg.get("projection_dim")
    projection_dim = int(projection_dim) if projection_dim is not None else None
    normalize = bool(cfg.get("normalize", False))
    trainable = bool(cfg.get("trainable", False))
    return Dinov3TwoTowerFrameEncoder(
        backbone_name=backbone,
        pretrained_path=pretrained_path,
        projection_dim=projection_dim,
        normalize=normalize,
        trainable=trainable,
    )


FRAME_ENCODER_BUILDERS: Dict[str, callable] = {
    "conv": _build_conv_frame_encoder,
    "residual_cnn": _build_residual_frame_encoder,
    "convnext": _build_convnext_frame_encoder,
    "dinov3_stack": _build_dinov3_stack_encoder,
    "dinov3_two_tower": _build_dinov3_two_tower_encoder,
}


def _build_transformer_sequence_encoder(input_dim: int, sequence_length: int, cfg: Mapping[str, Any]) -> nn.Module:
    num_layers = int(cfg.get("num_layers", 4))
    num_heads = int(cfg.get("num_heads", 8))
    ff_dim = int(cfg.get("ff_dim", input_dim * 4))
    dropout = float(cfg.get("dropout", 0.1))
    activation = str(cfg.get("activation", "gelu"))
    return TransformerSequenceEncoder(
        input_dim,
        sequence_length,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
        activation=activation,
    )


def _build_lstm_sequence_encoder(input_dim: int, sequence_length: int, cfg: Mapping[str, Any]) -> nn.Module:
    hidden_dim = int(cfg.get("hidden_dim", input_dim))
    num_layers = int(cfg.get("num_layers", 2))
    dropout = float(cfg.get("dropout", 0.1))
    bidirectional = bool(cfg.get("bidirectional", False))
    return LSTMSequenceEncoder(
        input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    )


def _build_tcn_sequence_encoder(input_dim: int, sequence_length: int, cfg: Mapping[str, Any]) -> nn.Module:
    hidden_dim = int(cfg.get("hidden_dim", input_dim))
    num_layers = int(cfg.get("num_layers", 4))
    kernel_size = int(cfg.get("kernel_size", 3))
    dropout = float(cfg.get("dropout", 0.1))
    dilation_base = int(cfg.get("dilation_base", 2))
    causal = bool(cfg.get("causal", False))
    return TCNSequenceEncoder(
        input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        kernel_size=kernel_size,
        dropout=dropout,
        dilation_base=dilation_base,
        causal=causal,
    )


SEQUENCE_ENCODER_BUILDERS: Dict[str, callable] = {
    "transformer": _build_transformer_sequence_encoder,
    "lstm": _build_lstm_sequence_encoder,
    "tcn": _build_tcn_sequence_encoder,
}


def _build_prediction_head(input_dim: int, cfg: Mapping[str, Any]) -> nn.Module:
    hidden_layers = cfg.get("layers")
    if hidden_layers is None and "hidden_dim" in cfg:
        hidden_layers = [cfg["hidden_dim"]]
    dropout = float(cfg.get("dropout", 0.1))
    return PredictionHead(input_dim, hidden_layers=hidden_layers, dropout=dropout)


def _build_sequence_regressor(model_cfg: Mapping[str, Any]) -> SequenceRegressor:
    sequence_length = int(model_cfg.get("sequence_length", DEFAULT_SEQUENCE_LENGTH))
    frame_cfg = dict(model_cfg.get("frame_encoder", {}))
    frame_type = str(frame_cfg.pop("type", "conv")).lower()
    if frame_type not in FRAME_ENCODER_BUILDERS:
        raise ValueError(f"Unknown frame encoder '{frame_type}'. Options: {sorted(FRAME_ENCODER_BUILDERS)}")
    frame_encoder = FRAME_ENCODER_BUILDERS[frame_type](frame_cfg)

    temporal_cfg = dict(model_cfg.get("sequence_encoder", {}))
    if not temporal_cfg and "transformer" in model_cfg:
        temporal_cfg = dict(model_cfg["transformer"])
        temporal_cfg.setdefault("type", "transformer")
    temporal_type = str(temporal_cfg.pop("type", "transformer")).lower()
    if temporal_type not in SEQUENCE_ENCODER_BUILDERS:
        raise ValueError(
            f"Unknown sequence encoder '{temporal_type}'. Options: {sorted(SEQUENCE_ENCODER_BUILDERS)}"
        )
    temporal_encoder = SEQUENCE_ENCODER_BUILDERS[temporal_type](frame_encoder.output_dim, sequence_length, temporal_cfg)

    head_cfg = dict(model_cfg.get("head", {}))
    head = _build_prediction_head(getattr(temporal_encoder, "output_dim", frame_encoder.output_dim), head_cfg)
    summary_mode = model_cfg.get("summary", model_cfg.get("summary_mode", "last"))

    return SequenceRegressor(
        sequence_length=sequence_length,
        frame_encoder=frame_encoder,
        temporal_encoder=temporal_encoder,
        head=head,
        summary=summary_mode,
    )


def _build_frame_classifier(model_cfg: Mapping[str, Any]) -> FrameClassifier:
    frame_cfg = dict(model_cfg.get("frame_encoder", {}))
    frame_type = str(frame_cfg.pop("type", "conv")).lower()
    if frame_type not in FRAME_ENCODER_BUILDERS:
        raise ValueError(f"Unknown frame encoder '{frame_type}'. Options: {sorted(FRAME_ENCODER_BUILDERS)}")
    frame_encoder = FRAME_ENCODER_BUILDERS[frame_type](frame_cfg)

    head_cfg = dict(model_cfg.get("head", {}))
    hidden_layers = head_cfg.get("layers") or head_cfg.get("hidden_layers")
    dropout = float(head_cfg.get("dropout", 0.1))
    head = PredictionHead(frame_encoder.output_dim, hidden_layers=hidden_layers, dropout=dropout)
    return FrameClassifier(frame_encoder=frame_encoder, head=head)


MODEL_BUILDERS: Dict[str, callable] = {
    "sequence_regressor": _build_sequence_regressor,
    "frame_classifier": _build_frame_classifier,
}


def create_model(model_cfg: Mapping[str, Any]) -> nn.Module:
    if "name" not in model_cfg:
        raise KeyError("Model config must include a 'name'")
    name = str(model_cfg["name"]).lower()
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model type: {name}. Available: {sorted(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[name](model_cfg)


__all__ = [
    "create_model",
    "SequenceRegressor",
    "FrameClassifier",
    "FRAME_ENCODER_BUILDERS",
    "SEQUENCE_ENCODER_BUILDERS",
]
