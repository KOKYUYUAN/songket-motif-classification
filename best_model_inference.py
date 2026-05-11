"""Unified inference utilities for all four Songket motif classification models.

Supports: AlexNet, VGG-19, ResNet-50, GoogLeNet/Inception V3
All models share the same 3-class head and ImageNet preprocessing pipeline.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

CLASS_NAMES: Sequence[str] = (
    "bunga_pecah_lapan",
    "pucuk_rebung",
    "tampuk_manggis",
)

NUM_CLASSES = len(CLASS_NAMES)

# Human-readable display names keyed by model ID
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "alexnet":    "AlexNet",
    "vgg19":      "VGG-19",
    "resnet50":   "ResNet-50",
    "googlenet":  "GoogLeNet / Inception V3",
}

# Default checkpoint filenames (placed next to this script)
_DEFAULT_CKPT_NAMES: Dict[str, str] = {
    "alexnet":   "songket_motif_alexnet_final.pth",
    "vgg19":     "songket_motif_vgg19_final.pth",
    "resnet50":  "songket_motif_resnet50_final.pth",
    "googlenet": "songket_motif_googlenet_final.pth",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PredictionItem:
    label: str
    probability: float


@dataclass
class PredictionResult:
    model_id: str
    predicted_label: str
    confidence: float
    inference_ms: float
    top_k: List[PredictionItem]
    error: str | None = field(default=None)

    @property
    def display_name(self) -> str:
        return MODEL_DISPLAY_NAMES.get(self.model_id, self.model_id)

    @property
    def predicted_label_pretty(self) -> str:
        return self.predicted_label.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Checkpoint path helpers
# ---------------------------------------------------------------------------

def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def default_model_path(model_id: str) -> str:
    """Return the expected .pth path for a given model ID."""
    filename = _DEFAULT_CKPT_NAMES.get(model_id)
    if filename is None:
        raise ValueError(f"Unknown model ID '{model_id}'. Choose from: {list(_DEFAULT_CKPT_NAMES)}")
    return os.path.join(_script_dir(), filename)


# ---------------------------------------------------------------------------
# Model builders  (backbone + replaced head)
# ---------------------------------------------------------------------------

def _build_alexnet(num_classes: int) -> nn.Module:
    m = models.alexnet(weights=None)
    m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
    return m


def _build_vgg19(num_classes: int) -> nn.Module:
    # Must match training: vgg19_bn (VGG19 with BatchNorm), NOT plain vgg19
    m = models.vgg19_bn(weights=None)
    m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
    return m


def _build_resnet50(num_classes: int) -> nn.Module:
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def _build_googlenet(num_classes: int) -> nn.Module:
    m = models.googlenet(weights=None, aux_logits=True)
    # Main head — matches training: model.fc = nn.Linear(...)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    # Auxiliary heads — training used .fc (not .fc2), so we match that exactly
    if getattr(m, "aux1", None) is not None and hasattr(m.aux1, "fc"):
        m.aux1.fc = nn.Linear(m.aux1.fc.in_features, num_classes)
    if getattr(m, "aux2", None) is not None and hasattr(m.aux2, "fc"):
        m.aux2.fc = nn.Linear(m.aux2.fc.in_features, num_classes)
    return m


_BUILDERS = {
    "alexnet":   _build_alexnet,
    "vgg19":     _build_vgg19,
    "resnet50":  _build_resnet50,
    "googlenet": _build_googlenet,
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_id: str,
    model_path: str | None = None,
    device: torch.device | None = None,
) -> tuple[nn.Module, torch.device]:
    """Load a single model checkpoint and return (model, device)."""
    chosen_device = device or default_device()
    chosen_path   = model_path or default_model_path(model_id)

    if not os.path.exists(chosen_path):
        raise FileNotFoundError(
            f"Checkpoint not found for '{model_id}': {chosen_path}"
        )

    builder = _BUILDERS.get(model_id)
    if builder is None:
        raise ValueError(f"No builder registered for model ID '{model_id}'.")

    model = builder(NUM_CLASSES)
    state = torch.load(chosen_path, map_location=chosen_device)
    model.load_state_dict(state)
    model.to(chosen_device)
    model.eval()
    return model, chosen_device


def load_all_models(
    model_paths: Dict[str, str] | None = None,
    device: torch.device | None = None,
) -> Dict[str, tuple[nn.Module, torch.device]]:
    """Load all four models. Returns {model_id: (model, device)}.

    Pass model_paths={'alexnet': '/path/to/file.pth', ...} to override defaults.
    Missing paths raise FileNotFoundError; errors are surfaced individually so
    that the UI can show partial results when one checkpoint is unavailable.
    """
    paths = model_paths or {}
    loaded: Dict[str, tuple[nn.Module, torch.device]] = {}
    for mid in _BUILDERS:
        path = paths.get(mid)  # None → uses default
        loaded[mid] = load_model(mid, model_path=path, device=device)
    return loaded


# ---------------------------------------------------------------------------
# Preprocessing  (shared ImageNet pipeline)
# ---------------------------------------------------------------------------

_PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    return _PREPROCESS(image.convert("RGB")).unsqueeze(0)


# ---------------------------------------------------------------------------
# Single-model inference
# ---------------------------------------------------------------------------

def classify_image(
    image: Image.Image,
    model: nn.Module,
    device: torch.device,
    model_id: str = "alexnet",
    top_k: int = 3,
) -> PredictionResult:
    batch = preprocess_image(image).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(batch)
        # GoogLeNet returns a named-tuple when aux_logits=True during eval;
        # extract the main logits in that case.
        logits = out.logits if hasattr(out, "logits") else out
        probs = torch.softmax(logits[0], dim=0)
    t1 = time.perf_counter()

    k = min(top_k, NUM_CLASSES)
    top_probs, top_idx = torch.topk(probs, k=k)

    top_items = [
        PredictionItem(label=CLASS_NAMES[i], probability=float(p))
        for p, i in zip(top_probs.cpu().tolist(), top_idx.cpu().tolist())
    ]
    best = top_items[0]

    return PredictionResult(
        model_id=model_id,
        predicted_label=best.label,
        confidence=best.probability,
        inference_ms=(t1 - t0) * 1000.0,
        top_k=top_items,
    )


# ---------------------------------------------------------------------------
# Multi-model inference  (run all four, return list)
# ---------------------------------------------------------------------------

def classify_all(
    image: Image.Image,
    models_dict: Dict[str, tuple[nn.Module, torch.device]],
    top_k: int = 3,
) -> List[PredictionResult]:
    """Run inference with every loaded model and return results in a fixed order."""
    order = ["alexnet", "vgg19", "resnet50", "googlenet"]
    results: List[PredictionResult] = []

    for mid in order:
        if mid not in models_dict:
            continue
        model, device = models_dict[mid]
        try:
            result = classify_image(image, model, device, model_id=mid, top_k=top_k)
        except Exception as exc:  # pragma: no cover
            result = PredictionResult(
                model_id=mid,
                predicted_label="error",
                confidence=0.0,
                inference_ms=0.0,
                top_k=[],
                error=str(exc),
            )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Confidence guidance  (shared helper)
# ---------------------------------------------------------------------------

def confidence_guidance(confidence: float) -> str:
    if confidence >= 0.90:
        return "High confidence — suitable for automatic tagging with periodic audits."
    if confidence >= 0.70:
        return "Moderate confidence — useful for decision support, keep a human review step."
    return "Low confidence — use as a suggestion only; prioritise manual verification."