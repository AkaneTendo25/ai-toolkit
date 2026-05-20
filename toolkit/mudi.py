"""MuDI (Multi-subject Decoupled Identification) Seg-Mix augmentation.

Reference: Jang et al., "Identity Decoupling for Multi-Subject Personalization
of Text-to-Image Models", NeurIPS 2024 (arXiv 2404.04243). Reference impl:
github.com/agwmon/MuDI (FLUX branch is the closest analog for our flow-matching
backbone).

Method summary
--------------
MuDI is a *data-centric* defense against identity leakage between LoRAs trained
on different subjects: instead of touching the LoRA architecture or the loss
function, it augments the training distribution. With probability ``prob`` the
dataloader composes a synthetic two-subject training image by cropping the
foreground of an anchor sample (subject A) and a randomly sampled peer
(subject B) by their SAM masks, placing them side by side on a neutral canvas,
and rewriting the caption to mention both triggers. The composite mask is
returned through ai-toolkit's existing ``mask_tensor`` channel so the trainer's
``mask_multiplier`` weights the loss toward foreground (a soft alpha floor of
``mask_min_value`` ≈ 0.3 reproduces the paper's soft-mask formula
``M_soft = M * (1 - α) + α``).

This module owns the augmentation primitives only — ``MuDIConfig``,
``extract_subject_id``, ``image_collage`` and friends. The integration into
``AiToolkitDataset`` (probability gate, peer sampling, caption merge) lives in
``toolkit/data_loader.py`` and uses these helpers.

Mean-shifted noise initialization (Jang et al. §3.2) is provided as
standalone helpers — ``compute_mean_latent`` / ``mean_shifted_noise`` /
``save_mean_latent`` / ``load_mean_latent``. They are not wired into any
inference pipeline because the FLUX branch authors did not validate the
trick for flow matching; integrate when there is empirical evidence that
Seg-Mix alone underperforms on this backbone.

Out of scope here:
* SAM / OWLv2 mask generation — offline, see ``scripts/mudi_sam_preprocess.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class MuDIConfig:
    enabled: bool = False
    prob: float = 0.3
    """Per-sample probability of triggering Seg-Mix. Authors recommend 0.3."""
    soft_alpha: float = 0.3
    """Background floor for the composite mask in [0, 1]. Reproduces
    ``M_soft = M * (1 - α) + α`` from the paper. 0 = hard mask, 1 = no mask."""
    subject_id_pattern: str = "parent_dir"
    """How to derive a subject id from a sample's file path.
        * ``parent_dir`` — DreamBooth-style ``dataset/SUBJECT/img_*.png``
        * ``filename_prefix:<sep>`` — characters before the first ``<sep>``
          in the basename (e.g. ``filename_prefix:_`` for ``subject_a_001.png``)
    """
    pair_trigger_format: str = "{a} and {b}"
    """Caption merge template. ``{a}`` is replaced by the anchor's caption
    and ``{b}`` by the peer's. The default reproduces the paper's
    ``"<a> and <b>"`` form."""
    layout_mode: str = "side_by_side"
    """``side_by_side`` (anchor left, peer right) or ``random_position``."""
    scale_jitter: float = 0.2
    """Random scale of each cropped subject in [1 - jitter, 1 + jitter]."""
    feather_blur_px: int = 8
    """Gaussian blur radius applied to the composite mask before returning,
    to soften visible SAM segmentation seams. Set 0 to disable."""
    mean_shift_alpha: float = 0.0
    """Mean-shifted noise initialization (paper §3.2). Mixes initial
    noise toward per-subject mean latent: noise' = alpha * mean[subject] +
    (1 - alpha) * noise. Default 0.0 = disabled. Paper sweeps 0.05–0.20."""
    mean_latents_path: str = ""
    """Path to the precomputed mean-latents .pt file (dict subject_id → latent).
    Generated offline by an external script (see paper §3.2). Required
    when mean_shift_alpha > 0."""

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["MuDIConfig"]:
        if data is None:
            return None
        if not data.get("enabled", False):
            return cls(enabled=False)
        cfg = cls(
            enabled=True,
            prob=float(data.get("prob", 0.3)),
            soft_alpha=float(data.get("soft_alpha", 0.3)),
            subject_id_pattern=str(data.get("subject_id_pattern", "parent_dir")),
            pair_trigger_format=str(data.get("pair_trigger_format", "{a} and {b}")),
            layout_mode=str(data.get("layout_mode", "side_by_side")),
            scale_jitter=float(data.get("scale_jitter", 0.2)),
            feather_blur_px=int(data.get("feather_blur_px", 8)),
            mean_shift_alpha=float(data.get("mean_shift_alpha", 0.0)),
            mean_latents_path=str(data.get("mean_latents_path", "")),
        )
        if not 0.0 <= cfg.prob <= 1.0:
            raise ValueError(f"MuDI prob must be in [0, 1], got {cfg.prob}")
        if not 0.0 <= cfg.soft_alpha <= 1.0:
            raise ValueError(f"MuDI soft_alpha must be in [0, 1], got {cfg.soft_alpha}")
        if not 0.0 <= cfg.mean_shift_alpha <= 1.0:
            raise ValueError(f"MuDI mean_shift_alpha must be in [0, 1], got {cfg.mean_shift_alpha}")
        if cfg.mean_shift_alpha > 0 and not cfg.mean_latents_path:
            raise ValueError(
                "MuDI mean_shift_alpha > 0 requires mean_latents_path; "
                "precompute mean latents offline first (see paper §3.2)."
            )
        if cfg.layout_mode not in ("side_by_side", "random_position"):
            raise ValueError(
                f"MuDI layout_mode must be 'side_by_side' or 'random_position', "
                f"got {cfg.layout_mode!r}"
            )
        # Probe pair_trigger_format with placeholder strings so unknown
        # placeholders ({c}, {x}, ...) crash at config-load time rather
        # than mid-training.
        try:
            cfg.pair_trigger_format.format(a="_a_", b="_b_")
        except (KeyError, IndexError) as e:
            raise ValueError(
                f"MuDI pair_trigger_format must reference only {{a}} and "
                f"{{b}}; got {cfg.pair_trigger_format!r} which raised "
                f"{type(e).__name__}: {e}"
            )
        return cfg


def extract_subject_id(file_path: str, pattern: str) -> str:
    """Derive a subject identifier from ``file_path`` using one of the
    supported patterns. Returns the empty string when the pattern matches no
    structure (callers can use this signal to skip Seg-Mix gracefully).
    """
    if pattern == "parent_dir":
        parent = os.path.basename(os.path.dirname(file_path))
        return parent
    if pattern.startswith("filename_prefix:"):
        sep = pattern.split(":", 1)[1]
        if not sep:
            raise ValueError(
                f"filename_prefix requires a non-empty separator after the "
                f"colon (e.g. 'filename_prefix:_'); got {pattern!r}. An "
                f"empty separator would lump every file into the same id."
            )
        base = os.path.basename(file_path)
        name, _ = os.path.splitext(base)
        head, _, _ = name.partition(sep)
        return head
    raise ValueError(f"Unknown subject_id_pattern: {pattern!r}")


def _bbox_from_mask(mask: torch.Tensor) -> Optional[Tuple[int, int, int, int]]:
    """Return ``(top, left, bottom, right)`` bounding box of the non-zero
    region of ``mask`` shape ``[1, H, W]``. ``None`` when mask is all zeros.
    """
    binary = mask.squeeze(0) > 0.0
    if not binary.any():
        return None
    rows = binary.any(dim=1)
    cols = binary.any(dim=0)
    top = int(rows.nonzero()[0].item())
    bottom = int(rows.nonzero()[-1].item()) + 1
    left = int(cols.nonzero()[0].item())
    right = int(cols.nonzero()[-1].item()) + 1
    return top, left, bottom, right


def _crop_to_bbox(
    image: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Crop ``image`` and ``mask`` to the mask's foreground bbox. Returns the
    full tensors unchanged when the mask has no foreground (defensive)."""
    bbox = _bbox_from_mask(mask)
    if bbox is None:
        return image, mask
    t, l, b, r = bbox
    return image[:, t:b, l:r], mask[:, t:b, l:r]


def _apply_scale_jitter(
    image: torch.Tensor, mask: torch.Tensor, jitter: float, generator: torch.Generator
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resize image+mask by a uniform random factor in [1-jitter, 1+jitter]."""
    if jitter <= 0.0:
        return image, mask
    factor = 1.0 + (torch.rand((), generator=generator).item() * 2.0 - 1.0) * jitter
    factor = max(0.1, factor)
    _, h, w = image.shape
    new_h = max(1, int(round(h * factor)))
    new_w = max(1, int(round(w * factor)))
    image = F.interpolate(image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
    mask = F.interpolate(mask.unsqueeze(0), size=(new_h, new_w), mode="nearest").squeeze(0)
    return image, mask


def _gaussian_blur1d(t: torch.Tensor, kernel_size: int, sigma: float, dim: int) -> torch.Tensor:
    if kernel_size <= 1:
        return t
    half = kernel_size // 2
    x = torch.arange(-half, half + 1, dtype=torch.float32, device=t.device)
    kernel = torch.exp(-(x * x) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    if dim == -1:
        kernel = kernel.view(1, 1, 1, -1)
        pad = (half, half, 0, 0)
    else:
        kernel = kernel.view(1, 1, -1, 1)
        pad = (0, 0, half, half)
    pad_mode = "replicate"
    t_pad = F.pad(t.unsqueeze(0), pad, mode=pad_mode)
    out = F.conv2d(t_pad, kernel, groups=1).squeeze(0)
    return out


def _feather_mask(mask: torch.Tensor, radius_px: int) -> torch.Tensor:
    """Soft-edge a binary [1, H, W] mask by Gaussian blur. ``radius_px == 0``
    disables blurring."""
    if radius_px <= 0:
        return mask
    kernel_size = max(3, 2 * radius_px + 1)
    sigma = max(1.0, radius_px / 2.0)
    out = _gaussian_blur1d(mask, kernel_size, sigma, dim=-1)
    out = _gaussian_blur1d(out, kernel_size, sigma, dim=-2)
    return out.clamp(0.0, 1.0)


def _composite(
    canvas: torch.Tensor,
    canvas_mask: torch.Tensor,
    patch: torch.Tensor,
    patch_mask: torch.Tensor,
    top: int,
    left: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Place ``patch`` (image+mask) onto ``canvas`` at offset (top, left).
    Patch is alpha-composited using its own mask; canvas_mask is OR'd with
    the patch mask in the placed region.
    """
    _, ch, cw = canvas.shape
    _, ph, pw = patch.shape
    bot = min(top + ph, ch)
    right = min(left + pw, cw)
    h = bot - top
    w = right - left
    if h <= 0 or w <= 0:
        return canvas, canvas_mask
    sub_patch = patch[:, :h, :w]
    sub_mask = patch_mask[:, :h, :w]
    region_canvas = canvas[:, top:bot, left:right]
    region_mask = canvas_mask[:, top:bot, left:right]
    new_region = sub_patch * sub_mask + region_canvas * (1.0 - sub_mask)
    canvas[:, top:bot, left:right] = new_region
    canvas_mask[:, top:bot, left:right] = torch.maximum(region_mask, sub_mask)
    return canvas, canvas_mask


def image_collage(
    image_a: torch.Tensor,
    mask_a: torch.Tensor,
    image_b: torch.Tensor,
    mask_b: torch.Tensor,
    cfg: MuDIConfig,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compose a two-subject Seg-Mix sample.

    Inputs:
        ``image_a``, ``image_b`` : ``[C, H, W]`` float tensors.
        ``mask_a``, ``mask_b``   : ``[1, H, W]`` mask tensors with the
            foreground non-zero. They do not need to be strictly binary —
            any positive value is treated as foreground.

    Output: ``(collage_image, collage_mask)`` with the **same H, W as
    image_a** (so bucketed dataloaders keep working) and the composite mask
    in ``[0, 1]``. The mask is later normalized into the soft range
    ``[soft_alpha, 1]`` by the caller via the existing ``mask_min_value``
    pipeline in ai-toolkit.
    """
    if generator is None:
        generator = torch.Generator(device="cpu")
    if image_a.shape[-2:] != mask_a.shape[-2:]:
        raise ValueError(
            f"image_collage: image_a spatial shape {tuple(image_a.shape[-2:])} "
            f"does not match mask_a spatial shape {tuple(mask_a.shape[-2:])}"
        )
    if image_b.shape[-2:] != mask_b.shape[-2:]:
        raise ValueError(
            f"image_collage: image_b spatial shape {tuple(image_b.shape[-2:])} "
            f"does not match mask_b spatial shape {tuple(mask_b.shape[-2:])}"
        )
    target_h, target_w = image_a.shape[-2:]
    canvas = torch.zeros_like(image_a)
    canvas_mask = torch.zeros_like(mask_a)

    crop_a, crop_mask_a = _crop_to_bbox(image_a, mask_a)
    crop_b, crop_mask_b = _crop_to_bbox(image_b, mask_b)
    crop_a, crop_mask_a = _apply_scale_jitter(
        crop_a, crop_mask_a, cfg.scale_jitter, generator
    )
    crop_b, crop_mask_b = _apply_scale_jitter(
        crop_b, crop_mask_b, cfg.scale_jitter, generator
    )

    # Side-by-side: each crop occupies up to half the canvas width. We rescale
    # to fit while preserving aspect ratio.
    def _fit(c_img, c_mask, max_h, max_w):
        _, h, w = c_img.shape
        scale = min(max_h / h, max_w / w, 1.0) if (h > 0 and w > 0) else 1.0
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        c_img = F.interpolate(c_img.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
        c_mask = F.interpolate(c_mask.unsqueeze(0), size=(new_h, new_w), mode="nearest").squeeze(0)
        return c_img, c_mask

    half_w = target_w // 2
    crop_a, crop_mask_a = _fit(crop_a, crop_mask_a, target_h, half_w)
    crop_b, crop_mask_b = _fit(crop_b, crop_mask_b, target_h, target_w - half_w)

    # If a crop is wider/taller than its allotted region (after _fit failed
    # to shrink it enough — happens e.g. when target_w is odd and the half
    # rounds down) we have to give up some of the placement freedom rather
    # than silently clip in _composite. We re-shrink the crop to fit so the
    # downstream _composite never has to crop pixels off the canvas edge.
    def _enforce_fit(c_img, c_mask, max_h, max_w):
        _, h, w = c_img.shape
        if h <= max_h and w <= max_w:
            return c_img, c_mask
        scale = min(max_h / max(h, 1), max_w / max(w, 1))
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        c_img = F.interpolate(c_img.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
        c_mask = F.interpolate(c_mask.unsqueeze(0), size=(new_h, new_w), mode="nearest").squeeze(0)
        return c_img, c_mask

    crop_a, crop_mask_a = _enforce_fit(crop_a, crop_mask_a, target_h, half_w)
    crop_b, crop_mask_b = _enforce_fit(crop_b, crop_mask_b, target_h, target_w - half_w)

    if cfg.layout_mode == "side_by_side":
        top_a = (target_h - crop_a.shape[-2]) // 2
        left_a = (half_w - crop_a.shape[-1]) // 2
        top_b = (target_h - crop_b.shape[-2]) // 2
        left_b = half_w + (target_w - half_w - crop_b.shape[-1]) // 2
    else:  # random_position — keep both fully on-canvas
        def _rand(low, high):
            # _enforce_fit guarantees crop fits; high >= low here, but the
            # max() guard is defensive against future refactors that might
            # shrink the canvas dimensions.
            high = max(low, high)
            if high <= low:
                return low
            return int(torch.randint(low, high + 1, (1,), generator=generator).item())

        top_a = _rand(0, target_h - crop_a.shape[-2])
        left_a = _rand(0, half_w - crop_a.shape[-1])
        top_b = _rand(0, target_h - crop_b.shape[-2])
        left_b = _rand(half_w, target_w - crop_b.shape[-1])

    canvas, canvas_mask = _composite(canvas, canvas_mask, crop_a, crop_mask_a, top_a, left_a)
    canvas, canvas_mask = _composite(canvas, canvas_mask, crop_b, crop_mask_b, top_b, left_b)

    canvas_mask = _feather_mask(canvas_mask, cfg.feather_blur_px)
    return canvas, canvas_mask


def make_pair_caption(caption_a: str, caption_b: str, fmt: str) -> str:
    """Merge two single-subject captions into a Seg-Mix prompt.

    The default ``fmt = "{a} and {b}"`` reproduces the paper's
    ``"<a> and <b>"`` form. Triggers / tokens already present in the
    individual captions are preserved.
    """
    return fmt.format(a=caption_a, b=caption_b)


# ----- Mean-shifted noise initialization (Jang et al. §3.2) ------------------
#
# Helpers only. The caller is responsible for VAE-encoding the subject's
# training images into latents and for replacing the initial noise of the
# inference pipeline with the result of ``mean_shifted_noise``. None of this
# is wired into the trainer or the diffusers pipeline yet — the flow-matching
# variant of MuDI was not validated by the original authors and we do not have
# empirical evidence that Seg-Mix alone underperforms on Qwen-Image-2512.


def compute_mean_latent(latents: List[torch.Tensor]) -> torch.Tensor:
    """Average a list of subject latents into a single mean latent.

    All latents must share spatial shape; this is the caller's job (the
    standard pattern is to encode every training image at the same
    resolution before calling). The mean is computed in float32 to limit
    accumulated rounding even when the inputs are bf16/fp16.
    """
    if not latents:
        raise ValueError("compute_mean_latent: empty latent list")
    ref_shape = latents[0].shape
    for i, t in enumerate(latents):
        if t.shape != ref_shape:
            raise ValueError(
                f"compute_mean_latent: latent {i} has shape {tuple(t.shape)}; "
                f"expected {tuple(ref_shape)}"
            )
    stacked = torch.stack([t.detach().to(torch.float32) for t in latents], dim=0)
    return stacked.mean(dim=0).to(latents[0].dtype)


def mean_shifted_noise(
    noise: torch.Tensor, mean_latent: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Bias an initial-noise tensor toward a precomputed subject mean latent.

    Returns ``alpha * mean_latent + (1 - alpha) * noise``. With ``alpha = 0``
    the call is a no-op; with ``alpha = 1`` the noise is replaced entirely
    by the mean. The paper sweeps small values (≈ 0.05–0.2) at inference.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"mean_shifted_noise: alpha must be in [0, 1], got {alpha}")
    if noise.shape != mean_latent.shape:
        raise ValueError(
            f"mean_shifted_noise: noise shape {tuple(noise.shape)} != "
            f"mean_latent shape {tuple(mean_latent.shape)}"
        )
    mean_cast = mean_latent.to(device=noise.device, dtype=noise.dtype)
    return alpha * mean_cast + (1.0 - alpha) * noise


def save_mean_latent(mean_latent: torch.Tensor, path: str) -> None:
    """Persist a mean latent to disk for reuse at inference."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save(mean_latent.detach().cpu(), path)


def load_mean_latent(path: str, device: Optional[str] = None) -> torch.Tensor:
    """Load a mean latent saved by ``save_mean_latent``."""
    tensor = torch.load(path, map_location="cpu")
    if device is not None:
        tensor = tensor.to(device)
    return tensor
