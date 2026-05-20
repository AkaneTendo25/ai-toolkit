"""Orthogonal Adaptation (OrthA) initialization for LoRA layers.

Reference: Po et al., "Orthogonal Adaptation for Modular Customization of Diffusion
Models", CVPR 2024 (arXiv 2312.02432).

Method summary
--------------
Standard LoRA parameterizes weight delta as ΔW = A·B^T with A∈R^{out×r},
B∈R^{in×r} both trainable. When several adapters {A_i, B_i} are merged via
weighted sum Σ_i λ_i A_i B_i^T, residuals from concept j leak into concept i
because B_i^T B_j is generally non-zero.

OrthA fixes this by **freezing B_i to a row-subset of a shared orthonormal
matrix O ∈ R^{n×n}**: each concept independently samples ``rank`` distinct
indices and uses ``O[idxs]`` as its frozen down-projection. Different concepts
pick different subsets of orthonormal rows, so B_i^T B_j is exactly zero
whenever the index sets are disjoint, and otherwise restricted to the small
intersection. Only A is trained.

Notes on this implementation
----------------------------
- ``O`` is generated via QR of a Gaussian matrix (``torch.linalg.qr``). The
  master seed is shared across concepts (default 0) so every concept references
  the same canonical basis without having to communicate.
- Per-concept randomness lives in ``cfg.seed`` and selects which ``rank`` rows
  to use. With n=2048 and rank=32 the expected pairwise overlap between two
  concepts is ~0.5 indices — and even when overlap occurs, the chosen rows are
  still mutually orthonormal, so the multi-concept invariant holds.
- Bases are cached on disk as ``{basis_dir}/ortha_basis_{n}_seed{master_seed}.pt``
  so they are reused across runs and concepts.
- Conv2d layers are not supported: the frozen-row construction has no canonical
  meaning over a [out, in, kh, kw] weight tensor. Callers must fall back to the
  default Kaiming init for conv2d.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class OrthAConfig:
    enabled: bool = False
    seed: int = 0
    """Per-concept seed: selects the row-subset of the shared basis."""
    scale: float = 0.5
    """Multiplier applied to the frozen lora_down rows. Default 0.5 reproduces
    the reference HF Space implementation; for larger backbones a value in the
    range [0.25, 1.0] or 1/sqrt(rank) may be preferable and should be swept."""
    basis_dir: str = "./ortha_bases"
    master_seed: int = 0
    """Seed for the shared orthonormal basis. MUST match across all concepts
    that will later be merged together — otherwise B_i^T B_j is not zero."""

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["OrthAConfig"]:
        if data is None:
            return None
        if not data.get("enabled", False):
            return cls(enabled=False)
        cfg = cls(
            enabled=True,
            seed=int(data.get("seed", 0)),
            scale=float(data.get("scale", 0.5)),
            basis_dir=str(data.get("basis_dir", "./ortha_bases")),
            master_seed=int(data.get("master_seed", 0)),
        )
        # ``scale = 0`` would silently zero out lora_down for every wrapped
        # layer; the freeze then locks zero weights forever and the LoRA
        # behaves as "no adapter" without any error. Inf/NaN bypass naive
        # comparisons (``inf > 0`` is True, ``nan > 0`` is False so naive
        # ``cfg.scale <= 0`` lets NaN through) and would inject NaN/Inf
        # into the adapter on the first forward. Reject all of these.
        import math as _math
        if not (_math.isfinite(cfg.scale) and cfg.scale > 0):
            raise ValueError(
                f"OrthAConfig.scale must be a finite positive number; "
                f"got {cfg.scale!r}. A non-finite or non-positive scale "
                f"silently corrupts the adapter at init."
            )
        return cfg


_basis_memcache: Dict[str, torch.Tensor] = {}


def _basis_path(basis_dir: str, n: int, master_seed: int) -> str:
    return os.path.join(basis_dir, f"ortha_basis_{n}_seed{master_seed}.pt")


def get_or_make_basis(n: int, basis_dir: str, master_seed: int) -> torch.Tensor:
    """Return an [n, n] orthonormal matrix with deterministic seed.

    Generated lazily via QR of a Gaussian and cached both in-process and on
    disk. The basis is always returned in fp32 — callers cast as needed.
    """
    cache_key = f"{n}:{master_seed}"
    cached = _basis_memcache.get(cache_key)
    if cached is not None:
        return cached

    path = _basis_path(basis_dir, n, master_seed)
    if os.path.exists(path):
        basis = torch.load(path, map_location="cpu")
        if basis.shape != (n, n):
            raise RuntimeError(
                f"Cached OrthA basis at {path} has shape {tuple(basis.shape)}, "
                f"expected ({n}, {n}). Delete the stale file or change master_seed."
            )
        _basis_memcache[cache_key] = basis
        return basis

    gen = torch.Generator(device="cpu").manual_seed(master_seed)
    a = torch.randn(n, n, generator=gen, dtype=torch.float32)
    q, _ = torch.linalg.qr(a)
    # qr returns Q with orthonormal columns; the columns/rows of Q are an
    # orthonormal basis of R^n. We use rows downstream (see select_rows).

    os.makedirs(basis_dir, exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(q, tmp_path)
    os.replace(tmp_path, path)

    _basis_memcache[cache_key] = q
    return q


def select_rows(basis: torch.Tensor, rank: int, concept_seed: int) -> torch.Tensor:
    """Pick ``rank`` distinct rows from ``basis`` deterministically by seed.

    Rejects rank <= 0 (negative would slice ``randperm[:-1]`` and silently
    return the wrong subset; zero would produce an empty tensor that fails
    a downstream copy_). The index tensor is moved to ``basis.device`` so
    callers can pass either a CPU or a CUDA basis without hitting the
    "Expected all tensors to be on the same device" runtime error.
    """
    n = basis.shape[0]
    if rank <= 0:
        raise ValueError(
            f"OrthA: rank must be a positive integer; got {rank}."
        )
    if rank > n:
        raise ValueError(
            f"OrthA: rank={rank} exceeds basis size n={n}; cannot sample without replacement"
        )
    gen = torch.Generator(device="cpu").manual_seed(concept_seed)
    idx = torch.randperm(n, generator=gen)[:rank]
    return basis.index_select(0, idx.to(basis.device))


def apply_ortha_init(
    lora_down: torch.nn.Linear,
    lora_up: torch.nn.Module,
    rank: int,
    in_features: int,
    cfg: OrthAConfig,
) -> None:
    """In-place: overwrite lora_down with a frozen orthonormal row-subset.

    Preconditions:
        - lora_down is nn.Linear with weight shape [rank, in_features].
        - lora_up has been zero-initialized (so the initial residual is zero
          and the base model is unchanged at step 0). This is asserted.
    """
    if not isinstance(lora_down, torch.nn.Linear):
        raise TypeError(
            f"OrthA expects lora_down to be nn.Linear, got {type(lora_down).__name__}"
        )
    if rank > in_features:
        raise ValueError(
            f"OrthA: rank={rank} > in_features={in_features}; cannot init"
        )
    # Idempotency: applying twice within one Python process would silently
    # re-sample the row subset and overwrite the previously frozen basis.
    # We store the marker on lora_down for the in-process guard. Note that
    # this attribute is *not* persisted in state_dict; cross-process resume
    # legitimately re-runs apply_ortha_init at network construction time
    # before load_state_dict copies the saved weights back, so the second
    # call is correct in that scenario.
    if getattr(lora_down, "_ortha_applied", False):
        raise RuntimeError(
            "apply_ortha_init has already been applied to this lora_down "
            "module; calling it again would discard the existing frozen "
            "row subset."
        )

    up_weight = getattr(lora_up, "weight", None)
    if up_weight is not None:
        if not torch.all(up_weight == 0):
            raise AssertionError(
                "OrthA invariant violated: lora_up.weight must be zero at init "
                "so the initial residual is zero. Check that the standard LoRA "
                "init zeroed lora_up before apply_ortha_init runs."
            )

    basis = get_or_make_basis(in_features, cfg.basis_dir, cfg.master_seed)
    rows = select_rows(basis, rank, cfg.seed) * cfg.scale

    target_dtype = lora_down.weight.dtype
    target_device = lora_down.weight.device
    with torch.no_grad():
        lora_down.weight.copy_(rows.to(dtype=target_dtype, device=target_device))
    lora_down.weight.requires_grad_(False)
    lora_down._ortha_applied = True
