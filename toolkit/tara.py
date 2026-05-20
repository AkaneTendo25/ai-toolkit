"""Token-Aware LoRA (TARA) — config, controller, and helpers.

Reference: Peng et al., "Token-Aware LoRA for Composable Personalization",
AAAI 2026 (arXiv 2508.08812). Reference impl: github.com/YuqiPeng77/TARA
(SD1.5 / SDXL only — the MM-DiT port lives in this file plus the
TARALoRAModule subclass in toolkit/lora_special.py).

Method summary
--------------
Standard LoRA on a text-conditioned attention's K/V projections updates
**every** text-token position equally, so per-concept residuals leak across
tokens at composition time. TARA gates the LoRA delta along the token axis
to a binary mask M_i that is 1 only at the position(s) of the rare
identifier token V_i^*, then adds an alignment loss L_align that pulls
K(V_i^*) toward K(class_i) so that V_i^*'s attention map lands on the
class's image region.

This implementation
-------------------
* TARAConfig — YAML-facing dataclass.
* TARAController — process-wide singleton holding the active mask, the
  rare/class token positions for the current step, and a per-layer capture
  buffer for L_align. Singleton state is sufficient for single-concept
  training and single-concept inference. Composing N>1 TARA-LoRAs at
  inference time requires per-network state and is left as future work.
* tokenize_and_locate — handles multi-subword V_i^* by including every
  subword index in the mask.
* The actual LoRAModule subclass that hooks the mask / TAL-capture into the
  ai-toolkit forward path lives in toolkit/lora_special.py (TARALoRAModule),
  so this file has no dependency on lora_special and can be imported there
  without a circular import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class TARAConfig:
    enabled: bool = False
    rare_token: str = ""
    """The identifier token (e.g. "xlo") whose K/V projections carry the
    concept. Must be present in every training caption."""
    class_token: str = ""
    """The class anchor (e.g. "man", "dog") that L_align pulls V* toward.
    Must be present in every training caption near rare_token."""
    lambda_align: float = 1.0
    target_text_modules: List[str] = field(
        default_factory=lambda: ["add_k_proj", "add_v_proj"]
    )
    """Substrings used by LoRASpecialNetwork.only_if_contains to restrict
    LoRA wrapping to text-stream K/V projections. The defaults match
    Qwen-Image's QwenDoubleStreamAttnProcessor2_0."""
    prompt_template: Optional[str] = None
    """Optional chat-format template (with ``{}`` for the user prompt) used
    by the pipeline before encoding. Qwen-Image wraps every prompt in a
    system/user template; the system prefix is then dropped before the
    transformer sees the embeds. ``tokenize_and_locate*`` needs to know
    about this so the per-prompt mask matches the runtime
    ``encoder_hidden_states`` token-axis length. Leave ``None`` for
    pipelines without a wrapper (SD1.5 / SDXL). If unset and the pipeline
    exposes ``prompt_template_encode``, SDTrainer auto-populates this at
    init time."""
    template_drop_idx: int = 0
    """Number of leading tokens dropped from the wrapped encoding. Must be
    paired with ``prompt_template`` — both unset or both set. Auto-populated
    from ``pipeline.prompt_template_encode_start_idx`` when available."""

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["TARAConfig"]:
        if data is None:
            return None
        if not data.get("enabled", False):
            return cls(enabled=False)
        cfg = cls(
            enabled=True,
            rare_token=str(data.get("rare_token", "")),
            class_token=str(data.get("class_token", "")),
            lambda_align=float(data.get("lambda_align", 1.0)),
        )
        custom_modules = data.get("target_text_modules")
        if custom_modules:
            cfg.target_text_modules = list(custom_modules)
        if "prompt_template" in data and data["prompt_template"] is not None:
            cfg.prompt_template = str(data["prompt_template"])
        if "template_drop_idx" in data and data["template_drop_idx"] is not None:
            cfg.template_drop_idx = int(data["template_drop_idx"])
        if (cfg.prompt_template is None) ^ (cfg.template_drop_idx == 0):
            raise ValueError(
                "TARAConfig: prompt_template and template_drop_idx must be "
                "specified together (or both omitted)."
            )
        if not cfg.rare_token:
            raise ValueError("TARAConfig: rare_token must be a non-empty string when enabled")
        if not cfg.class_token:
            raise ValueError("TARAConfig: class_token must be a non-empty string when enabled")
        # A negative lambda would push K(V*) *away* from K(class), which is
        # the opposite of what TAL is supposed to do. NaN bypasses naive
        # ``< 0`` checks (every NaN comparison is False) and would inject
        # NaN straight into total_loss; Inf would explode the gradient on
        # the first step. Require a finite, non-negative weight.
        import math as _math
        if not (_math.isfinite(cfg.lambda_align) and cfg.lambda_align >= 0):
            raise ValueError(
                f"TARAConfig: lambda_align must be a finite non-negative "
                f"number; got {cfg.lambda_align!r}. NaN/Inf would silently "
                f"corrupt the training loss."
            )
        if cfg.rare_token == cfg.class_token:
            raise ValueError(
                f"TARAConfig: rare_token and class_token must differ; both "
                f"are {cfg.rare_token!r}, which makes L_align trivially zero."
            )
        return cfg


class TARAController:
    """Process-wide singleton for the active token mask and TAL capture buffer.

    Single-concept training and single-concept inference set the mask once per
    step (or once per generation) and let every TARALoRAModule consult the
    same state. Multi-concept inference (composing N adapters in one forward)
    is *not* supported by this controller — it would require per-network
    masks; mark as TBD when that becomes a requirement.

    DDP / multi-GPU note. Each process owns its own TARAController state;
    masks and TAL buffers do not synchronize across ranks via NCCL. With a
    deterministic batch sampler all ranks happen to compute the same per-step
    mask out of identical captions, so per-step gradient on TAL is consistent.
    With non-deterministic samplers ranks may build different masks; the
    DDP-averaged gradient still includes TAL contributions but each is local
    to its rank's batch. Document as a known limitation when adopting DDP.
    """

    _mask: Optional[torch.Tensor] = None
    """Either ``[T]`` (broadcast across the batch) or ``[B, T]`` (per-sample).
    The set_mask helper accepts both shapes."""
    _rare_indices: Optional[torch.Tensor] = None
    """For batched per-sample mode the tensor is ``[B, k]`` (each row holds
    the V* subword indices of the corresponding sample). For broadcast mode
    it stays ``[k]``. Padded with -1 when samples have different subword
    counts; consumers ignore -1."""
    _class_indices: Optional[torch.Tensor] = None
    _seq_len: Optional[int] = None
    _capture_active: bool = False
    _tal_buffer: Dict[str, Dict[str, torch.Tensor]] = {}
    _mask_apply_count: int = 0
    """Counts how many TARALoRAModule.forward calls successfully applied the
    mask (shape matched). Diagnostic only — used by smoke tests / inference
    instrumentation to verify the mask is actually firing on text-stream
    layers and not silently passing through. Reset on ``reset()`` and
    ``set_mask(...)`` so each generation run starts from zero."""
    _mask_passthrough_count: int = 0
    """Counts pass-through events (shape mismatch). Should match the number
    of image-stream-layer × forward calls, plus any unconditional-CFG
    forwards where the negative prompt has a different token-axis."""

    @classmethod
    def reset(cls) -> None:
        cls._mask = None
        cls._rare_indices = None
        cls._class_indices = None
        cls._seq_len = None
        cls._capture_active = False
        cls._tal_buffer = {}
        cls._mask_apply_count = 0
        cls._mask_passthrough_count = 0

    @classmethod
    def get_mask_apply_count(cls) -> int:
        return cls._mask_apply_count

    @classmethod
    def get_mask_passthrough_count(cls) -> int:
        return cls._mask_passthrough_count

    @classmethod
    def increment_mask_apply(cls) -> None:
        cls._mask_apply_count += 1

    @classmethod
    def increment_mask_passthrough(cls) -> None:
        cls._mask_passthrough_count += 1

    @classmethod
    def set_token_indices(
        cls,
        rare_indices: torch.Tensor,
        class_indices: torch.Tensor,
        seq_len: int,
    ) -> None:
        cls._rare_indices = rare_indices.to(torch.long).clone()
        cls._class_indices = class_indices.to(torch.long).clone()
        cls._seq_len = int(seq_len)

    @classmethod
    def set_mask(
        cls,
        mask: torch.Tensor,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Store the active mask. Accepts ``[T]`` or ``[B, T]`` and pre-casts
        to the target device/dtype so the per-forward hot path does not pay
        a dtype/device copy on every step.
        """
        if mask.dim() not in (1, 2):
            raise ValueError(
                f"TARAController.set_mask: expected 1D or 2D tensor, got shape {tuple(mask.shape)}"
            )
        m = mask.detach().clone()
        if device is not None:
            m = m.to(device=device)
        if dtype is not None:
            m = m.to(dtype=dtype)
        cls._mask = m
        cls._seq_len = int(m.shape[-1])
        # Reset counters on every new mask so each step / generation run
        # measures its own apply/passthrough ratio independently.
        cls._mask_apply_count = 0
        cls._mask_passthrough_count = 0

    @classmethod
    def get_mask(cls) -> Optional[torch.Tensor]:
        return cls._mask

    @classmethod
    def get_seq_len(cls) -> Optional[int]:
        return cls._seq_len

    @classmethod
    def get_rare_indices(cls) -> Optional[torch.Tensor]:
        return cls._rare_indices

    @classmethod
    def get_class_indices(cls) -> Optional[torch.Tensor]:
        return cls._class_indices

    @classmethod
    def enable_capture(cls, on: bool) -> None:
        cls._capture_active = bool(on)

    @classmethod
    def is_capture_active(cls) -> bool:
        return cls._capture_active

    @classmethod
    def clear_buffer(cls) -> None:
        cls._tal_buffer = {}

    @classmethod
    def record_layer(
        cls,
        layer_name: str,
        base_class: torch.Tensor,
        full_v: torch.Tensor,
        valid_class: Optional[torch.Tensor] = None,
        valid_v: Optional[torch.Tensor] = None,
    ) -> None:
        """Store per-layer K-vector slices for L_align. ``valid_class`` and
        ``valid_v`` (each ``[B, k]``) flag positions that came from real
        subwords vs -1 padding; compute_tal_loss uses them to take a masked
        mean so padded slots do not bias the loss downward.

        Idempotent per (layer, step). Gradient checkpointing replays the
        forward during backward, which would call this method a second time
        with freshly recomputed tensors. Saving both versions doubles the
        effective layer count (and PyTorch flags the saved-tensor mismatch
        with ``CheckpointError: A different number of tensors was saved
        during the original forward and recomputation``). Keep the first
        capture and ignore the replay.
        """
        if layer_name in cls._tal_buffer:
            return
        cls._tal_buffer[layer_name] = {
            "base_class": base_class,
            "full_v": full_v,
            "valid_class": valid_class,
            "valid_v": valid_v,
        }

    @classmethod
    def compute_tal_loss(cls) -> torch.Tensor:
        """Mean over text-stream layers of MSE between K_base[class_idx] and
        (K_base + ΔK)[V*_idx]. Returns zero scalar (no grad) if no layers
        have been captured this step.

        For multi-subword V* or class tokens we average the K-vectors across
        the subword axis first (weighted by the valid mask so -1 padded slots
        do not bias the average), then take MSE between the two averages.
        At single-subword (k=1, no padding) this is identical to paper Eq. 5;
        at k>1 it is the natural generalization paper does not specify.
        """
        if not cls._tal_buffer:
            return torch.zeros((), dtype=torch.float32)

        def _masked_mean(t: torch.Tensor, valid: Optional[torch.Tensor]) -> torch.Tensor:
            # t: [B, k, D]; valid: [B, k] or None
            if valid is None:
                return t.mean(dim=1)
            v = valid.to(t.device).float().unsqueeze(-1)  # [B, k, 1]
            denom = v.sum(dim=1).clamp(min=1.0)            # [B, 1]
            return (t * v).sum(dim=1) / denom              # [B, D]

        per_layer = []
        for entry in cls._tal_buffer.values():
            base_class = entry["base_class"].float()  # [B, k_class, D]
            full_v = entry["full_v"].float()          # [B, k_v, D]
            valid_class = entry.get("valid_class")
            valid_v = entry.get("valid_v")
            base_class_avg = _masked_mean(base_class, valid_class)  # [B, D]
            full_v_avg = _masked_mean(full_v, valid_v)              # [B, D]
            per_layer.append(((base_class_avg - full_v_avg) ** 2).mean())
        loss = torch.stack(per_layer).mean()
        return loss


def _find_subword_positions(
    token_ids: List[int],
    needle_ids: List[int],
) -> List[int]:
    """Return positions of the contiguous needle_ids subsequence inside
    token_ids. If the needle appears multiple times, returns the first match
    (training captions are expected to mention the rare/class token once).
    """
    if not needle_ids:
        return []
    n, m = len(token_ids), len(needle_ids)
    for i in range(n - m + 1):
        if token_ids[i : i + m] == needle_ids:
            return list(range(i, i + m))
    return []


def _locate_via_offsets(
    tokenizer,
    prompt: str,
    word: str,
) -> Optional[List[int]]:
    """BPE-aware word location via offset_mapping.

    Returns a list of token positions whose char ranges overlap the first
    occurrence of ``word`` in ``prompt``, or ``None`` if the tokenizer does
    not expose offset mapping (caller falls back to the id-subsequence
    matcher). Uses Python ``str.find`` rather than regex so punctuation in
    ``word`` does not need escaping.

    Why this is needed: BPE tokenizers (Qwen2-VL, GPT-style) emit a
    different token id for a word at the start of a string ("Smith" →
    "Smi"+"th") than for the same word mid-string after a space (" Smith"
    → " Sm"+"ith"). The naive id-subsequence search trained the model
    to recognize the standalone form but never finds the in-context form,
    which is the form that actually appears in real captions.
    """
    char_pos = prompt.find(word)
    if char_pos < 0:
        return None
    char_end = char_pos + len(word)
    try:
        encoded = tokenizer(
            prompt,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
    except (TypeError, ValueError, NotImplementedError):
        # Slow / Python-side tokenizers raise NotImplementedError on
        # return_offsets_mapping; some custom wrappers use TypeError or
        # silently drop the kwarg. In all cases fall back to the
        # leading-space probe.
        return None
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        return None
    matches: List[int] = []
    for tok_idx, span in enumerate(offsets):
        # span may be a tuple or a list/None for special tokens.
        if span is None:
            continue
        try:
            tok_start, tok_end = int(span[0]), int(span[1])
        except (TypeError, IndexError):
            continue
        # Special tokens emit (0, 0) — skip them.
        if tok_start == tok_end == 0:
            continue
        # Half-open interval overlap: [tok_start, tok_end) ∩ [char_pos, char_end) ≠ ∅
        if tok_start < char_end and tok_end > char_pos:
            matches.append(tok_idx)
    return matches if matches else None


def tokenize_and_locate(
    tokenizer,
    prompt: str,
    rare_token: str,
    class_token: str,
    prompt_template: Optional[str] = None,
    template_drop_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize ``prompt`` and locate the rare and class tokens within the
    resulting id sequence. Returns ``(rare_indices, class_indices, mask)``.

    The mask has shape ``[seq_len]`` (no batch dimension) with 1.0 at every
    subword position belonging to ``rare_token`` and 0.0 elsewhere. Both
    ``rare_indices`` and ``class_indices`` are LongTensors that may have
    length > 1 when the tokenizer splits the corresponding word.

    Template-aware mode (Qwen-Image, FLUX, etc.). When ``prompt_template`` is
    given (a Python format string with ``{}`` for the user prompt) and
    ``template_drop_idx > 0``, the runtime ``encoder_hidden_states`` shape is
    ``len(encode(template.format(prompt))) - template_drop_idx`` because the
    pipeline drops the system/user prefix. The mask is sized to that
    effective length; rare/class positions stay valid because the dropped
    prefix lives at the head of the full encoding so the kept slice begins
    exactly at the first user-prompt token.

    Raises if either token cannot be located — the caller decides whether to
    skip the sample or fail loudly.
    """
    full_ids = tokenizer.encode(prompt, add_special_tokens=True)
    # First try BPE-aware char-offset location (handles leading-space variants
    # for Qwen2-VL / Qwen2 / GPT-style BPE). Fall back to id-subsequence
    # matching for tokenizers that do not expose offset_mapping.
    rare_pos = _locate_via_offsets(tokenizer, prompt, rare_token)
    if rare_pos is None:
        rare_ids = tokenizer.encode(rare_token, add_special_tokens=False)
        rare_pos = _find_subword_positions(full_ids, rare_ids)
        if not rare_pos:
            # Last-resort probe with leading space — covers BPE merges.
            rare_ids_lead = tokenizer.encode(" " + rare_token, add_special_tokens=False)
            rare_pos = _find_subword_positions(full_ids, rare_ids_lead)
    class_pos = _locate_via_offsets(tokenizer, prompt, class_token)
    if class_pos is None:
        class_ids = tokenizer.encode(class_token, add_special_tokens=False)
        class_pos = _find_subword_positions(full_ids, class_ids)
        if not class_pos:
            class_ids_lead = tokenizer.encode(" " + class_token, add_special_tokens=False)
            class_pos = _find_subword_positions(full_ids, class_ids_lead)

    if not rare_pos:
        raise ValueError(
            f"TARA tokenize_and_locate: rare_token {rare_token!r} not found in prompt {prompt!r}"
        )
    if not class_pos:
        raise ValueError(
            f"TARA tokenize_and_locate: class_token {class_token!r} not found in prompt {prompt!r}"
        )

    seq_len = _compute_runtime_seq_len(
        tokenizer, prompt, full_ids, prompt_template, template_drop_idx
    )
    mask = torch.zeros(seq_len, dtype=torch.float32)
    rare_pos_clipped = [p for p in rare_pos if p < seq_len]
    if rare_pos_clipped:
        mask[rare_pos_clipped] = 1.0
    class_pos_clipped = [p for p in class_pos if p < seq_len]

    return (
        torch.tensor(rare_pos_clipped or rare_pos, dtype=torch.long),
        torch.tensor(class_pos_clipped or class_pos, dtype=torch.long),
        mask,
    )


def _compute_runtime_seq_len(
    tokenizer,
    prompt: str,
    raw_full_ids: List[int],
    prompt_template: Optional[str],
    template_drop_idx: int,
) -> int:
    """Compute the runtime ``encoder_hidden_states`` token-axis length.

    For models without a chat-template wrapper (SD1.5, SDXL — the helper's
    historical use case) the raw encoded length already matches what the
    text encoder produces (CLIP / T5 do their own padding outside this
    helper, but the per-prompt mask broadcasts with ``repeat_interleave``
    in TARALoRAModule.forward). We return the raw length as the legacy
    behavior.

    For Qwen-Image (and any pipeline that wraps the user prompt in a
    chat-format template and then drops the system prefix), the runtime
    length is ``len(encode(template.format(prompt))) - drop_idx``. The
    raw-prompt token positions stay valid because the dropped prefix sits
    at the head of the full encoding — the kept slice starts at the first
    user-prompt token.
    """
    if prompt_template is None or template_drop_idx <= 0:
        return len(raw_full_ids)
    wrapped = prompt_template.format(prompt)
    wrapped_ids = tokenizer.encode(wrapped, add_special_tokens=True)
    runtime_len = len(wrapped_ids) - template_drop_idx
    if runtime_len <= 0:
        # Pathological: drop_idx larger than full encoding. Fall back to
        # raw length so the caller still gets a usable mask.
        return len(raw_full_ids)
    return runtime_len


def make_token_mask(seq_len: int, rare_indices: torch.Tensor) -> torch.Tensor:
    """Construct a [seq_len] binary mask with 1.0 at every rare_index."""
    mask = torch.zeros(seq_len, dtype=torch.float32)
    mask[rare_indices.to(torch.long)] = 1.0
    return mask


def tokenize_and_locate_batch(
    tokenizer,
    prompts: List[str],
    rare_token: str,
    class_token: str,
    prompt_template: Optional[str] = None,
    template_drop_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized variant for a batch of prompts. Returns

    * ``rare_indices`` LongTensor ``[B, max_rare_subwords]``, padded with -1
    * ``class_indices`` LongTensor ``[B, max_class_subwords]``, padded with -1
    * ``mask`` FloatTensor ``[B, max_seq_len]``, padded with 0.0

    Padding is on the right. Different prompts are padded up to the longest
    encoded length in the batch so per-sample masks share the same T_txt and
    can be applied as a single broadcast multiply on ``[B, T_txt, D]``.
    """
    if not prompts:
        raise ValueError("tokenize_and_locate_batch: prompts must be non-empty")

    per_sample_runtime_len: List[int] = []
    per_sample_rare: List[List[int]] = []
    per_sample_class: List[List[int]] = []
    for p in prompts:
        full_ids = tokenizer.encode(p, add_special_tokens=True)
        rare_pos = _locate_via_offsets(tokenizer, p, rare_token)
        if rare_pos is None:
            rare_ids = tokenizer.encode(rare_token, add_special_tokens=False)
            rare_pos = _find_subword_positions(full_ids, rare_ids)
            if not rare_pos:
                rare_ids_lead = tokenizer.encode(" " + rare_token, add_special_tokens=False)
                rare_pos = _find_subword_positions(full_ids, rare_ids_lead)
        class_pos = _locate_via_offsets(tokenizer, p, class_token)
        if class_pos is None:
            class_ids = tokenizer.encode(class_token, add_special_tokens=False)
            class_pos = _find_subword_positions(full_ids, class_ids)
            if not class_pos:
                class_ids_lead = tokenizer.encode(" " + class_token, add_special_tokens=False)
                class_pos = _find_subword_positions(full_ids, class_ids_lead)
        if not rare_pos:
            raise ValueError(
                f"TARA tokenize_and_locate_batch: rare_token {rare_token!r} not found in prompt {p!r}"
            )
        if not class_pos:
            raise ValueError(
                f"TARA tokenize_and_locate_batch: class_token {class_token!r} not found in prompt {p!r}"
            )
        runtime_len = _compute_runtime_seq_len(
            tokenizer, p, full_ids, prompt_template, template_drop_idx
        )
        # Drop positions that fall outside the runtime slice (cannot happen
        # under the documented Qwen-Image layout but guards against
        # tokenizer drift / unexpected templates).
        rare_pos = [r for r in rare_pos if r < runtime_len]
        class_pos = [c for c in class_pos if c < runtime_len]
        if not rare_pos:
            raise ValueError(
                f"TARA tokenize_and_locate_batch: rare_token {rare_token!r} positions "
                f"are outside the runtime slice (template_drop_idx may be too large) "
                f"for prompt {p!r}"
            )
        if not class_pos:
            raise ValueError(
                f"TARA tokenize_and_locate_batch: class_token {class_token!r} positions "
                f"are outside the runtime slice for prompt {p!r}"
            )
        per_sample_runtime_len.append(runtime_len)
        per_sample_rare.append(rare_pos)
        per_sample_class.append(class_pos)

    max_seq = max(per_sample_runtime_len)
    max_rare = max(len(s) for s in per_sample_rare)
    max_class = max(len(s) for s in per_sample_class)
    B = len(prompts)

    mask = torch.zeros(B, max_seq, dtype=torch.float32)
    rare_idx = torch.full((B, max_rare), -1, dtype=torch.long)
    class_idx = torch.full((B, max_class), -1, dtype=torch.long)
    for i, (rare_pos, class_pos) in enumerate(zip(per_sample_rare, per_sample_class)):
        for r in rare_pos:
            mask[i, r] = 1.0
        rare_idx[i, : len(rare_pos)] = torch.tensor(rare_pos, dtype=torch.long)
        class_idx[i, : len(class_pos)] = torch.tensor(class_pos, dtype=torch.long)
    return rare_idx, class_idx, mask
