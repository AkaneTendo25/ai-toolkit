from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch


@dataclass
class OrthoLoRAParamEntry:
    name: str
    role: str
    param: torch.nn.Parameter


@dataclass
class OrthoLoRAWindowState:
    entries: List[OrthoLoRAParamEntry]
    window_start_grads: List[Optional[torch.Tensor]]
    task_labels: List[str]
    task_grads: List[List[Optional[torch.Tensor]]]


def _clone_optional_tensor(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    return value.detach().clone()


def _add_optional_tensors(
    left: Optional[torch.Tensor], right: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    if left is None:
        return _clone_optional_tensor(right)
    if right is None:
        return _clone_optional_tensor(left)
    return left + right.to(device=left.device, dtype=left.dtype)


def _sub_optional_tensors(
    left: Optional[torch.Tensor], right: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    if left is None and right is None:
        return None
    if left is None:
        assert right is not None
        return -right.detach().clone()
    if right is None:
        return left.detach().clone()
    return left - right.to(device=left.device, dtype=left.dtype)


def _sum_optional_tensors(
    tensors: Sequence[Optional[torch.Tensor]],
) -> Optional[torch.Tensor]:
    total: Optional[torch.Tensor] = None
    for tensor in tensors:
        total = _add_optional_tensors(total, tensor)
    return total


def _gradient_dot(
    left: Sequence[Optional[torch.Tensor]],
    right: Sequence[Optional[torch.Tensor]],
) -> float:
    total = 0.0
    for left_tensor, right_tensor in zip(left, right):
        if left_tensor is None or right_tensor is None:
            continue
        total += float(
            torch.sum(
                left_tensor.detach().to(torch.float32)
                * right_tensor.detach().to(torch.float32)
            ).item()
        )
    return total


def _gradient_norm_sq(grads: Sequence[Optional[torch.Tensor]]) -> float:
    return max(_gradient_dot(grads, grads), 0.0)


def project_conflicting_gradients(
    task_grads: Sequence[Sequence[Optional[torch.Tensor]]],
    shuffle_order: bool = True,
) -> tuple[List[List[Optional[torch.Tensor]]], int]:
    projected: List[List[Optional[torch.Tensor]]] = [
        [_clone_optional_tensor(tensor) for tensor in grads] for grads in task_grads
    ]
    if len(projected) <= 1:
        return projected, 0

    order = list(range(len(projected)))
    if shuffle_order:
        permutation = torch.randperm(len(order)).tolist()
        order = [order[idx] for idx in permutation]

    conflict_count = 0
    for i in order:
        for j in order:
            if i == j:
                continue
            dot = _gradient_dot(projected[i], projected[j])
            if dot >= 0.0:
                continue
            norm_sq = _gradient_norm_sq(projected[j])
            if norm_sq <= 1e-12:
                continue
            coeff = dot / norm_sq
            updated: List[Optional[torch.Tensor]] = []
            for current, reference in zip(projected[i], projected[j]):
                if current is None:
                    updated.append(None)
                    continue
                if reference is None:
                    updated.append(current)
                    continue
                updated.append(
                    current
                    - reference.to(device=current.device, dtype=current.dtype) * coeff
                )
            projected[i] = updated
            conflict_count += 1

    return projected, conflict_count


class OrthoLoRAHelper:
    def __init__(self, network_kwargs: Optional[Dict[str, Any]] = None):
        config = network_kwargs or {}
        nested_config = config.get("ortholora", {})
        if not isinstance(nested_config, dict):
            nested_config = {}

        self.enabled = bool(
            config.get("enable_ortholora", nested_config.get("enabled", False))
        )
        self.log_metrics = bool(
            config.get("ortholora_log_metrics", nested_config.get("log_metrics", True))
        )

        self._cached_network_id: Optional[int] = None
        self._cached_entries: List[OrthoLoRAParamEntry] = []
        self._window_state: Optional[OrthoLoRAWindowState] = None

    def reset_window(self) -> None:
        self._window_state = None

    def _discover_lora_entries(self, network: Any) -> List[OrthoLoRAParamEntry]:
        network_id = id(network)
        if self._cached_network_id == network_id:
            return self._cached_entries

        entries: List[OrthoLoRAParamEntry] = []
        seen_param_ids: set[int] = set()
        for name, param in network.named_parameters():
            if not getattr(param, "requires_grad", False):
                continue
            lowered = name.lower()
            role: Optional[str] = None
            if "lora_down" in lowered and lowered.endswith(".weight"):
                role = "lora_down"
            elif "lora_up" in lowered and lowered.endswith(".weight"):
                role = "lora_up"
            if role is None:
                continue
            param_id = id(param)
            if param_id in seen_param_ids:
                continue
            seen_param_ids.add(param_id)
            entries.append(OrthoLoRAParamEntry(name=name, role=role, param=param))

        self._cached_network_id = network_id
        self._cached_entries = entries
        return entries

    def _extract_task_label(self, batch: Any) -> str:
        if batch is None or getattr(batch, "file_items", None) is None:
            raise ValueError("OrthoLORA requires a batch with file_items")

        labels = [getattr(item, "ortholora_task_index", None) for item in batch.file_items]
        if any(label is None for label in labels):
            raise ValueError("OrthoLORA requires ortholora_task_index on every batch item")
        unique_labels = sorted({int(label) for label in labels})
        if len(unique_labels) != 1:
            raise ValueError(
                "OrthoLORA requires each microbatch to contain exactly one task dataset"
            )
        return str(unique_labels[0])

    def _ensure_window_state(self, network: Any) -> OrthoLoRAWindowState:
        entries = self._discover_lora_entries(network)
        if not entries:
            raise ValueError("OrthoLORA could not find any trainable LoRA A/B weights")

        if self._window_state is None:
            self._window_state = OrthoLoRAWindowState(
                entries=entries,
                window_start_grads=[
                    _clone_optional_tensor(entry.param.grad) for entry in entries
                ],
                task_labels=[],
                task_grads=[],
            )
            return self._window_state

        if len(self._window_state.entries) != len(entries):
            raise ValueError("OrthoLORA saw a different set of LoRA params mid-window")
        return self._window_state

    def accumulate_task_gradient(
        self,
        network: Any,
        batch: Any,
        per_sample_loss: torch.Tensor,
        scalar_multiplier: Optional[torch.Tensor] = None,
    ) -> None:
        if not self.enabled or network is None:
            return

        state = self._ensure_window_state(network)
        task_label = self._extract_task_label(batch)
        if task_label in state.task_labels:
            raise ValueError(
                f"OrthoLORA received duplicate task {task_label} in one accumulation window"
            )

        scale = scalar_multiplier
        if scale is not None and torch.is_tensor(scale):
            scale = scale.to(device=per_sample_loss.device, dtype=per_sample_loss.dtype)

        task_loss = per_sample_loss.mean()
        if scale is not None:
            task_loss = task_loss * scale

        grads = torch.autograd.grad(
            task_loss,
            [entry.param for entry in state.entries],
            retain_graph=True,
            allow_unused=True,
        )
        state.task_labels.append(task_label)
        state.task_grads.append([_clone_optional_tensor(tensor) for tensor in grads])

    def finalize_window(self) -> Optional[Dict[str, float]]:
        if not self.enabled:
            return None

        state = self._window_state
        if state is None:
            return None

        if len(state.task_grads) < 2:
            raise ValueError(
                "OrthoLORA requires at least 2 task gradients in each accumulation window"
            )
        if len(state.task_grads) != len(set(state.task_labels)):
            raise ValueError("OrthoLORA accumulation window contains duplicate tasks")

        aggregate_surrogate_grads = [
            _sum_optional_tensors([task_grads[idx] for task_grads in state.task_grads])
            for idx in range(len(state.entries))
        ]
        projected_surrogate_grads: List[Optional[torch.Tensor]] = [
            _clone_optional_tensor(tensor) for tensor in aggregate_surrogate_grads
        ]

        conflict_pairs_down = 0
        conflict_pairs_up = 0
        for role in ("lora_down", "lora_up"):
            role_indices = [
                idx for idx, entry in enumerate(state.entries) if entry.role == role
            ]
            if not role_indices:
                continue

            role_task_grads = [
                [task_grads[idx] for idx in role_indices] for task_grads in state.task_grads
            ]
            projected_role_grads, conflict_count = project_conflicting_gradients(
                role_task_grads,
                shuffle_order=True,
            )
            role_sums = [
                _sum_optional_tensors([task_grads[idx] for task_grads in projected_role_grads])
                for idx in range(len(role_indices))
            ]
            for local_idx, param_idx in enumerate(role_indices):
                projected_surrogate_grads[param_idx] = role_sums[local_idx]
            if role == "lora_down":
                conflict_pairs_down = conflict_count
            else:
                conflict_pairs_up = conflict_count

        with torch.no_grad():
            for idx, entry in enumerate(state.entries):
                current_grad = _clone_optional_tensor(entry.param.grad)
                current_window_grad = _sub_optional_tensors(
                    current_grad,
                    state.window_start_grads[idx],
                )
                auxiliary_grad = _sub_optional_tensors(
                    current_window_grad,
                    aggregate_surrogate_grads[idx],
                )
                replacement_window_grad = _add_optional_tensors(
                    auxiliary_grad,
                    projected_surrogate_grads[idx],
                )
                final_grad = _add_optional_tensors(
                    state.window_start_grads[idx],
                    replacement_window_grad,
                )

                if final_grad is None:
                    entry.param.grad = None
                    continue

                final_grad = final_grad.to(
                    device=entry.param.device,
                    dtype=entry.param.dtype,
                )
                if entry.param.grad is None:
                    entry.param.grad = final_grad
                else:
                    entry.param.grad.copy_(final_grad)

        metrics = {
            "ortholora/active_groups": float(len(state.task_grads)),
            "ortholora/conflict_pairs_down": float(conflict_pairs_down),
            "ortholora/conflict_pairs_up": float(conflict_pairs_up),
            "ortholora/eligible_params": float(len(state.entries)),
        }
        self.reset_window()
        return metrics
