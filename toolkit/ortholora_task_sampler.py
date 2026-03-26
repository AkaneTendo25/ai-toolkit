from __future__ import annotations

import random
from typing import Dict, Iterator, List

import torch


def _split_concat_indices_by_task(dataset_group) -> Dict[int, List[int]]:
    datasets = getattr(dataset_group, "datasets", None)
    if datasets is None:
        raise ValueError("OrthoLORA requires a ConcatDataset with a datasets attribute")

    task_to_indices: Dict[int, List[int]] = {}
    global_offset = 0
    for dataset_idx, dataset in enumerate(datasets):
        dataset_len = len(dataset)
        indices = list(range(global_offset, global_offset + dataset_len))
        if indices:
            task_to_indices[int(dataset_idx)] = indices
        global_offset += dataset_len

    return task_to_indices


class TaskWindowIndexSampler(torch.utils.data.Sampler[int]):
    """Yield indices so each accumulation window contains all task datasets exactly once."""

    def __init__(
        self,
        *,
        dataset_group,
        accumulation_steps: int,
        seed: int = 0,
    ) -> None:
        if accumulation_steps < 2:
            raise ValueError(f"OrthoLORA requires accumulation_steps >= 2, got {accumulation_steps}")

        self.dataset_group = dataset_group
        self.accumulation_steps = int(accumulation_steps)
        self.seed = int(seed)
        self.epoch = 0

    def _current_groups(self) -> Dict[int, List[int]]:
        task_to_indices = _split_concat_indices_by_task(self.dataset_group)
        if len(task_to_indices) < 2:
            raise ValueError(
                f"OrthoLORA requires at least 2 non-empty task datasets, got {len(task_to_indices)}"
            )
        if self.accumulation_steps != len(task_to_indices):
            raise ValueError(
                "OrthoLORA requires gradient_accumulation_steps to equal the number of task datasets "
                f"(got {self.accumulation_steps} vs {len(task_to_indices)})"
            )
        return task_to_indices

    def __len__(self) -> int:
        task_to_indices = self._current_groups()
        window_count = min(len(indices) for indices in task_to_indices.values())
        return window_count * self.accumulation_steps

    def __iter__(self):
        task_to_indices = self._current_groups()
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        pools = {task_id: indices.copy() for task_id, indices in task_to_indices.items()}
        for indices in pools.values():
            rng.shuffle(indices)

        window_count = min(len(indices) for indices in pools.values())
        task_ids = sorted(pools.keys())
        ordered: List[int] = []
        for _ in range(window_count):
            chosen_tasks = task_ids.copy()
            rng.shuffle(chosen_tasks)
            for task_id in chosen_tasks:
                ordered.append(pools[task_id].pop())

        return iter(ordered)


class TaskWindowBatchSampler(torch.utils.data.Sampler[List[int]]):
    """Yield minibatches so each accumulation window contains all task datasets exactly once."""

    def __init__(
        self,
        *,
        dataset_group,
        batch_size: int,
        accumulation_steps: int,
        seed: int = 0,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"OrthoLORA requires batch_size >= 1, got {batch_size}")
        if accumulation_steps < 2:
            raise ValueError(f"OrthoLORA requires accumulation_steps >= 2, got {accumulation_steps}")

        self.dataset_group = dataset_group
        self.batch_size = int(batch_size)
        self.accumulation_steps = int(accumulation_steps)
        self.seed = int(seed)
        self.epoch = 0

    def _current_groups(self) -> Dict[int, List[int]]:
        task_to_indices = _split_concat_indices_by_task(self.dataset_group)
        if len(task_to_indices) < 2:
            raise ValueError(
                f"OrthoLORA requires at least 2 non-empty task datasets, got {len(task_to_indices)}"
            )
        if self.accumulation_steps != len(task_to_indices):
            raise ValueError(
                "OrthoLORA requires gradient_accumulation_steps to equal the number of task datasets "
                f"(got {self.accumulation_steps} vs {len(task_to_indices)})"
            )
        return task_to_indices

    def __len__(self) -> int:
        task_to_indices = self._current_groups()
        window_count = min(len(indices) // self.batch_size for indices in task_to_indices.values())
        return window_count * self.accumulation_steps

    def __iter__(self) -> Iterator[List[int]]:
        task_to_indices = self._current_groups()
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        pools = {task_id: indices.copy() for task_id, indices in task_to_indices.items()}
        for indices in pools.values():
            rng.shuffle(indices)

        per_task_batches: Dict[int, List[List[int]]] = {}
        for task_id, indices in pools.items():
            batch_count = len(indices) // self.batch_size
            per_task_batches[task_id] = [
                indices[start : start + self.batch_size]
                for start in range(0, batch_count * self.batch_size, self.batch_size)
            ]

        window_count = min(len(batches) for batches in per_task_batches.values())
        task_ids = sorted(per_task_batches.keys())
        ordered: List[List[int]] = []
        for window_idx in range(window_count):
            chosen_tasks = task_ids.copy()
            rng.shuffle(chosen_tasks)
            for task_id in chosen_tasks:
                ordered.append(per_task_batches[task_id][window_idx])

        return iter(ordered)
