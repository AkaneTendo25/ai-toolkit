"""MuDI mask preprocessing stub.

The MuDI (Jang et al., NeurIPS 2024) Seg-Mix augmentation expects a binary
foreground mask alongside every training image. This script is a stub — it
documents the offline pipeline the authors recommend but does not run any of
the heavy detectors. The user is expected to either:

1. Use the official notebook at github.com/agwmon/MuDI
   (``automatic_mask_generation.ipynb``), which combines OWLv2 (open-vocab
   bounding-box detection from a text query like ``"a person in a suit"``)
   with SAM (mask refinement seeded by the bbox), and ship the resulting
   masks in the layout expected by ai-toolkit's seg_mix config:

       <mask_path>/<subject_id>/<image_basename>.png

2. Or substitute any other instance segmenter (SAM2, YOLO-World + SAM, a
   commercial API) — Seg-Mix only cares that the mask cleanly isolates the
   target subject.

We deliberately keep this out of ai-toolkit's training loop because: (a) SAM
checkpoints are several gigabytes and not part of the training environment,
(b) the masks are one-shot artefacts that benefit from manual QC rather than
silent regeneration on every run, and (c) the segmentation choice is
orthogonal to the LoRA training itself.
"""

from __future__ import annotations

import sys


_INSTRUCTIONS = """\
MuDI mask preprocessing is offline. To generate masks:

  1. Clone github.com/agwmon/MuDI and follow
     `automatic_mask_generation.ipynb`. It uses OWLv2 + SAM on a per-image
     bbox prompt and writes binary masks (1 = subject, 0 = background).

  2. Arrange masks to mirror your dataset layout, e.g.

       dataset_root/subject_a/img_001.png         -> masks_root/subject_a/img_001.png
       dataset_root/subject_b/img_002.png    -> masks_root/subject_b/img_002.png

  3. Set ``mask_path: <masks_root>`` in your ai-toolkit dataset config. With
     ``seg_mix.enabled: true`` the dataloader will pick masks up
     automatically.

  4. (Recommended.) Spend ~1 hour reviewing masks visually; bad masks are
     the dominant failure mode of MuDI per the paper §8 limitations.

This script does not run any model — it only prints these instructions.
"""


def main() -> int:
    print(_INSTRUCTIONS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
