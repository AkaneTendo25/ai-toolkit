"""Aggregator for built-in diffusion model adapters.

Each adapter is wrapped in a per-package try/except so a missing or
out-of-date diffusers pipeline (LTX2, Flux2, Z-Image, …) does not bring
down the whole ai-toolkit loader. Adapters that fail to import are
omitted from ``AI_TOOLKIT_MODELS`` and a warning is printed once at
startup; the remaining adapters keep working.
"""

import warnings

AI_TOOLKIT_MODELS = []


def _try_import(import_callable, label):
    """Run ``import_callable`` (a zero-arg function returning a list of
    model classes), append the result to ``AI_TOOLKIT_MODELS``, and
    swallow ImportError so a single bad extension never blocks loading
    the others."""
    try:
        models = import_callable()
    except ImportError as e:
        warnings.warn(
            f"[ai-toolkit] skipping diffusion_model adapter {label!r}: {e}",
            stacklevel=2,
        )
        return
    AI_TOOLKIT_MODELS.extend(models)


def _chroma():
    from .chroma import ChromaModel, ChromaRadianceModel
    return [ChromaModel, ChromaRadianceModel]


def _hidream():
    from .hidream import HidreamModel, HidreamE1Model
    return [HidreamModel, HidreamE1Model]


def _f_light():
    from .f_light import FLiteModel
    return [FLiteModel]


def _omnigen2():
    from .omnigen2 import OmniGen2Model
    return [OmniGen2Model]


def _flux_kontext():
    from .flux_kontext import FluxKontextModel
    return [FluxKontextModel]


def _wan22():
    from .wan22 import Wan225bModel, Wan2214bModel, Wan2214bI2VModel
    return [Wan225bModel, Wan2214bI2VModel, Wan2214bModel]


def _qwen_image():
    from .qwen_image import QwenImageModel, QwenImageEditModel, QwenImageEditPlusModel
    return [QwenImageModel, QwenImageEditModel, QwenImageEditPlusModel]


def _flux2():
    from .flux2 import Flux2Model, Flux2Klein4BModel, Flux2Klein9BModel
    return [Flux2Model, Flux2Klein4BModel, Flux2Klein9BModel]


def _z_image():
    from .z_image import ZImageModel
    return [ZImageModel]


def _ltx2():
    from .ltx2 import LTX2Model, LTX23Model
    return [LTX2Model, LTX23Model]


def _zeta_chroma():
    from .zeta_chroma import ZetaChromaModel
    return [ZetaChromaModel]


def _ernie_image():
    from .ernie_image import ErnieImageModel
    return [ErnieImageModel]


def _nucleus_image():
    from .nucleus_image import NucleusImageModel
    return [NucleusImageModel]


def _hidream_o1():
    from .hidream.hidream_o1_model import HidreamO1Model
    return [HidreamO1Model]


def _z_image_l2p():
    from .z_image.z_image_l2p_model import ZImageL2PModel
    return [ZImageL2PModel]


def _ideogram4():
    from .ideogram4 import Ideogram4Model
    return [Ideogram4Model]


def _anima():
    from .anima import AnimaModel
    return [AnimaModel]


def _prx_pixel_t2i():
    from .prx_pixel_t2i import PRXPixelT2IModel
    return [PRXPixelT2IModel]


def _krea2():
    from .krea2 import Krea2Model
    return [Krea2Model]


def _boogu_image():
    from .boogu_image import BooguImageModel, BooguImageEditModel
    return [BooguImageModel, BooguImageEditModel]


def _mageflow():
    from .mageflow import MageFlowModel, MageFlowEditModel
    return [MageFlowModel, MageFlowEditModel]


for _label, _fn in (
    ("chroma", _chroma),
    ("hidream", _hidream),
    ("f_light", _f_light),
    ("omnigen2", _omnigen2),
    ("flux_kontext", _flux_kontext),
    ("wan22", _wan22),
    ("qwen_image", _qwen_image),
    ("flux2", _flux2),
    ("z_image", _z_image),
    ("ltx2", _ltx2),
    ("zeta_chroma", _zeta_chroma),
    ("ernie_image", _ernie_image),
    ("nucleus_image", _nucleus_image),
    ("hidream_o1", _hidream_o1),
    ("z_image_l2p", _z_image_l2p),
    ("anima", _anima),
    ("ideogram4", _ideogram4),
    ("prx_pixel_t2i", _prx_pixel_t2i),
    ("krea2", _krea2),
    ("boogu_image", _boogu_image),
    ("mageflow", _mageflow),
):
    _try_import(_fn, _label)
