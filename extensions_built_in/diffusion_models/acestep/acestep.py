import os
from typing import List, Optional, TYPE_CHECKING, Tuple, Union

import torch
import torch.nn.functional as F
import yaml

from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.models.base_model import BaseModel
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.accelerator import unwrap_model

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO


scheduler_config = {
    "num_train_timesteps": 1000,
    "use_dynamic_shifting": False,
    "shift": 3.0,
}


def _import_acestep_diffusers():
    try:
        from diffusers import AceStepPipeline, AceStepConditionEncoder, AceStepDiTModel
    except Exception as e:
        raise ImportError(
            "AceStepModel requires a diffusers build with ACE-Step support "
            "(PR #13095 classes: AceStepPipeline, AceStepConditionEncoder, AceStepDiTModel). "
            "Install a diffusers build that contains those classes."
        ) from e
    return AceStepPipeline, AceStepConditionEncoder, AceStepDiTModel


class AceStepModel(BaseModel):
    arch = "acestep"

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs
        )
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ["AceStepDiTModel"]
        self.condition_encoder = None

        mk = self.model_config.model_kwargs or {}
        self.vocal_language = mk.get("vocal_language", "en")
        self.prompt_audio_duration = float(mk.get("prompt_audio_duration", 30.0))
        self.max_text_length = int(mk.get("max_text_length", 256))
        self.max_lyric_length = int(mk.get("max_lyric_length", 2048))
        self.instruction = mk.get("instruction", None)
        self.bpm = mk.get("bpm", None)
        self.keyscale = mk.get("keyscale", None)
        self.timesignature = mk.get("timesignature", None)
        self.lyrics_separator = mk.get("lyrics_separator", "|||")
        self.use_caption_as_lyrics = bool(mk.get("use_caption_as_lyrics", True))
        self.default_lyrics = mk.get("default_lyrics", "")

        self.sample_shift = float(mk.get("sample_shift", 3.0))
        self.sample_use_tiled_decode = bool(mk.get("sample_use_tiled_decode", True))

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        # Audio-only model; image buckets are not used for latent shape.
        return 1

    def _split_prompt_and_lyrics(self, prompt: str) -> Tuple[str, str]:
        if (
            self.lyrics_separator
            and isinstance(prompt, str)
            and self.lyrics_separator in prompt
        ):
            text_part, lyric_part = prompt.split(self.lyrics_separator, 1)
            return text_part.strip(), lyric_part.strip()
        if self.use_caption_as_lyrics:
            return prompt, prompt
        return prompt, self.default_lyrics

    def load_model(self):
        AceStepPipeline, _, _ = _import_acestep_diffusers()

        dtype = self.torch_dtype
        model_path = self.model_config.name_or_path

        self.print_and_status_update("Loading ACE-Step model")
        pipe: AceStepPipeline = AceStepPipeline.from_pretrained(
            model_path, torch_dtype=dtype
        )

        self.noise_scheduler = AceStepModel.get_train_scheduler()

        if self.model_config.low_vram:
            pipe.transformer.to("cpu")
        else:
            pipe.transformer = pipe.transformer.to(self.device_torch)

        pipe.text_encoder.to(self.device_torch, dtype=dtype)
        pipe.text_encoder.requires_grad_(False)
        pipe.text_encoder.eval()

        pipe.condition_encoder.to(self.device_torch, dtype=dtype)
        pipe.condition_encoder.requires_grad_(False)
        pipe.condition_encoder.eval()

        self.vae = pipe.vae
        self.text_encoder = [pipe.text_encoder]
        self.tokenizer = [pipe.tokenizer]
        self.model = pipe.transformer
        self.condition_encoder = pipe.condition_encoder
        self.pipeline = pipe

        self.print_and_status_update("Model Loaded")

    def get_generation_pipeline(self):
        AceStepPipeline, _, _ = _import_acestep_diffusers()

        pipeline = AceStepPipeline(
            vae=unwrap_model(self.vae),
            text_encoder=unwrap_model(self.text_encoder[0]),
            tokenizer=self.tokenizer[0],
            transformer=unwrap_model(self.transformer),
            condition_encoder=unwrap_model(self.condition_encoder),
        )
        pipeline = pipeline.to(self.device_torch)
        return pipeline

    def generate_single_image(
        self,
        pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        batch_size = conditional_embeds.text_embeds.shape[0]

        latents = pipeline.prepare_latents(
            batch_size=batch_size,
            audio_duration=self.prompt_audio_duration,
            dtype=self.torch_dtype,
            device=self.device_torch,
            generator=generator,
            latents=gen_config.latents,
        )

        src_latents = torch.zeros_like(latents)
        chunk_mask = torch.ones_like(latents)
        context_latents = torch.cat([src_latents, chunk_mask], dim=-1)

        t_schedule = pipeline._get_timestep_schedule(
            num_inference_steps=gen_config.num_inference_steps,
            shift=self.sample_shift,
            device=self.device_torch,
            dtype=self.torch_dtype,
        )

        do_cfg = gen_config.guidance_scale > 1.0 and unconditional_embeds is not None
        xt = latents
        num_steps = len(t_schedule)

        for step_idx in range(num_steps):
            current_t = t_schedule[step_idx].item()
            t_curr = current_t * torch.ones(
                (batch_size,), device=self.device_torch, dtype=self.torch_dtype
            )

            if do_cfg:
                vt_cond = self.transformer(
                    hidden_states=xt,
                    timestep=t_curr,
                    timestep_r=t_curr,
                    encoder_hidden_states=conditional_embeds.text_embeds.to(
                        self.device_torch, dtype=self.torch_dtype
                    ),
                    context_latents=context_latents,
                    return_dict=False,
                )[0]
                vt_uncond = self.transformer(
                    hidden_states=xt,
                    timestep=t_curr,
                    timestep_r=t_curr,
                    encoder_hidden_states=unconditional_embeds.text_embeds.to(
                        self.device_torch, dtype=self.torch_dtype
                    ),
                    context_latents=context_latents,
                    return_dict=False,
                )[0]
                vt = vt_uncond + gen_config.guidance_scale * (vt_cond - vt_uncond)
            else:
                vt = self.transformer(
                    hidden_states=xt,
                    timestep=t_curr,
                    timestep_r=t_curr,
                    encoder_hidden_states=conditional_embeds.text_embeds.to(
                        self.device_torch, dtype=self.torch_dtype
                    ),
                    context_latents=context_latents,
                    return_dict=False,
                )[0]

            if step_idx == num_steps - 1:
                xt = xt - vt * t_curr.unsqueeze(-1).unsqueeze(-1)
                break

            next_t = t_schedule[step_idx + 1].item()
            dt = current_t - next_t
            dt_tensor = dt * torch.ones(
                (batch_size, 1, 1), device=self.device_torch, dtype=self.torch_dtype
            )
            xt = xt - vt * dt_tensor

        audio_latents = xt.transpose(1, 2)
        if self.sample_use_tiled_decode:
            audio = pipeline._tiled_decode(audio_latents)
        else:
            audio = self.vae.decode(audio_latents).sample

        if audio.dtype != torch.float32:
            audio = audio.float()
        std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
        std[std < 1.0] = 1.0
        audio = audio / std

        return audio

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: PromptEmbeds,
        **kwargs,
    ):
        self.model.to(self.device_torch)

        original_dim = latent_model_input.dim()
        if original_dim == 4:
            if latent_model_input.shape[-1] != 1:
                raise ValueError(
                    "ACE-Step expects latent tensors shaped [B, C, T, 1] for training."
                )
            # [B, C, T, 1] -> [B, T, C]
            hidden_states = latent_model_input.squeeze(-1).transpose(1, 2)
        elif original_dim == 3:
            hidden_states = latent_model_input
        else:
            raise ValueError(
                f"ACE-Step latent rank must be 3 or 4, got {latent_model_input.shape}"
            )

        batch_size = hidden_states.shape[0]

        if len(timestep.shape) == 0:
            timestep = timestep.unsqueeze(0)
        if timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.repeat(batch_size)
        timestep = timestep.to(self.device_torch, dtype=self.torch_dtype)
        if torch.max(timestep).item() > 1.0:
            timestep = timestep / 1000.0

        src_latents = torch.zeros_like(hidden_states)
        chunk_mask = torch.ones_like(hidden_states)
        context_latents = torch.cat([src_latents, chunk_mask], dim=-1)

        encoder_hidden_states = text_embeddings.text_embeds.to(
            self.device_torch, dtype=self.torch_dtype
        )
        encoder_attention_mask = None
        if text_embeddings.attention_mask is not None:
            encoder_attention_mask = text_embeddings.attention_mask.to(self.device_torch)

        noise_pred = self.transformer(
            hidden_states=hidden_states.to(self.device_torch, dtype=self.torch_dtype),
            timestep=timestep,
            timestep_r=timestep,
            encoder_hidden_states=encoder_hidden_states,
            context_latents=context_latents.to(self.device_torch, dtype=self.torch_dtype),
            encoder_attention_mask=encoder_attention_mask,
            return_dict=False,
        )[0]

        if original_dim == 4:
            # [B, T, C] -> [B, C, T, 1]
            return noise_pred.transpose(1, 2).unsqueeze(-1)
        return noise_pred

    def get_prompt_embeds(self, prompt: Union[str, List[str]]) -> PromptEmbeds:
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch, dtype=self.torch_dtype)
        if self.pipeline.condition_encoder.device != self.device_torch:
            self.pipeline.condition_encoder.to(self.device_torch, dtype=self.torch_dtype)

        if isinstance(prompt, str):
            prompt_list = [prompt]
        else:
            prompt_list = [p.strip() for p in prompt]

        text_prompt_list = []
        lyric_list = []
        for p in prompt_list:
            text_prompt, lyrics = self._split_prompt_and_lyrics(p)
            text_prompt_list.append(text_prompt)
            lyric_list.append(lyrics)

        (
            text_hidden_states,
            text_attention_mask,
            lyric_hidden_states,
            lyric_attention_mask,
        ) = self.pipeline.encode_prompt(
            prompt=text_prompt_list,
            lyrics=lyric_list,
            device=self.device_torch,
            vocal_language=self.vocal_language,
            audio_duration=self.prompt_audio_duration,
            instruction=self.instruction,
            bpm=self.bpm,
            keyscale=self.keyscale,
            timesignature=self.timesignature,
            max_text_length=self.max_text_length,
            max_lyric_length=self.max_lyric_length,
        )

        batch_size = len(text_prompt_list)
        timbre_fix_frame = 750
        timbre_hidden_dim = self.condition_encoder.config.timbre_hidden_dim
        refer_audio_acoustic = torch.zeros(
            batch_size,
            timbre_fix_frame,
            timbre_hidden_dim,
            device=self.device_torch,
            dtype=self.torch_dtype,
        )
        refer_audio_order_mask = torch.arange(
            batch_size, device=self.device_torch, dtype=torch.long
        )

        encoder_hidden_states, encoder_attention_mask = self.condition_encoder(
            text_hidden_states=text_hidden_states.to(self.device_torch, dtype=self.torch_dtype),
            text_attention_mask=text_attention_mask.to(self.device_torch),
            lyric_hidden_states=lyric_hidden_states.to(self.device_torch, dtype=self.torch_dtype),
            lyric_attention_mask=lyric_attention_mask.to(self.device_torch),
            refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic,
            refer_audio_order_mask=refer_audio_order_mask,
        )

        pe = PromptEmbeds(encoder_hidden_states)
        pe.attention_mask = encoder_attention_mask
        return pe

    def _normalize_waveform(
        self, waveform: torch.Tensor, sample_rate: int, target_rate: int
    ) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(torch.float32)

        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        waveform = waveform[:2]

        if sample_rate != target_rate:
            try:
                import torchaudio

                waveform = torchaudio.functional.resample(
                    waveform, sample_rate, target_rate
                )
            except Exception:
                target_len = int(waveform.shape[-1] * target_rate / sample_rate)
                waveform = F.interpolate(
                    waveform.unsqueeze(0),
                    size=target_len,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0)

        waveform = torch.clamp(waveform, -1.0, 1.0)
        return waveform

    def encode_audio(self, audio_data_list):
        if self.vae.device == torch.device("cpu"):
            self.vae.to(self.device_torch)
        self.vae.eval()
        self.vae.requires_grad_(False)

        target_rate = int(getattr(self.vae.config, "sampling_rate", 48000))
        encoded = [None] * len(audio_data_list)
        max_latent_t = 0
        latent_channels = None

        for i, audio_data in enumerate(audio_data_list):
            if audio_data is None:
                continue

            waveform = audio_data["waveform"]
            sample_rate = int(audio_data.get("sample_rate", target_rate))
            waveform = self._normalize_waveform(waveform, sample_rate, target_rate)
            waveform = waveform.unsqueeze(0).to(self.device_torch, dtype=self.vae.dtype)

            with torch.no_grad():
                lat = self.vae.encode(waveform).latent_dist.sample()

            lat = lat.to(self.device_torch, dtype=self.torch_dtype)
            encoded[i] = lat
            max_latent_t = max(max_latent_t, lat.shape[-1])
            if latent_channels is None:
                latent_channels = lat.shape[1]

        if latent_channels is None:
            raise ValueError("No valid audio data found in batch.")

        padded = []
        for lat in encoded:
            if lat is None:
                lat = torch.zeros(
                    (1, latent_channels, max_latent_t),
                    device=self.device_torch,
                    dtype=self.torch_dtype,
                )
            elif lat.shape[-1] < max_latent_t:
                pad_t = max_latent_t - lat.shape[-1]
                lat = F.pad(lat, (0, pad_t))
            elif lat.shape[-1] > max_latent_t:
                lat = lat[:, :, :max_latent_t]
            padded.append(lat)

        latents = torch.cat(padded, dim=0)
        # Keep shape [B, C, T, 1] to fit image/video trainer tensor assumptions.
        return latents.unsqueeze(-1)

    def encode_images(self, image_list: List[torch.Tensor], device=None, dtype=None):
        raise ValueError(
            "ACE-Step training expects audio latents. Use an audio dataset with `do_audio: true`."
        )

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        _, _, AceStepDiTModel = _import_acestep_diffusers()
        transformer: AceStepDiTModel = unwrap_model(self.model)
        transformer.save_pretrained(
            save_directory=os.path.join(output_path, "transformer"),
            safe_serialization=True,
        )

        meta_path = os.path.join(output_path, "aitk_meta.yaml")
        with open(meta_path, "w") as f:
            yaml.dump(meta, f)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def get_base_model_version(self):
        return "acestep"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["layers"]
