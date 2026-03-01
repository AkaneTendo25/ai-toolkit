from toolkit.extension import Extension


class AceStepLoraTrainerExtension(Extension):
    uid = "acestep_lora_trainer"
    name = "ACE-Step LoRA Trainer"

    @classmethod
    def get_process(cls):
        from .AceStepLoraTrainerProcess import AceStepLoraTrainerProcess

        return AceStepLoraTrainerProcess


AI_TOOLKIT_EXTENSIONS = [
    AceStepLoraTrainerExtension,
]
