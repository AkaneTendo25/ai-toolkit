import os
import shlex
import subprocess
import sys
from collections import OrderedDict
from typing import TYPE_CHECKING, List

from jobs.process import BaseExtensionProcess

if TYPE_CHECKING:
    from jobs import ExtensionJob


class AceStepLoraTrainerProcess(BaseExtensionProcess):
    SUPPORTED_SUBCOMMANDS = {"fixed", "vanilla", "estimate"}

    VALUE_FLAGS = OrderedDict(
        [
            ("base_model", "--base-model"),
            ("device", "--device"),
            ("precision", "--precision"),
            ("num_workers", "--num-workers"),
            ("batch_size", "--batch-size"),
            ("gradient_accumulation", "--gradient-accumulation"),
            ("epochs", "--epochs"),
            ("warmup_steps", "--warmup-steps"),
            ("weight_decay", "--weight-decay"),
            ("max_grad_norm", "--max-grad-norm"),
            ("seed", "--seed"),
            ("shift", "--shift"),
            ("num_inference_steps", "--num-inference-steps"),
            ("optimizer_type", "--optimizer-type"),
            ("scheduler_type", "--scheduler-type"),
            ("learning_rate", "--learning-rate"),
            ("rank", "--rank"),
            ("alpha", "--alpha"),
            ("dropout", "--dropout"),
            ("bias", "--bias"),
            ("attention_type", "--attention-type"),
            ("lokr_linear_dim", "--lokr-linear-dim"),
            ("lokr_linear_alpha", "--lokr-linear-alpha"),
            ("lokr_factor", "--lokr-factor"),
            ("save_every", "--save-every"),
            ("resume_from", "--resume-from"),
            ("log_dir", "--log-dir"),
            ("log_every", "--log-every"),
            ("log_heavy_every", "--log-heavy-every"),
            ("sample_every_n_epochs", "--sample-every-n-epochs"),
            ("cfg_ratio", "--cfg-ratio"),
            ("estimate_batches", "--estimate-batches"),
            ("top_k", "--top-k"),
            ("granularity", "--granularity"),
            ("estimate_output", "--output"),
            ("audio_dir", "--audio-dir"),
            ("dataset_json", "--dataset-json"),
            ("tensor_output", "--tensor-output"),
            ("max_duration", "--max-duration"),
            ("adapter_type", "--adapter-type"),
        ]
    )

    STORE_TRUE_FLAGS = OrderedDict(
        [
            ("plain", "--plain"),
            ("yes", "--yes"),
            ("preprocess", "--preprocess"),
            ("lokr_decompose_both", "--lokr-decompose-both"),
            ("lokr_use_tucker", "--lokr-use-tucker"),
            ("lokr_use_scalar", "--lokr-use-scalar"),
            ("lokr_weight_decompose", "--lokr-weight-decompose"),
        ]
    )

    BOOLEAN_OPTIONAL_FLAGS = OrderedDict(
        [
            ("pin_memory", "--pin-memory"),
            ("persistent_workers", "--persistent-workers"),
            ("gradient_checkpointing", "--gradient-checkpointing"),
            ("offload_encoder", "--offload-encoder"),
        ]
    )

    def __init__(
        self,
        process_id: int,
        job: "ExtensionJob",
        config: OrderedDict,
    ):
        super().__init__(process_id, job, config)

        self.acestep_repo_path = self.get_conf("acestep_repo_path", required=True)
        self.subcommand = self.get_conf("subcommand", "fixed")
        self.use_uv = self.get_conf("use_uv", False)
        self.python_executable = self.get_conf("python_executable", sys.executable)
        self.dry_run = self.get_conf("dry_run", False)
        self.checkpoint_dir = self.get_conf("checkpoint_dir", required=True)
        self.model_variant = self.get_conf("model_variant", "turbo")

        self.dataset_dir = self.get_conf("dataset_dir", None)
        self.output_dir = self.get_conf("output_dir", None)

        self.target_modules = self.get_conf("target_modules", None)
        self.extra_args = self.get_conf("extra_args", [])
        if self.extra_args is None:
            self.extra_args = []
        if not isinstance(self.extra_args, list):
            raise ValueError('"extra_args" must be a list of CLI args')

    def _validate(self):
        if self.subcommand not in self.SUPPORTED_SUBCOMMANDS:
            raise ValueError(
                f"Unsupported subcommand '{self.subcommand}'. "
                f"Supported: {sorted(self.SUPPORTED_SUBCOMMANDS)}"
            )

        if not os.path.isdir(self.acestep_repo_path):
            raise ValueError(
                f"acestep_repo_path does not exist or is not a directory: {self.acestep_repo_path}"
            )

        train_script = os.path.join(self.acestep_repo_path, "train.py")
        if not os.path.isfile(train_script):
            raise ValueError(f"Could not find ACE-Step train script at: {train_script}")

        if self.subcommand in {"fixed", "vanilla", "estimate"} and self.dataset_dir is None:
            raise ValueError('"dataset_dir" is required for fixed/vanilla/estimate subcommands')

        if self.subcommand in {"fixed", "vanilla"} and self.output_dir is None:
            raise ValueError('"output_dir" is required for fixed/vanilla subcommands')

        if self.target_modules is not None and not isinstance(self.target_modules, (list, str)):
            raise ValueError('"target_modules" must be a list of module names or a space-delimited string')

    def _add_optional_flags(self, command: List[str]):
        for key, flag in self.VALUE_FLAGS.items():
            value = self.get_conf(key, None)
            if value is None:
                continue
            command.extend([flag, str(value)])

        for key, flag in self.STORE_TRUE_FLAGS.items():
            value = self.get_conf(key, False)
            if value:
                command.append(flag)

        for key, flag in self.BOOLEAN_OPTIONAL_FLAGS.items():
            value = self.get_conf(key, None)
            if value is None:
                continue
            command.append(flag if bool(value) else f"--no-{flag[2:]}")

        if self.target_modules is not None:
            target_modules = self.target_modules
            if isinstance(target_modules, str):
                target_modules = [x for x in target_modules.split(" ") if x.strip() != ""]
            if len(target_modules) > 0:
                command.extend(["--target-modules", *target_modules])

        if len(self.extra_args) > 0:
            command.extend([str(arg) for arg in self.extra_args])

    def _build_command(self) -> List[str]:
        if self.use_uv:
            command = ["uv", "run", "train.py", self.subcommand]
        else:
            command = [self.python_executable, "train.py", self.subcommand]

        command.extend(["--checkpoint-dir", self.checkpoint_dir])
        command.extend(["--model-variant", self.model_variant])

        if self.dataset_dir is not None:
            command.extend(["--dataset-dir", self.dataset_dir])

        if self.output_dir is not None:
            command.extend(["--output-dir", self.output_dir])

        self._add_optional_flags(command)
        return command

    def run(self):
        super().run()
        self._validate()
        command = self._build_command()

        print(f"Launching ACE-Step {self.subcommand} training command:")
        print("  " + " ".join(shlex.quote(part) for part in command))
        print(f"Working directory: {self.acestep_repo_path}")
        if self.dry_run:
            print("dry_run=true: command was validated and built, but not executed.")
            return

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        process = None
        try:
            process = subprocess.Popen(
                command,
                cwd=self.acestep_repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
                env=env,
            )

            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")

            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"ACE-Step training exited with code {return_code}")

            print("ACE-Step command completed successfully")
        except KeyboardInterrupt:
            if process is not None and process.poll() is None:
                process.terminate()
            raise
