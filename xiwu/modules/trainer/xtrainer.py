from typing import Any, Dict, Optional, Tuple

import os, sys
from pathlib import Path

here = Path(__file__).parent

import math
import transformers
from transformers import Trainer
from transformers.utils import (
    is_accelerate_available,
)

if is_accelerate_available():
    from accelerate import Accelerator
    from accelerate.utils import (
        GradientAccumulationPlugin,
        DataLoaderConfiguration,
    )

try:
    from xiwu.version import __version__
except:
    sys.path.insert(1, str(here.parent.parent))
    from xiwu.version import __version__
from xiwu.configs import TrainingArgs, ModelArgs, DataArgs
from xiwu import ASSEMBLER
from xiwu.apis.fastchat_api import (
    # make_supervised_data_module, 
    trainer_save_model_safe,
)
import hepai
from .dataloader import DataAssembler


class XTrainer(Trainer):

    def create_accelerator_and_postprocess(self):
        """overwrite to fix Accelerator >= 0.29.2 issue with `dataloader_config`."""
        grad_acc_kwargs = {"num_steps": self.args.gradient_accumulation_steps}
        grad_acc_kwargs["sync_with_dataloader"] = False
        gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)

        # create accelerator object
        dataloader_config = DataLoaderConfiguration(**self.args.accelerator_config.to_dict())
        self.accelerator = Accelerator(
            deepspeed_plugin=self.args.deepspeed_plugin,
            gradient_accumulation_plugin=gradient_accumulation_plugin,
            # **self.args.accelerator_config.to_dict(),
            dataloader_config=dataloader_config,  # fix for Accelerate >= 0.29.2
        )
        # some Trainer classes need to use `gather` instead of `gather_for_metrics`, thus we store a flag
        self.gather_function = self.accelerator.gather_for_metrics

        # deepspeed and accelerate flags covering both trainer args and accelerate launcher
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None

        # post accelerator creation setup
        if self.is_fsdp_enabled:
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            fsdp_plugin.limit_all_gathers = self.args.fsdp_config.get(
                "limit_all_gathers", fsdp_plugin.limit_all_gathers
            )
            if is_accelerate_available("0.23.0"):
                fsdp_plugin.activation_checkpointing = self.args.fsdp_config.get(
                    "activation_checkpointing", fsdp_plugin.activation_checkpointing
                )
                if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
                    raise ValueError(
                        "The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg "
                        "can't be set to True simultaneously. Please use FSDP's activation_checkpointing logic "
                        "when using FSDP."
                    )

        if self.is_deepspeed_enabled and getattr(self.args, "hf_deepspeed_config", None) is None:
            self.propagate_args_to_deepspeed()

        # `save_only_model` can't be used with DeepSpeed/FSDP along with `load_best_model_at_end`
        if (
            self.args.save_only_model
            and (self.is_deepspeed_enabled or self.is_fsdp_enabled)
            and self.args.load_best_model_at_end
        ):
            wrapper = "DeepSpeed" if self.is_deepspeed_enabled else "FSDP"
            raise ValueError(f"{wrapper} can't be used with `save_only_model` along with `load_best_model_at_end`.")

        # `auto_find_batch_size` isn't yet supported with DeepSpeed/FSDP
        if (self.is_deepspeed_enabled or self.is_fsdp_enabled) and self.args.auto_find_batch_size:
            wrapper = "DeepSpeed" if self.is_deepspeed_enabled else "FSDP"
            raise NotImplementedError(f"`{wrapper}` doesn't support `auto_find_batch_size`.")


  
def train():
    global local_rank

    model_args, data_args, training_args = hepai.parse_args((ModelArgs, DataArgs, TrainingArgs))
    local_rank = training_args.local_rank

    # Set RoPE scaling factor
    config = ASSEMBLER.load_config_from_pretrained(
    # config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    # Load model and tokenizer
    model = ASSEMBLER.load_model_from_pretrained(
    # model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = ASSEMBLER.load_tokenizer_from_pretrained(
    # tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
        trust_remote_code=model_args.trust_remote_code,
    )

    if tokenizer.pad_token != tokenizer.unk_token:  # <unk>
        tokenizer.pad_token = tokenizer.unk_token

    # Load data
    data_module = DataAssembler.make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Start trainner
    trainer: Trainer = XTrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    if list(Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    model.config.use_cache = True
    trainer.save_state()
    if trainer.is_deepspeed_enabled:
        trainer.save_model()
    else:
        trainer_save_model_safe(trainer)


if __name__ == "__main__":
    train()
