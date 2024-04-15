from typing import Any, Dict, Optional, Tuple

import os, sys
from pathlib import Path
import transformers
from transformers import Trainer
here = Path(__file__).parent
import math

try:
    from xiwu.version import __version__
except:
    sys.path.insert(1, str(here.parent.parent))
    from xiwu.version import __version__
from xiwu.configs import TrainingArgs, ModelArgs, DataArgs
from xiwu import ASSEMBLER
from xiwu.apis.fastchat_api import (
    make_supervised_data_module, trainer_save_model_safe,
)
import hepai

class XTrainer():

    @classmethod
    def train(cls):
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
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

        # Start trainner
        trainer = Trainer(
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
    XTrainer.train()
