# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

import os, sys
from pathlib import Path
here = Path(__file__).parent

# from llama2_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
# Need to call this before importing transformers.
from fastchat.train.llama2_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

replace_llama_attn_with_flash_attn()

try:
    from xiwu.version import __version__
except:
    sys.path.insert(1, str(here.parent.parent.parent))
    from xiwu.version import __version__


# from fastchat.train.train import train
from xiwu.modules.trainer import xtrainer

if __name__ == "__main__":
    xtrainer.train()
    # train()
