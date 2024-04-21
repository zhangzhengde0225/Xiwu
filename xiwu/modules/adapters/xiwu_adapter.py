from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import warnings
from xiwu.apis.fastchat_api import (
    BaseModelAdapter, register_model_adapter,
    Conversation, get_conv_template, conv_templates,
    SeparatorStyle
)
from ..base.base_adapter import XBaseModelAdapter
from xiwu import XConversation


class XiwuAdapter(XBaseModelAdapter):
    """Model adapater for Xiwu models"""
    conv = XConversation(
        name='xiwu',
        system_message="""You are Xiwu, answer questions conversationally. Gives helpful, detailed, and polite answers to the user's questions.""",
        roles=("USER", "ASSISTANT"),
        offset=0,
        messages=[],
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
    description = "HEP·Xiwu: a customized LLM for High Energy Physics"
    author = "HepAI Team, IHEP, CAS"
    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "xiwu" in model_path.lower()

    def get_default_conv_template(self) -> Conversation:
        return self.conv

    def raise_warning_for_old_weights(self, model):
        if isinstance(model, LlamaForCausalLM) and model.model.vocab_size > 32000:
            warnings.warn(
                "\nYou are probably using the old Vicuna-v0 model, "
                "which will generate unexpected results with the "
                "current fastchat.\nYou can try one of the following methods:\n"
                "1. Upgrade your weights to the new Vicuna-v1.3: https://github.com/lm-sys/FastChat#vicuna-weights.\n"
                "2. Use the old conversation template by `python3 -m fastchat.serve.cli --model-path /path/to/vicuna-v0 --conv-template one_shot`\n"
                "3. Downgrade fschat to fschat==0.1.10 (Not recommonded).\n"
            )

    

