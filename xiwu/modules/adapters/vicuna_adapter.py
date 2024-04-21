
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import warnings
from xiwu.apis.fastchat_api import (
    BaseModelAdapter, register_model_adapter,
    Conversation, get_conv_template, conv_templates,
    SeparatorStyle
)
from xiwu import XConversation
from ..base.base_adapter import XBaseModelAdapter


class VicunaAdapter(XBaseModelAdapter):
    "Model adapater for Vicuna models (e.g., lmsys/vicuna-7b-v1.3)"
    conv = XConversation(
        name='vicuna',
        system_message="""You are Vicuna, answer questions conversationally. Gives helpful, detailed, and polite answers to the user's questions.""",
        roles=("USER", "ASSISTANT"),
        offset=0,
        messages=[],
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
    description = "Vicuna: a large language model trained by LMSYS team"
    author = "LMSYS"
    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "vicuna" in model_path.lower()

    def get_default_conv_template(self) -> Conversation:
        """
        hepai/vicuna-xxx
        """
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
            