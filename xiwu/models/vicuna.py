"""
加载模型和单个推理
"""

import os, sys
from pathlib import Path
import dataclasses
from dataclasses import dataclass, field
import argparse
import hai

here = Path(__file__).parent

try:
    from xiwu.version import __version__
except:
    sys.path.insert(1, str(here.parent.parent))
    from xiwu.version import __version__
from xiwu.models.base_model import XBaseModel
from xiwu.apis import fastchat_api as fsapi
from xiwu.apis.fastchat_api import *

class Vicuna(XBaseModel):
    def __init__(self, args=None, **kwargs):
        super().__init__()
        self.args = self._init_args(args, **kwargs)
        self._model, self._tokenizer = self._init_model_and_tokenizer()
        self._generate_stream_func = None
        # self.chatio = SimpleChatIO() if self.args.style == 'simple' else RichChatIO()
        self.chatio = fsapi.SimpleChatIO()

    def _init_args(self, args, **kwargs):
        default_args = VicunaArgs()
        if args is not None:
            default_args.__dict__.update(args.__dict__)
        default_args.__dict__.update(**kwargs)
        return default_args
    
    def _init_model_and_tokenizer(self):
        if self.args.lazy_loading:
            return None, None
        return self.load_model()


@dataclasses.dataclass
class VicunaArgs:
    model_path: str = '/data/zzd/weights/vicuna/vicuna-7b-v1.5-16k'  # "The path to the weights. This can be a local folder or a Hugging Face repo ID."
    # model_path: str = '/dg_workfs/Beijing-CC/zdzhang/DghpcData/weights/vicuna/vicuna-7b-v1.5-16k'
    # model_path: str = '/dg_workfs/Beijing-CC/zdzhang/DghpcData/weights/weights/vicuna/vicuna-7b' 
    device: str = "cuda"  # The device type, i.e. ["cpu", "cuda", "mps", "npu"]
    gpus: int = None  # A single GPU like 1 or multiple GPUs like 0,2
    num_gpus: int = 1  # The number of GPUs to use
    max_gpu_memory: str = None  # The maximum memory per gpu. Use a string like '13Gib'
    load_8bit: bool = False  # Use 8-bit quantization
    cpu_offloading: bool = False  # Only when using 8-bit quantization: Offload excess weights to the CPU that don't fit on the GPU
    conv_template: str = None  # Conversation prompt template.
    conv_system_msg: str = None  # Conversation system message.
    temperature: float = 0.7  # The temperature for sampling
    max_new_tokens: int = 512  # The maximum number of tokens to generate
    lazy_loading: bool = True  # Load the model and tokenizer lazily
    
    repetition_penalty: float = 1.0  # The repetition penalty
    no_history: bool = False 
    style: str = field(default="simple", metadata={"help": "Display style.", "choices": ["simple", "rich", "programmatic"]})
    multiline: bool = field(default=False, metadata={"help": "Enable multiline input. Use ESC+Enter for newline."})
    mouse: bool = field(default=False, metadata={"help": "Enable mouse support for cursor positioning."})
    judge_sent_end: bool = field(default=False, metadata={"help": "Whether enable the correction logic that interrupts the output of sentences due to EOS."})
    debug: bool = field(default=False, metadata={"help": "Print useful debug information (e.g., prompts)"})
    
    revision: str = field(default="main", metadata={"help": "Hugging Face Hub model revision identifier"})
    dtype: str = field(default=None, metadata={"help": "Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.", "choices": ["float32", "float16", "bfloat16"]})
    gptq_skpt: str = field(default=None, metadata={"help": "Used for GPTQ. The path to the local GPTQ checkpoint."})
    gptq_wbits: int = field(default=16, metadata={"help": "Used for GPTQ. #bits to use for quantization"})
    gptq_groupsize: int = field(default=-1, metadata={"help": "Used for GPTQ. Groupsize to use for quantization; default uses full row."})
    gptq_act_order: bool = field(default=False, metadata={"help": "Used for GPTQ. Whether to apply the activation order GPTQ heuristic"})
    awq_ckpt: str = field(default=None, metadata={"help": "Used for AWQ. Load quantized model. The path to the local AWQ checkpoint."})
    awq_wbits: int = field(default=16, metadata={"help": "Used for AWQ. #bits to use for AWQ quantization", "choices": [4, 16]})
    awq_groupsize: int = field(default=-1, metadata={"help": "Used for AWQ. Groupsize to use for AWQ quantization; default uses full row."})
    enable_exllama: bool = field(default=False, metadata={"help": "Used for exllamabv2. Enable exllamaV2 inference framework."})
    exllama_max_seq_len: int = field(default=4096, metadata={"help": "Used for exllamabv2. Max sequence length to use for exllamav2 framework; default 4096 sequence length."})
    exllama_gpu_split: str = field(default=None, metadata={"help": "Used for exllamabv2. Comma-separated list of VRAM (in GB) to use per GPU. Example: 20,7,7"})
    
if __name__ == '__main__':
    args = hai.parse_args_into_dataclasses(VicunaArgs)
    # args.model_path = f'/data/zzd/vicuna/xiwu-13b-20230503'
    # args.model_path = f'/data/zzd/xiwu/xiwu-13b-20230509'
    # args.model_path = "/data/zzd/vicuna/vicuna-7b"
    # args.lazy_loading = False
    
    chatbot = Vicuna(args)
    prompts = ['who are you?', '你是谁', '你好', '你能做什么']
    # prompts = prompts[:1]
    for prompt in prompts:
        print(f'User: {prompt}')
        ret = chatbot.continuous_inference(prompt)
        for i in ret:
            sys.stdout.write(i)
            sys.stdout.flush()
            # print([i])
        print()
    

    


