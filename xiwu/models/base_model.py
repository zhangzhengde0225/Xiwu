"""
Xiwu的基础模型，实现一些简单的功能
"""
from typing import Any
from dataclasses import dataclass, field
import os, sys
from pathlib import Path
here = Path(__file__).parent

try:
    from xiwu.version import __version__
except:
    sys.path.insert(1, str(here.parent.parent))
    from xiwu.version import __version__
from xiwu.apis import fastchat_api as fsapi
from xiwu.apis.fastchat_api import (
    get_generate_stream_function,
    load_model,
    get_conv_template,
    get_conversation_template,
    get_context_length,
    Conversation
)
from xiwu import PRETRAINED_WEIGHTS_DIR


@dataclass
class XBaseModelArgs:
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
    

class XBaseModel:
    args: XBaseModelArgs
    model: Any

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        if self._model is None:
            self._model, self._tokenizer = self.load_model()
        return self._model
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._model, self._tokenizer = self.load_model()
        return self._tokenizer

    @property
    def generate_stream_func(self):
        return get_generate_stream_function(self.model, self.args.model_path)

    def search_local_model(self, model_path):
        """自动搜索本地权重，如果存在则返回本地路径，否则返回原路径"""
        if os.path.exists(f'{PRETRAINED_WEIGHTS_DIR}/{model_path}'):
            return f'{PRETRAINED_WEIGHTS_DIR}/{model_path}'
        return model_path

    def load_model(self):
        args = self.args
        model_path = self.search_local_model(args.model_path)
        model, tokenizer = load_model(
             model_path,
             args.device, 
             args.num_gpus, 
             max_gpu_memory=args.max_gpu_memory, 
             load_8bit=args.load_8bit, 
             cpu_offloading=args.cpu_offloading,
            #  gptq_config=gptq_config,
            #  awq_config=awq_config,
            #  revision=revision,
             debug=args.debug
             )
        return model, tokenizer
    
    def get_conv(self):
        conv_template = self.args.conv_template
        if conv_template:
            return get_conv_template(conv_template)
        else:
            return get_conversation_template(self.args.model_path)
    
    def get_prompt_by_conv(self, conv: Conversation):
        # is_chatglm = "chatglm" in self.args.model_path.lower()
        is_chatglm = "chatglm" in str(type(self.model)).lower()
        if is_chatglm:
            prompt = conv.messages[conv.offset:]
        else:
            prompt = conv.get_prompt()
        return prompt

    def continuous_inference(self, prompt, **kwargs):
        """
        由用户输入的prompt
        """
        conv = self.get_conv()  # 构建Conversation对象
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        # prompt = self.get_prompt_by_conv(conv)
        prompt = conv.get_prompt()
        output_stream = self.inference(
            prompt,
            stop_str=conv.stop_str,
            stop_token_ids=conv.stop_token_ids,
            **kwargs
            )
        # outputs = stream_output(output_stream)  # 打印
        # conv.messages[-1][-1] = outputs.strip()
        return output_stream

    def inference(self, prev_text, **kwargs):
        """
        注意！！此处的prev_text包含所有的对话内容和系统提示，需由Conversation对象生成
        """
        args = self.args
        temperature = kwargs.pop("temperature", args.temperature)

        stop_str = kwargs.pop("stop_str", None)
        stop_token_ids = kwargs.pop("stop_token_ids", None)
        repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        judge_sent_end = kwargs.pop("judge_sent_end", False)
        context_len = get_context_length(self.model.config)

        gen_params = {
            "model": args.model_path,  # 其实没用
            "prompt": prev_text,
            "temperature": temperature,
            "max_new_tokens": args.max_new_tokens,
            "stop": stop_str,
            "stop_token_ids": stop_token_ids,
            "repetition_penalty": repetition_penalty,
            "echo": False,
        }

        output_stream = self.generate_stream_func(
            self.model, 
            self.tokenizer, 
            gen_params, 
            args.device, 
            context_len=context_len,
            judge_sent_end=judge_sent_end,
            # stream_interval=2,    
            )
        
        return self.chatio.stream_output(output_stream)
    
