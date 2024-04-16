

from typing import Optional, List
import os, sys
from pathlib import Path
from dataclasses import dataclass, field
from .constant import RUNS_DIR, DATASETS_DIR
import transformers

here = Path(__file__).parent


@dataclass
class BaseArgs:
    ## Initialize
    model_path: str = field(default="xiwu/xiwu-13b-20230509", metadata={"help": "The path to the weights. This can be a local folder or a Hugging Face repo ID."})
    model_name: Optional[str] = field(default=None, metadata={"help": "The model name, will be contained in the output, if None, the model path will be used."})
    device: str = field(default="cuda", metadata={"help": "The device type", "choices": ["cpu", "cuda", "mps", "xpu", "npu"]})
    gpus: Optional[str] = field(default=None, metadata={"help": "A single GPU like 1 or multiple GPUs like 0,2"})
    num_gpus: int = field(default=1, metadata={"help": "Number of GPUs to use"})
    max_gpu_memory: Optional[str] = field(default=None, metadata={"help": "The maximum memory per GPU for storing model weights. Use a string like '13Gib'"})
    load_8bit: bool = field(default=False, metadata={"help": "Use 8-bit quantization"})
    dtype: Optional[str] = field(default=None, metadata={"help": "Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.", "choices": ["float32", "float16", "bfloat16"]})
    lazy_loading: bool = field(default=True, metadata={"help": "Load the model and tokenizer lazily"})
    revision: str = field(default="main", metadata={"help": "Hugging Face Hub model revision identifier"})
    
    ## Run-time
    cpu_offloading: bool = field(default=False, metadata={"help": "Only when using 8-bit quantization: Offload excess weights to the CPU that don't fit on the GPU"})
    conv_template: Optional[str] = field(default=None, metadata={"help": "Conversation prompt template."})
    conv_system_msg: Optional[str] = field(default=None, metadata={"help": "Conversation system message."})
    no_history: bool = field(default=False, metadata={"help": "Do not keep history of conversations."})
    
    # Generate Params
    temperature: float = field(default=0.7, metadata={"help": "Temperature for text generation."})
    top_p: float = field(default=1.0, metadata={"help": "Top-p for text generation."})
    top_k: int = field(default=-1, metadata={"help": "Top-k for text generation."})
    repetition_penalty: float = field(default=1.0, metadata={"help": "Repetition penalty for text generation."})
    max_new_tokens: int = field(default=512, metadata={"help": "Maximum number of new tokens to generate."})
    logprobs: int = field(default=None, metadata={"help": "Include log probabilities in the output."})
    echo: bool = field(default=True, metadata={"help": "Echo the input text."})
    stop_str: str = field(default=None, metadata={"help": "Stop generation at this string. can be single string or a list of strings."})
    # stop_token_ids: Optional[list] = field(default=None, metadata={"help": "Stop generation at this token ID, default is from Tokenizer"})
    judge_sent_end: bool = field(default=False, metadata={"help": "Whether enable the correction logic that interrupts the output of sentences due to EOS."})
    stream: bool = field(default=False, metadata={"help": "Enable streaming output."})
    oai_format: bool = field(default=True, metadata={"help": "Enable OAI format for streaming output."})

    # For CLI
    style: str = field(default="simple", metadata={"help": "Display style.", "choices": ["simple", "rich", "programmatic"]})
    multiline: bool = field(default=False, metadata={"help": "Enable multiline input. Use ESC+Enter for newline."})
    mouse: bool = field(default=False, metadata={"help": "[Rich Style]: Enable mouse support for cursor positioning."})
    debug: bool = field(default=False, metadata={"help": "Print useful debug information (e.g., prompts)"})

    # Quantization
    gptq_ckpt: Optional[str] = field(default=None, metadata={"help": "Used for GPTQ. The path to the local GPTQ checkpoint."})
    gptq_wbits: int = field(default=16, metadata={"help": "Used for GPTQ. #bits to use for quantization", "choices": [2, 3, 4, 8, 16]})
    gptq_groupsize: int = field(default=-1, metadata={"help": "Used for GPTQ. Groupsize to use for quantization; default uses full row."})
    gptq_act_order: bool = field(default=False, metadata={"help": "Used for GPTQ. Whether to apply the activation order GPTQ heuristic"})
    awq_ckpt: Optional[str] = field(default=None, metadata={"help": "Used for AWQ. Load quantized model. The path to the local AWQ checkpoint."})
    awq_wbits: int = field(default=16, metadata={"help": "Used for AWQ. #bits to use for AWQ quantization", "choices": [4, 16]})
    awq_groupsize: int = field(default=-1, metadata={"help": "Used for AWQ. Groupsize to use for AWQ quantization; default uses full row."})
    enable_exllama: bool = field(default=False, metadata={"help": "Used for exllamabv2. Enable exllamaV2 inference framework."})
    exllama_max_seq_len: int = field(default=4096, metadata={"help": "Used for exllamabv2. Max sequence length to use for exllamav2 framework; default 4096 sequence length."})
    exllama_gpu_split: Optional[str] = field(default=None, metadata={"help": "Used for exllamabv2. Comma-separated list of VRAM (in GB) to use per GPU. Example: 20,7,7"})
    exllama_cache_8bit: bool = field(default=False, metadata={"help": "Used for exllamabv2. Use 8-bit cache to save VRAM."})
    enable_xft: bool = field(default=False, metadata={"help": "Used for xFasterTransformer. Enable xFasterTransformer inference framework."})
    xft_max_seq_len: int = field(default=4096, metadata={"help": "Used for xFasterTransformer. Max sequence length to use for xFasterTransformer framework; default 4096 sequence length."})
    xft_dtype: Optional[str] = field(default=None, metadata={"help": "Override the default dtype. If not set, it will use bfloat16 for first token and float16 next tokens on CPU.", "choices": ["fp16", "bf16", "int8", "bf16_fp16", "bf16_int8"]})
    
    
@dataclass
class ModelArgs:
    model_name_or_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.5-16k")
    trust_remote_code: bool = field(default=False, metadata={"help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"},)
    padding_side: str = field(default="right", metadata={"help": "The padding side in tokenizer"})


@dataclass
class DataArgs:
    data_path: str = field(default=f'{DATASETS_DIR}/hep_text_v1.0', metadata={"help": "Path to the training data."})
    eval_data_path: str = field(default=None, metadata={"help": "Path to the evaluation data."})
    lazy_preprocess: bool = True

@dataclass
class TrainingArgs(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=576, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},)
    output_dir: str = field(default=RUNS_DIR, metadata={"help": "The output directory where the model predictions and checkpoints will be written."})
    # fsdp: str = "full_shard auto_wrap offload"
    # fsdp_config: str = f"{here}/fsdp_config.json"



