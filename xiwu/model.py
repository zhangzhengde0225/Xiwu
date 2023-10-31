import math
import os
import re
import sys
from typing import Dict, List, Optional
import warnings

if sys.version_info >= (3, 9):
    from functools import cache
else:
    from functools import lru_cache as cache

import accelerate
import psutil
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaForCausalLM,
    T5Tokenizer,
)
import dataclasses
import gc
import glob
import os

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from huggingface_hub import snapshot_download
import torch
from torch import Tensor
from torch.nn import functional as F
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    AutoModelForSeq2SeqLM,
)

from conversation import*

class BaseModelAdapter:
    """The base and the default model adapter."""

    use_fast_tokenizer = True

    def match(self, model_path: str):
        return True

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=self.use_fast_tokenizer,
                revision=revision,
                trust_remote_code=True,
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, use_fast=False, revision=revision, trust_remote_code=True
            )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
            )
        except NameError:
            model = AutoModel.from_pretrained(
                model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
            )
        return model, tokenizer

    def load_compress_model(self, model_path, device, torch_dtype, revision="main"):
        return load_compress_model(
            model_path,
            device,
            torch_dtype,
            use_fast=self.use_fast_tokenizer,
            revision=revision,
        )

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("one_shot")

def get_compressed_list(module, prefix=""):
    compressed_list = []
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.Linear:
            full_name = (
                f"{prefix}.{attr_str}.weight" if prefix else f"{attr_str}.weight"
            )
            compressed_list.append(full_name)
    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        for each in get_compressed_list(child, child_prefix):
            compressed_list.append(each)
    return compressed_list

from dataclasses import dataclass
@dataclass
class CompressionConfig:
    """Group-wise quantization."""
    num_bits: int
    group_size: int
    group_dim: int
    symmetric: bool
    enabled: bool = True

default_compression_config = CompressionConfig(
    num_bits=8, group_size=256, group_dim=1, symmetric=True, enabled=True
)

def compress(tensor, config):
    """Simulate group-wise quantization."""
    if not config.enabled:
        return tensor

    group_size, num_bits, group_dim, symmetric = (
        config.group_size,
        config.num_bits,
        config.group_dim,
        config.symmetric,
    )
    assert num_bits <= 8

    original_shape = tensor.shape
    num_groups = (original_shape[group_dim] + group_size - 1) // group_size
    new_shape = (
        original_shape[:group_dim]
        + (num_groups, group_size)
        + original_shape[group_dim + 1 :]
    )

    # Pad
    pad_len = (group_size - original_shape[group_dim] % group_size) % group_size
    if pad_len != 0:
        pad_shape = (
            original_shape[:group_dim] + (pad_len,) + original_shape[group_dim + 1 :]
        )
        tensor = torch.cat(
            [tensor, torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)],
            dim=group_dim,
        )
    data = tensor.view(new_shape)

    # Quantize
    if symmetric:
        B = 2 ** (num_bits - 1) - 1
        scale = B / torch.max(data.abs(), dim=group_dim + 1, keepdim=True)[0]
        data = data * scale
        data = data.clamp_(-B, B).round_().to(torch.int8)
        return data, scale, original_shape
    else:
        B = 2**num_bits - 1
        mn = torch.min(data, dim=group_dim + 1, keepdim=True)[0]
        mx = torch.max(data, dim=group_dim + 1, keepdim=True)[0]

        scale = B / (mx - mn)
        data = data - mn
        data.mul_(scale)

        data = data.clamp_(0, B).round_().to(torch.uint8)
        return data, mn, scale, original_shape


def load_compress_model(model_path, device, torch_dtype, use_fast, revision="main"):
    # partially load model
    # `use_fast=True`` is not supported for some models.
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=use_fast, revision=revision, trust_remote_code=True
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=~use_fast, revision=revision, trust_remote_code=True
        )
    with init_empty_weights():
        # `trust_remote_code` should be set as `True` for both AutoConfig and AutoModel
        config = AutoConfig.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            revision=revision,
        )
        # some models are loaded by AutoModel but not AutoModelForCausalLM,
        # such as chatglm, chatglm2
        try:
            # google/flan-* models are based on an AutoModelForSeq2SeqLM.
            if "T5Config" in str(type(config)):
                model = AutoModelForSeq2SeqLM.from_config(
                    config, trust_remote_code=True
                )
            else:
                model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        except NameError:
            model = AutoModel.from_config(config, trust_remote_code=True)
        linear_weights = get_compressed_list(model)
    if os.path.exists(model_path):
        # `model_path` is a local folder
        base_pattern = os.path.join(model_path, "pytorch_model*.bin")
    else:
        # `model_path` is a cached Hugging Face repo
        # We don't necessarily need to download the model' repo again if there is a cache.
        # So check the default huggingface cache first.
        model_path_temp = os.path.join(
            os.path.expanduser("~"),
            ".cache/huggingface/hub",
            "models--" + model_path.replace("/", "--"),
            "snapshots/",
        )
        downloaded = False
        if os.path.exists(model_path_temp):
            temp_last_dir = os.listdir(model_path_temp)[-1]
            model_path_temp = os.path.join(model_path_temp, temp_last_dir)
            base_pattern = os.path.join(model_path_temp, "pytorch_model*.bin")
            files = glob.glob(base_pattern)
            if len(files) > 0:
                downloaded = True

        if downloaded:
            model_path = model_path_temp
        else:
            model_path = snapshot_download(model_path, revision=revision)
        base_pattern = os.path.join(model_path, "pytorch_model*.bin")

    files = glob.glob(base_pattern)
    if len(files) == 0:
        raise ValueError(
            f"Cannot find any model weight files. "
            f"Please check your (cached) weight path: {model_path}"
        )

    compressed_state_dict = {}
    for filename in tqdm(files):
        tmp_state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
        for name in tmp_state_dict:
            if name in linear_weights:
                tensor = tmp_state_dict[name].to(device, dtype=torch_dtype)
                compressed_state_dict[name] = compress(
                    tensor, default_compression_config
                )
            else:
                compressed_state_dict[name] = tmp_state_dict[name].to(
                    device, dtype=torch_dtype
                )
            tmp_state_dict[name] = None
            tensor = None
            gc.collect()
            torch.cuda.empty_cache()
            if device == "xpu":
                torch.xpu.empty_cache()
            if device == "npu":
                torch.npu.empty_cache()

    for name in model.state_dict():
        if name not in linear_weights:
            set_module_tensor_to_device(
                model, name, device, value=compressed_state_dict[name]
            )
    apply_compressed_weight(model, compressed_state_dict, device)

    if torch_dtype == torch.float16:
        model.half()
    model.to(device)
    model.eval()

    return model, tokenizer

def apply_compressed_weight(module, compressed_state_dict, target_device, prefix=""):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.Linear:
            full_name = (
                f"{prefix}.{attr_str}.weight" if prefix else f"{attr_str}.weight"
            )
            setattr(
                module,
                attr_str,
                CLinear(
                    compressed_state_dict[full_name], target_attr.bias, target_device
                ),
            )
    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        apply_compressed_weight(
            child, compressed_state_dict, target_device, child_prefix
        )

class CLinear(nn.Module):
    """Compressed Linear Layer."""

    def __init__(self, weight=None, bias=None, device=None):
        super().__init__()
        if weight is None:
            self.weight = None
        elif isinstance(weight, Tensor):
            self.weight = compress(weight.data.to(device), default_compression_config)
        else:
            self.weight = weight
        self.bias = bias

    def forward(self, input: Tensor) -> Tensor:
        weight = decompress(self.weight, default_compression_config)
        if self.bias is None:
            return F.linear(input.to(weight.dtype), weight)
        return F.linear(input.to(weight.dtype), weight, self.bias.to(weight.dtype))

def decompress(packed_data, config):
    """Simulate group-wise dequantization."""
    if not config.enabled:
        return packed_data

    group_size, num_bits, group_dim, symmetric = (
        config.group_size,
        config.num_bits,
        config.group_dim,
        config.symmetric,
    )

    # Dequantize
    if symmetric:
        data, scale, original_shape = packed_data
        data = data / scale
    else:
        data, mn, scale, original_shape = packed_data
        data = data / scale
        data.add_(mn)

    # Unpad
    pad_len = (group_size - original_shape[group_dim] % group_size) % group_size
    if pad_len:
        padded_original_shape = (
            original_shape[:group_dim]
            + (original_shape[group_dim] + pad_len,)
            + original_shape[group_dim + 1 :]
        )
        data = data.reshape(padded_original_shape)
        indices = [slice(0, x) for x in original_shape]
        return data[indices].contiguous()
    else:
        return data.view(original_shape)

def remove_parent_directory_name(model_path):
    """Remove parent directory name."""
    if model_path[-1] == "/":
        model_path = model_path[:-1]
    return model_path.split("/")[-1]

class VicunaAdapter(BaseModelAdapter):
    "Model adapater for Vicuna models (e.g., lmsys/vicuna-7b-v1.3)" ""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "vicuna" in model_path.lower()
     
    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=self.use_fast_tokenizer, revision=revision
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        )
        self.raise_warning_for_old_weights(model)
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        if "v0" in remove_parent_directory_name(model_path):
            return get_conv_template("one_shot")
        return get_conv_template("vicuna_v1.1")

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

# A global registry for all model adapters
# TODO (lmzheng): make it a priority queue.
model_adapters: List[BaseModelAdapter] = []

def register_model_adapter(cls):
    """Register a model adapter."""
    model_adapters.append(cls())
register_model_adapter(VicunaAdapter)


@cache
def get_model_adapter(model_path: str) -> BaseModelAdapter:
    """Get a model adapter for a model_path."""
    model_path_basename = os.path.basename(os.path.normpath(model_path))

    # print(f'model_path_basename: {model_path_basename}')
    # print(model_adapters)
    # Try the basename of model_path at first
    for adapter in model_adapters:
        if adapter.match(model_path_basename) and type(adapter) != BaseModelAdapter:
            
            return adapter

    # Then try the full path
    for adapter in model_adapters:
        if adapter.match(model_path):
            # print(f'adapter: {adapter}')
            return adapter

    raise ValueError(f"No valid model adapter for {model_path}")

def get_conversation_template(model_path: str) -> Conversation:
    """Get the default conversation template."""
    adapter = get_model_adapter(model_path)
    return adapter.get_default_conv_template(model_path)

def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()

register_conv_template(
    Conversation(
        name="vicuna_v1.1",
        system_message="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("USER", "ASSISTANT"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

