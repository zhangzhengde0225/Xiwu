
import os, sys
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
import warnings
import math
import psutil
from functools import cache
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

import torch
from xiwu import CONST
from xiwu.apis.fastchat_api import *
from ..base.base_adapter import XBaseModelAdapter
from ..adapters.xiwu_adapter import XiwuAdapter
from ..adapters.vicuna_adapter import VicunaAdapter

peft_share_base_weights = (
    os.environ.get("PEFT_SHARE_BASE_WEIGHTS", "false").lower() == "true"
)

class XAssembler():

    def __init__(self):
        self.model_adapters: List[BaseModelAdapter] = model_adapters
        self.conv_templates: List[Conversation] = conv_templates

        self._post_register()

    def _post_register(self):
        self.register_model_adapter(XiwuAdapter, index=0)  # 注册适配器
        self.register_model_adapter(VicunaAdapter, index=0)

        self.register_conv_template(XiwuAdapter.conv, override=False)
        self.register_conv_template(VicunaAdapter.conv, override=True)
        
    def register_model_adapter(self, cls: BaseModelAdapter, index: Optional[int] = None):
        """Register a model adapter."""
        if index is not None:
            self.model_adapters.insert(index, cls())
        else:
            self.model_adapters.append(cls())

    def register_conv_template(self, template: Conversation, override: bool = False):
        """Register a new conversation template."""
        if not override:
            assert (
                template.name not in conv_templates
            ), f"{template.name} has been registered."

        self.conv_templates[template.name] = template

    @cache
    def get_model_adapter(self, model_path: str) -> BaseModelAdapter:
        """Get a model adapter for a model_path."""
        model_path_basename = os.path.basename(os.path.normpath(model_path))

        # Try the basename of model_path at first
        for adapter in self.model_adapters:
            if adapter.match(model_path_basename) and type(adapter) != BaseModelAdapter:
                return adapter

        # Then try the full path
        for adapter in self.model_adapters:
            if adapter.match(model_path):
                return adapter

        raise ValueError(f"No valid model adapter for {model_path}")
    
    def get_conv_template(self, name: str) -> Conversation:
        """Get a conversation template by name."""
        return self.conv_templates[name].copy()

    def get_conversation_template(self, model_path: str) -> Conversation:
        """Automatically get a conversation template for a model_path."""
        adapter = self.get_model_adapter(model_path)
        return adapter.get_default_conv_template()

    def load_model(self,
        model_path: str,
        device: str = "cuda",
        num_gpus: int = 1,
        max_gpu_memory: Optional[str] = None,
        dtype: Optional[torch.dtype | str] = None,
        load_8bit: bool = False,
        cpu_offloading: bool = False,
        gptq_config: Optional[GptqConfig] = None,
        awq_config: Optional[AWQConfig] = None,
        exllama_config: Optional[ExllamaConfig] = None,
        xft_config: Optional[XftConfig] = None,
        revision: str = "main",
        debug: bool = False,
    ): 
        """Load a model from Hugging Face."""
        import accelerate
        if isinstance(dtype, str):
            dtype = str_to_torch_dtype(dtype)

        # get model adapter
        adapter = self.get_model_adapter(model_path)

        # Handle device mapping
        cpu_offloading = raise_warning_for_incompatible_cpu_offloading_configuration(
            device, load_8bit, cpu_offloading
        )
        if device == "cpu":
            kwargs = {"torch_dtype": torch.float32}
            if CONST.CPU_ISA in ["avx512_bf16", "amx"]:
                try:
                    import intel_extension_for_pytorch as ipex

                    kwargs = {"torch_dtype": torch.bfloat16}
                except ImportError:
                    warnings.warn(
                        "Intel Extension for PyTorch is not installed, it can be installed to accelerate cpu inference"
                    )
        elif device == "cuda":
            kwargs = {"torch_dtype": torch.float16}
            if num_gpus != 1:
                kwargs["device_map"] = "auto"
                if max_gpu_memory is None:
                    kwargs[
                        "device_map"
                    ] = "sequential"  # This is important for not the same VRAM sizes
                    available_gpu_memory = get_gpu_memory(num_gpus)
                    kwargs["max_memory"] = {
                        i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                        for i in range(num_gpus)
                    }
                else:
                    kwargs["max_memory"] = {i: max_gpu_memory for i in range(num_gpus)}
        elif device == "mps":
            kwargs = {"torch_dtype": torch.float16}
            import transformers

            version = tuple(int(v) for v in transformers.__version__.split("."))
            if version < (4, 35, 0):
                # NOTE: Recent transformers library seems to fix the mps issue, also
                # it has made some changes causing compatibility issues with our
                # original patch. So we only apply the patch for older versions.

                # Avoid bugs in mps backend by not using in-place operations.
                replace_llama_attn_with_non_inplace_operations()
        elif device == "xpu":
            kwargs = {"torch_dtype": torch.bfloat16}
            # Try to load ipex, while it looks unused, it links into torch for xpu support
            try:
                import intel_extension_for_pytorch as ipex
            except ImportError:
                warnings.warn(
                    "Intel Extension for PyTorch is not installed, but is required for xpu inference."
                )
        elif device == "npu":
            kwargs = {"torch_dtype": torch.float16}
            # Try to load ipex, while it looks unused, it links into torch for xpu support
            try:
                import torch_npu
            except ImportError:
                warnings.warn("Ascend Extension for PyTorch is not installed.")
        else:
            raise ValueError(f"Invalid device: {device}")

        if cpu_offloading:
            # raises an error on incompatible platforms
            from transformers import BitsAndBytesConfig

            if "max_memory" in kwargs:
                kwargs["max_memory"]["cpu"] = (
                    str(math.floor(psutil.virtual_memory().available / 2**20)) + "Mib"
                )
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit_fp32_cpu_offload=cpu_offloading
            )
            kwargs["load_in_8bit"] = load_8bit
        elif load_8bit:
            if num_gpus != 1:
                warnings.warn(
                    "8-bit quantization is not supported for multi-gpu inference."
                )
            else:
                model, tokenizer = adapter.load_compress_model(
                    model_path=model_path,
                    device=device,
                    torch_dtype=kwargs["torch_dtype"],
                    revision=revision,
                )
                if debug:
                    print(model)
                return model, tokenizer
        elif awq_config and awq_config.wbits < 16:
            assert (
                awq_config.wbits == 4
            ), "Currently we only support 4-bit inference for AWQ."
            model, tokenizer = load_awq_quantized(model_path, awq_config, device)
            if num_gpus != 1:
                device_map = accelerate.infer_auto_device_map(
                    model,
                    max_memory=kwargs["max_memory"],
                    no_split_module_classes=[
                        "OPTDecoderLayer",
                        "LlamaDecoderLayer",
                        "BloomBlock",
                        "MPTBlock",
                        "DecoderLayer",
                    ],
                )
                model = accelerate.dispatch_model(
                    model, device_map=device_map, offload_buffers=True
                )
            else:
                model.to(device)
            return model, tokenizer
        elif gptq_config and gptq_config.wbits < 16:
            model, tokenizer = load_gptq_quantized(model_path, gptq_config)
            if num_gpus != 1:
                device_map = accelerate.infer_auto_device_map(
                    model,
                    max_memory=kwargs["max_memory"],
                    no_split_module_classes=["LlamaDecoderLayer"],
                )
                model = accelerate.dispatch_model(
                    model, device_map=device_map, offload_buffers=True
                )
            else:
                model.to(device)
            return model, tokenizer
        elif exllama_config:
            model, tokenizer = load_exllama_model(model_path, exllama_config)
            return model, tokenizer
        elif xft_config:
            model, tokenizer = load_xft_model(model_path, xft_config)
            return model, tokenizer
        kwargs["revision"] = revision

        if dtype is not None:  # Overwrite dtype if it is provided in the arguments.
            kwargs["torch_dtype"] = dtype

        if os.environ.get("FASTCHAT_USE_MODELSCOPE", "False").lower() == "true":
            # download model from ModelScope hub,
            # lazy import so that modelscope is not required for normal use.
            try:
                from modelscope.hub.snapshot_download import snapshot_download

                if not os.path.exists(model_path):
                    model_path = snapshot_download(model_id=model_path, revision=revision)
            except ImportError as e:
                warnings.warn(
                    "Use model from www.modelscope.cn need pip install modelscope"
                )
                raise e

        # Load model
        model, tokenizer = adapter.load_model(model_path, kwargs)

        if (
            device == "cpu"
            and kwargs["torch_dtype"] is torch.bfloat16
            and CONST.CPU_ISA is not None
        ):
            model = ipex.optimize(model, dtype=kwargs["torch_dtype"])

        if (device == "cuda" and num_gpus == 1 and not cpu_offloading) or device in (
            "mps",
            "xpu",
            "npu",
        ):
            model.to(device)

        if device == "xpu":
            model = torch.xpu.optimize(model, dtype=kwargs["torch_dtype"], inplace=True)

        if debug:
            print(model)

        return model, tokenizer
    
    def adapt_local_path(self, model_path):
        if os.path.exists(f'{CONST.PRETRAINED_WEIGHTS_DIR}/{model_path}'):
            model_path = f'{CONST.PRETRAINED_WEIGHTS_DIR}/{model_path}'
        return model_path

    def load_config_from_pretrained(self, pretrained_model_name_or_path, **kwargs):
        """
        Load a configuration object from a pretrained model for training.
        """
        model_path = self.adapt_local_path(pretrained_model_name_or_path)
        config = AutoConfig.from_pretrained(
            model_path,
            **kwargs
        )
        return config
    
    def load_model_from_pretrained(self, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load a model from a pretrained model for training.
        """
        model_path = self.adapt_local_path(pretrained_model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            *model_args,
            **kwargs
        )
        return model
    
    def load_tokenizer_from_pretrained(self, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        Load a tokenizer from a pretrained model for training.
        """
        model_path = self.adapt_local_path(pretrained_model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            *inputs,
            **kwargs
        )
        return tokenizer

    def get_generate_stream_function(self, model: torch.nn.Module, model_path: str):
        """Get the generate_stream function for inference."""
        # 先从adapter里获取stream_function
        adapter = self.get_model_adapter(model_path)
        if hasattr(adapter, "generate_stream"):
            return adapter.generate_stream

        # 从fastchat中实现的stream_function获取
        from fastchat.serve.inference import generate_stream

        model_type = str(type(model)).lower()
        # such as: "<class 'transformers.models.llama.modeling_llama.llamaforcausallm'>"
        is_peft = "peft" in model_type
        is_chatglm = "chatglm" in model_type
        is_falcon = "rwforcausallm" in model_type
        is_codet5p = "codet5p" in model_type
        is_exllama = "exllama" in model_type
        is_xft = "xft" in model_type
        is_yuan = "yuan" in model_type

        if is_chatglm:
            return generate_stream_chatglm
        elif is_falcon:
            return generate_stream_falcon
        elif is_codet5p:
            return generate_stream_codet5p
        elif is_exllama:
            return generate_stream_exllama
        elif is_xft:
            return generate_stream_xft
        elif is_yuan:
            return generate_stream_yuan2

        elif peft_share_base_weights and is_peft:
            # Return a curried stream function that loads the right adapter
            # according to the model_name available in this context.  This ensures
            # the right weights are available.
            @torch.inference_mode()
            def generate_stream_peft(
                model,
                tokenizer,
                params: Dict,
                device: str,
                context_len: int,
                stream_interval: int = 2,
                judge_sent_end: bool = False,
            ):
                model.set_adapter(model_path)
                base_model_type = str(type(model.base_model.model))
                is_chatglm = "chatglm" in base_model_type
                is_falcon = "rwforcausallm" in base_model_type
                is_codet5p = "codet5p" in base_model_type
                is_exllama = "exllama" in base_model_type
                is_xft = "xft" in base_model_type
                is_yuan = "yuan" in base_model_type

                generate_stream_function = generate_stream
                if is_chatglm:
                    generate_stream_function = generate_stream_chatglm
                elif is_falcon:
                    generate_stream_function = generate_stream_falcon
                elif is_codet5p:
                    generate_stream_function = generate_stream_codet5p
                elif is_exllama:
                    generate_stream_function = generate_stream_exllama
                elif is_xft:
                    generate_stream_function = generate_stream_xft
                elif is_yuan:
                    generate_stream_function = generate_stream_yuan2
                for x in generate_stream_function(
                    model,
                    tokenizer,
                    params,
                    device,
                    context_len,
                    stream_interval,
                    judge_sent_end,
                ):
                    yield x

            return generate_stream_peft
        else:
            return generate_stream
