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
    Conversation,
    GptqConfig, AWQConfig, ExllamaConfig, XftConfig
)
from xiwu import CONST, ASSEMBLER
from xiwu.configs.configs import BaseArgs
from ..adapters.adapt_oai import OAIAdapter


@dataclass
class XBaseModelArgs(BaseArgs):  # 继承了
    pass


class XBaseModel:

    def __init__(self, args: BaseArgs=None, **kwargs) -> None:
        self.args = args or XBaseModelArgs()
        self.args = self._merge_args(**kwargs)
        self._model, self._tokenizer = self._init_model_and_tokenizer()
        self._generate_stream_func = None
        self.name = self._init_model_name()

    def _merge_args(self, **kwargs):
        self.args.__dict__.update(**kwargs)
        # default_args = VicunaArgs()
        # if args is not None:
        #     default_args.__dict__.update(args.__dict__)
        # default_args.__dict__.update(**kwargs)
        # return default_args
        return self.args
        

    def _init_model_and_tokenizer(self):
        if self.args.lazy_loading:
            return None, None
        return self.load_model()
    
    def _init_model_name(self):
        if self.args.model_name is not None:
            return self.args.model_name
        model_name = self.args.model_path.split("/")[-1]
        return model_name
    
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
        if os.path.exists(f'{CONST.PRETRAINED_WEIGHTS_DIR}/{model_path}'):
            return f'{CONST.PRETRAINED_WEIGHTS_DIR}/{model_path}'
        return model_path

    def load_model(self):
        args = self.args
        if args.gpus:
            if len(args.gpus.split(",")) < args.num_gpus:
                raise ValueError(
                    f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
                )
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
            os.environ["XPU_VISIBLE_DEVICES"] = args.gpus
        if args.enable_exllama:
            exllama_config = ExllamaConfig(
                max_seq_len=args.exllama_max_seq_len,
                gpu_split=args.exllama_gpu_split,
                cache_8bit=args.exllama_cache_8bit,
            )
        else:
            exllama_config = None
        if args.enable_xft:
            xft_config = XftConfig(
                max_seq_len=args.xft_max_seq_len,
                data_type=args.xft_dtype,
            )
            if args.device != "cpu":
                print("xFasterTransformer now is only support CPUs. Reset device to CPU")
                args.device = "cpu"
        else:
            xft_config = None

        model_path = self.search_local_model(args.model_path)
        
        model, tokenizer = ASSEMBLER.load_model(
            model_path=model_path,
            device=args.device,
            num_gpus=args.num_gpus,
            max_gpu_memory=args.max_gpu_memory,
            dtype=args.dtype,
            load_8bit=args.load_8bit,
            cpu_offloading=args.cpu_offloading,
            gptq_config=GptqConfig(
                    ckpt=args.gptq_ckpt or args.model_path,
                    wbits=args.gptq_wbits,
                    groupsize=args.gptq_groupsize,
                    act_order=args.gptq_act_order,
                ),
            awq_config=AWQConfig(
                    ckpt=args.awq_ckpt or args.model_path,
                    wbits=args.awq_wbits,
                    groupsize=args.awq_groupsize,
                ),
            exllama_config=exllama_config,
            xft_config=xft_config,
            revision=args.revision,
            debug=args.debug
        )
       
        return model, tokenizer
    
    def oai_messages2prompt(self, messages) -> str:
        """这种方法不将信息缓存到conv里，适用于单次对话，多轮需要外部自己维护对话信息"""
        conv = self.get_conv().copy()
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                if content is not None and len(content) > 0:
                    # conv.system = content
                    conv.system_message = content
            elif role == 'user':
                conv.append_message(conv.roles[0], content)
            elif role == 'assistant':
                conv.append_message(conv.roles[1], content)
            else:
                raise ValueError(f"Unknown role: {role}, only support 'system', 'user', 'assistant'")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        del conv
        return prompt
    
    def messages2conv(self, messages) -> Conversation:
        """
        注意，这种方法是缓存信息到conv里的，适用于后台自动多轮会话
        """
        conv = self.get_conv()
        # 读取messages中的系统消息
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                if content is not None and len(content) > 0:
                    conv.system = content
            elif role == 'user':
                conv.append_message(conv.roles[0], content)
            elif role == 'assistant':
                conv.append_message(conv.roles[1], content)
            else:
                raise ValueError(f"Unknown role: {role}, only support 'system', 'user', 'assistant'")
        conv.append_message(conv.roles[1], None)
        return conv
    
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
        max_new_tokens = kwargs.pop("max_new_tokens", args.max_new_tokens)
        stop_str = kwargs.pop("stop_str", args.stop_str)
        stop_token_ids = kwargs.pop("stop_token_ids", None)
        repetition_penalty = kwargs.pop("repetition_penalty", args.repetition_penalty)
        judge_sent_end = kwargs.pop("judge_sent_end", args.judge_sent_end)
        stream = kwargs.pop("stream", args.stream)
        context_len = get_context_length(self.model.config)
        echo = kwargs.pop("echo", args.echo)
        logprobs = kwargs.pop("logprobs", args.logprobs)
        top_p = kwargs.pop("top_p", args.top_p)
        top_k = kwargs.pop("top_k", args.top_k)

        gen_params = {
            "model": args.model_path,  # 其实没用
            "prompt": prev_text,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_new_tokens": max_new_tokens,
            "stop": stop_str,
            "stop_token_ids": stop_token_ids,
            "repetition_penalty": repetition_penalty,
            "echo": echo,
            "logprobs": logprobs,
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
        
        if args.oai_format:
            return OAIAdapter.convert_output_to_oai_format(
                output_stream,
                model_name=self.name, 
                stream=stream,
                )
        
        return output_stream
        # return self.chatio.stream_output(output_stream)

    
