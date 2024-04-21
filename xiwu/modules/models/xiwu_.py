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
from xiwu.apis import fastchat_api as fsapi
from xiwu.apis.fastchat_api import *
from xiwu.modules.base.base_model import XBaseModel
from xiwu.apis.xiwu_api import BaseArgs



class Xiwu(XBaseModel):
    def __init__(self, args=None, **kwargs):
        super().__init__()
        self.args = self._init_args(args, **kwargs)
        self._model, self._tokenizer = self._init_model_and_tokenizer()
        self._generate_stream_func = None
        self.chatio = fsapi.SimpleChatIO()

    def _init_args(self, args, **kwargs):
        default_args = XiwuArgs()
        if args is not None:
            default_args.__dict__.update(args.__dict__)
        default_args.__dict__.update(kwargs)
        return default_args
    
    def _init_model_and_tokenizer(self):
        if self.args.lazy_loading:
            return None, None
        return self.load_model()     


@dataclasses.dataclass
class XiwuArgs(BaseArgs):
    model_path: str = f'chathep/chathep-13b-20230509-2'


if __name__ == '__main__':
    args = hai.parse_args_into_dataclasses(XiwuArgs)
    # args.model_path = f'/data/zzd/vicuna/xiwu-13b-20230503'
    # args.model_path = f'/data/zzd/xiwu/xiwu-13b-20230509'
    # args.model_path = "/data/zzd/vicuna/vicuna-7b"
    # args.lazy_loading = False
    
    chatbot = Xiwu(args)
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
    

    
