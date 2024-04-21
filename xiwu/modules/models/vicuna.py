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
from xiwu.modules.base.base_model import XBaseModel
from xiwu import BaseArgs
from xiwu.apis import fastchat_api as fsapi
from xiwu.apis.fastchat_api import *

class Vicuna(XBaseModel):
    def __init__(self, args: BaseArgs=None, **kwargs):
        super().__init__(args=args, **kwargs)
        self.chatio = fsapi.SimpleChatIO()
    
    def inference(self, prev_text, **kwargs):
        chatcmpl: dict = super().inference(prev_text=prev_text, **kwargs)
        return chatcmpl
        # return self.chatio.stream_output(res)


@dataclasses.dataclass
class VicunaArgs(BaseArgs):
    model_path: str = '/data/zzd/weights/vicuna/vicuna-7b-v1.5-16k'  # "The path to the weights. This can be a local folder or a Hugging Face repo ID."
    # model_path: str = '/dg_workfs/Beijing-CC/zdzhang/DghpcData/weights/vicuna/vicuna-7b-v1.5-16k'
    # model_path: str = '/dg_workfs/Beijing-CC/zdzhang/DghpcData/weights/weights/vicuna/vicuna-7b' 
    
    
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
    

    


