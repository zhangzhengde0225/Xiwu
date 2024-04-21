import os, sys
from pathlib import Path
here = Path(__file__).parent

from typing import Optional, Generator, Union

from dataclasses import dataclass, field
try:
    from xiwu.version import __version__
except:
    sys.path.insert(1, str(here.parent.parent.parent))
    from xiwu.version import __version__
from xiwu import BaseArgs, XBaseModel
from xiwu.modules.models.xiwu_ import Xiwu
from xiwu.apis.fastchat_api import Conversation
import hepai
from hepai import BaseWorkerModel
from xiwu.modules.base.xmodel import XModel
from xiwu.utils import general



class WorkerModel(BaseWorkerModel):
    def __init__(self, args: BaseArgs, **kwargs):
        self.name = self._init_worker_name(args)
        self.xmodel: XBaseModel = XModel(args=args, **kwargs)

    def _init_worker_name(self, args: BaseArgs):
        if args.model_name is not None:
            return args.model_name
        if args.model_path is not None:
            model_name = args.model_path.split("/")[-2::]
            model_name = "/".join(model_name)
            return model_name
        raise ValueError("model_name or model_path must be provided")

    @BaseWorkerModel.auto_stream  # 自动将各种类型的输出转为流式输
    def inference(self, **kwargs):
        messages= kwargs.pop('messages', None)
        conv = self.xmodel.messages2conv(messages)
        prompt = self.xmodel.get_prompt_by_conv(conv)
        chatcmpl: dict = self.xmodel.inference(prev_text=prompt, **kwargs)
        return chatcmpl
    
    def chat_completions(self, messages: list, **kwargs) -> Generator:
        # conv = self.xmodel.messages2conv(messages)
        prompt = self.xmodel.oai_messages2prompt(messages=messages)
        chatcmpl: dict = self.xmodel.inference(prev_text=prompt, **kwargs)
        return chatcmpl



@dataclass
class ModelArgs(BaseArgs):  # (1) 实现WorkerModel
    model_path: str = field(default="hepai/xiwu-13b-16k-20240417", metadata={"help": "The path to the weights. This can be a local folder or a Hugging Face repo ID."})
    # model_path: str = field(default="lmsys/vicuna-7b-v1.5-16k", metadata={"help": "The path to the weights. This can be a local folder or a Hugging Face repo ID."})
    debug: bool = True

@dataclass
class WorkerArgs:  # (2) worker的参数配置和启动代码
    host: str = field(default="0.0.0.0", metadata={"help": "Worker's host ip address, enable to access from outside if set to `0.0.0.0`, otherwise only localhost can access"})
    port: str = field(default="auto", metadata={"help": "Worker's port, will be assigned automatically from 42902 to 42999 if set to `auto`"})
    controller_address: str = field(default="https://aiapi.ihep.ac.cn", metadata={"help": "Controller's address"})
    worker_address: str = field(default="auto", metadata={"help": "Worker's address, default is http://<ip>:<port>."})
    limit_model_concurrency: int = field(default=5, metadata={"help": "Limit the model's concurrency"})
    stream_interval: float = field(default=0., metadata={"help": "Extra interval for stream response"})
    no_register: bool = field(default=False, metadata={"help": "Do not register to controller"})
    permissions: str = field(default='groups: public,PAYG; owner: hepai@ihep.ac.cn', metadata={"help": "Model's permissions, separated by ;, e.g., 'groups: public; users: a, b; owner: c'"})
    description: str = field(default=None, metadata={"help": "Model's description, automatically get from model adapter if not set"})
    author: str = field(default=None, metadata={"help": "Model's author, automatically get from model adapter if not set"})
    test: bool = field(default=True, metadata={"help": "Test the output of the model if set to `True`"})


if __name__ == '__main__':
    model_args, worker_args = hepai.parse_args((ModelArgs, WorkerArgs))  # 解析多个参数类
    model: BaseWorkerModel = WorkerModel(args=model_args)  # 可传入更多参数
    worker_args.description = worker_args.description or model.xmodel.get_description()
    worker_args.author = worker_args.author or model.xmodel.get_author()
    print(model_args)
    print(worker_args)
    if worker_args.test:
        general.test_model(model, stream=False)
    hepai.worker.start(model=model, worker_args=worker_args)