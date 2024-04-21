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
from xiwu import CONST, ASSEMBLER
from xiwu import BaseArgs, XBaseModel
from xiwu.modules.models.xiwu_ import Xiwu
from xiwu.apis.fastchat_api import Conversation
import hepai
from hepai import BaseWorkerModel

class WorkerModel(BaseWorkerModel):
    def __init__(self, args, **kwargs):
        self.name = args.name  # name属性用于用于请求指定调研的模型
        self.xmodel: XBaseModel = Xiwu(args=args, **kwargs)

    @BaseWorkerModel.auto_stream  # 自动将各种类型的输出转为流式输
    def inference(self, **kwargs):
        messages= kwargs.pop('messages', None)
        conv = self.xmodel.messages2conv(messages)
        prompt = self.xmodel.get_prompt_by_conv(conv)
        chatcmpl: dict = self.xmodel.inference(prev_text=prompt, **kwargs)
        return chatcmpl
    
    def chat_completions(self, messages: list, **kwargs) -> Generator:
        conv = self.xmodel.messages2conv(messages)
        prompt = self.xmodel.get_prompt_by_conv(conv)
        chatcmpl: dict = self.xmodel.inference(prev_text=prompt, **kwargs)
        return chatcmpl


@dataclass
class ModelArgs:  # (1) 实现WorkerModel
    name: str = field(default="hepai/xiwu-13b", metadata={"help": "worker's name, used to register to controller"})
    model_path: str = field(default="xiwu/xiwu-13b-16k-20240417", metadata={"help": "The path to the weights. This can be a local folder or a Hugging Face repo ID."})


@dataclass
class WorkerArgs:  # (2) worker的参数配置和启动代码
    host: str = field(default="0.0.0.0", metadata={"help": "Worker's address, enable to access from outside if set to `0.0.0.0`, otherwise only localhost can access"})
    port: str = field(default="auto", metadata={"help": "Worker's port"})
    controller_address: str = field(default="https://aiapi.ihep.ac.cn", metadata={"help": "Controller's address"})
    worker_address: str = field(default="auto", metadata={"help": "Worker's address, default is http://<ip>:<port>, the port will be assigned automatically from 42902 to 42999"})
    limit_model_concurrency: int = field(default=5, metadata={"help": "Limit the model's concurrency"})
    stream_interval: float = field(default=0., metadata={"help": "Extra interval for stream response"})
    no_register: bool = field(default=False, metadata={"help": "Do not register to controller"})
    permissions: str = field(default='groups:all,PAYG,zc3900; owner: hepai@ihep.ac.cn', metadata={"help": "Model's permissions, separated by ;, e.g., 'groups: all; users: a, b; owner: c'"})
    description: str = field(default='Xiwu from HepAI', metadata={"help": "Model's description"})
    author: str = field(default='hepai', metadata={"help": "Model's author"})
    test: bool = field(default=True, metadata={"help": "Test mode, will not really start worker, just print the parameters"})

def run_worker(**kwargs):
    # worker_args = hai.parse_args_into_dataclasses(WorkerArgs)  # 解析参数
    model_args, worker_args = hepai.parse_args_into_dataclasses((ModelArgs, WorkerArgs))  # 解析多个参数类
    # print(worker_args)
    model = WorkerModel(args=model_args, **kwargs)
    
    print(model_args)
    print(worker_args)
    
    if worker_args.test:
        stream = True
        chatcmpl: dict = model.inference(messages=[
                {"role": "system", "content": "Answering questions conversationally"},
                {"role": "user", "content": 'who are you?'},
                ## 如果有多轮对话，可以继续添加，"role": "assistant", "content": "Hello there! How may I assist you today?"
                ## 如果有多轮对话，可以继续添加，"role": "user", "content": "I want to buy a car."
            ],
            stream=stream,
            )
        if not stream:
            choice = chatcmpl["choices"][0]
            response = choice["message"]["content"]
            print(response)
        else:
            for chunk in chatcmpl:
                choice = chunk["choices"][0]
                delta = choice['delta']['content']
                print(delta, end='', flush=True)
            pass

    hepai.worker.start(
        model=model,
        worker_args=worker_args,
        **kwargs
        )

if __name__ == '__main__':
    model_args, worker_args = hepai.parse_args((ModelArgs, WorkerArgs))  # 解析多个参数类
    print(model_args)
    print(worker_args)
    model: BaseWorkerModel = WorkerModel(args=model_args)  # 可传入更多参数
    if worker_args.test:
        messages=[
                {"role": "system", "content": "Answering questions conversationally"},
                {"role": "user", "content": 'who are you?'},
                ## 如果有多轮对话，可以继续添加，"role": "assistant", "content": "Hello there! How may I assist you today?"
                ## 如果有多轮对话，可以继续添加，"role": "user", "content": "I want to buy a car."
            ],
        res: Generator = model.chat_completions(messages=messages)
        for ret in res:
            print(ret)
    hepai.worker.start(model=model, worker_args=worker_args)