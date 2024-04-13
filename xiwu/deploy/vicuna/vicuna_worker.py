import os, sys
from pathlib import Path
here = Path(__file__).parent

from typing import Optional
from dataclasses import dataclass, field

try:
    from xiwu.version import __version__
except:
    sys.path.insert(1, str(here.parent.parent.parent))
    from xiwu.version import __version__
from xiwu import YamlConfig
from xiwu.modules.models.vicuna import Vicuna
from xiwu.modules.serve.cli import XChatIO
from xiwu import BaseArgs, XBaseModel
import hepai
from hepai import BaseWorkerModel

class WorkerModel(BaseWorkerModel):
    def __init__(self, args, **kwargs):
        self.name = args.name  # name属性用于用于请求指定调研的模型
        self.xmodel: XBaseModel = Vicuna(args=args, **kwargs)

    @BaseWorkerModel.auto_stream  # 自动将各种类型的输出转为流式输
    def inference(self, **kwargs):
        # 自己的执行逻辑, 例如: # 
        messages= kwargs.pop('messages', None)
        # conv = self.xmodel.messages2conv(messages)  # 注意，此处多次请求会自动把消息保存到conv中
        # prompt = self.xmodel.get_prompt_by_conv(conv)
        prompt = self.xmodel.oai_messages2prompt(messages)
        chatcmpl: dict = self.xmodel.inference(prev_text=prompt, **kwargs)
        return chatcmpl
    
    def chat_completions(self, **kwargs):
        chatcmpl = self.inference(**kwargs)
        # if self.xmodel.args.debug:
        #     res = XChatIO.stream_oai_chatcompletions(chatcmpl)
        #     full_response = ""
        #     for x in res:
        #         print(x, end='', flush=True)
        #         full_response += x
        #     print()
        return chatcmpl

# (1) 实现WorkerModel
@dataclass
class ModelArgs(BaseArgs):
    name: str = "lmsys/vicuna-7b"  # worker的名称，用于注册到控制器
    model_path: str = 'lmsys/vicuna-7b'
    # 其他参数

# (2) worker的参数配置和启动代码
# 用dataclasses修饰器快速定义参数类
@dataclass
class WorkerArgs:
    host: str = "0.0.0.0"  # worker的地址，0.0.0.0表示外部可访问，127.0.0.1表示只有本机可访问
    port: str = "auto"  # 默认从42902开始
    controller_address: str = "http://aiapi.ihep.ac.cn:42901"  # 控制器的地址
    worker_address: str = "auto"  # 默认是http://<ip>:<port>
    limit_model_concurrency: int = 5  # 限制模型的并发请求
    stream_interval: float = 0.  # 额外的流式响应间隔
    no_register: bool = False  # 不注册到控制器
    permissions: str = 'groups:all,PAYG,zc3900; owner: hepai@ihep.ac.cn'
    #permissions: str = 'groups: hepai;users: zdzhang@ihep.ac.cn,yaohd@ihep.ac.cn,siyangchen@ihep.ac.cn'  # 模型的权限授予，分为用户和组，用;分隔，例如：需要授权给所有组、a用户、b用户：'groups: all; users: a, b; owner: c'
    description: str = 'Vicuna from LMSYS'  # 模型的描述
    author: str = 'LMSYS'  # 模型的作者
    test: bool = True  # 测试模式，不会真正启动worker，只会打印参数

def run_worker(**kwargs):
    # worker_args = hai.parse_args_into_dataclasses(WorkerArgs)  # 解析参数
    model_args, worker_args = hepai.parse_args_into_dataclasses((ModelArgs, WorkerArgs))  # 解析多个参数类
    # print(worker_args)
    model = WorkerModel(args=model_args, **kwargs)
    
    print(model_args)
    print(worker_args)
    
    if worker_args.test:
        stream = True
        question = 'who are you?'
        chatcmpl: dict = model.inference(messages=[
                {"role": "system", "content": "Answering questions conversationally"},
                {"role": "user", "content": question},
                ## 如果有多轮对话，可以继续添加，"role": "assistant", "content": "Hello there! How may I assist you today?"
                ## 如果有多轮对话，可以继续添加，"role": "user", "content": "I want to buy a car."
            ],
            stream=stream,
            )
        print(f"Q: {question}")
        if not stream:
            choice = chatcmpl["choices"][0]
            response = choice["message"]["content"]
            print(response)
        else:
            print("A: ", end='')
            res = XChatIO.stream_oai_chatcompletions(chatcmpl)
            full_response = ""
            for x in res:
                print(x, end='', flush=True)
                full_response += x
            print()
            

    hepai.worker.start(model=model, worker_args=worker_args, **kwargs)

if __name__ == '__main__':
    run_worker()