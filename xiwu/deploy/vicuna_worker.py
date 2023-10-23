import os, sys
from pathlib import Path
here = Path(__file__).parent
if str(here.parent.parent) not in sys.path:
    sys.path.insert(0, str(here.parent.parent))
from typing import Optional
import hai
import torch
from hai import BaseWorkerModel
from dataclasses import dataclass, field
from xiwu.models.vicuna import Vicuna



class WorkerModel(BaseWorkerModel):
    def __init__(self, name, **kwargs):
        self.name = name  # name属性用于用于请求指定调研的模型
        self.vicuna= Vicuna()

    def messages2conv(self, messages):
        conv = self.vicuna.get_conv()
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

    @BaseWorkerModel.auto_stream  # 自动将各种类型的输出转为流式输
    def inference(self, **kwargs):
        # 自己的执行逻辑, 例如: # 
        messages= kwargs.pop('messages', None)
        conv = self.messages2conv(messages)
        prompt = self.vicuna.get_prompt_by_conv(conv)
        return self.vicuna.inference(prev_text=prompt)


# (1) 实现WorkerModel
@dataclass
class ModelArgs:
    name: str = "hepai/vicuna-7B"  # worker的名称，用于注册到控制器
    model_path: str ='/dg_workfs/Beijing-CC/zdzhang/DghpcData/weights/weights/vicuna/vicuna-7b'
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
    permissions: str = 'groups:all'
    #permissions: str = 'groups: hepai;users: zdzhang@ihep.ac.cn,yaohd@ihep.ac.cn,siyangchen@ihep.ac.cn'  # 模型的权限授予，分为用户和组，用;分隔，例如：需要授权给所有组、a用户、b用户：'groups: all; users: a, b; owner: c'
    description: str = 'GPT-3.5 is a large language model released by openai in Nov. 2022'  # 模型的描述
    author: str = 'hepai'  # 模型的作者
    test: bool = False  # 测试模式，不会真正启动worker，只会打印参数

def run_worker(**kwargs):
    # worker_args = hai.parse_args_into_dataclasses(WorkerArgs)  # 解析参数
    model_args, worker_args = hai.parse_args_into_dataclasses((ModelArgs, WorkerArgs))  # 解析多个参数类
    # print(worker_args)
    model = WorkerModel(  # 获取模型
        name=model_args.name
        # 此处可以传入其他参数
        )
    
    print(model_args)
    print(worker_args)
    
    if worker_args.test:
        ret = model.inference(messages=[
                {"role": "system", "content": "Answering questions conversationally"},
                {"role": "user", "content": 'What is your name?'},
                ## 如果有多轮对话，可以继续添加，"role": "assistant", "content": "Hello there! How may I assist you today?"
                ## 如果有多轮对话，可以继续添加，"role": "user", "content": "I want to buy a car."
            ])
        print(ret)
        #return

    hai.worker.start(
        daemon=False,  # 是否以守护进程的方式启动
        model=model,
        worker_args=worker_args,
        **kwargs
        )

if __name__ == '__main__':
    run_worker()