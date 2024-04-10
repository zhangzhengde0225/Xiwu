import os, sys
from pathlib import Path
here = Path(__file__).parent

try:
    from xiwu.version import __version__
except:
    sys.path.insert(1, str(here.parent.parent.parent))
    from xiwu.version import __version__
    
import hepai
from hepai import BaseWorkerModel
from dataclasses import dataclass, field


class WorkerModel(BaseWorkerModel):
    def __init__(self, name, **kwargs):
        self.name = name  # name属性用于用于请求指定调研的模型

    @BaseWorkerModel.auto_stream  # 自动将各种类型的输出转为流式输
    def inference(self, **kwargs):
        # 自己的执行逻辑, 例如: # 
        input = kwargs.pop('input', None)
        output = [1, 2, 3, 4, 5]  # 修改为自己的输出
        return output, input
        # for i in output:
        #     yield i  # 可以return返回python的基础类型或yield生成器

    def chat_completions(self, **kwargs):
        stream = kwargs.get('stream', False)
        if stream:
            return self.build_stream(**kwargs)
        else:
            return "data: output\n\n"
        
    def build_stream(self, **kwargs):
        for i in range(10):
            yield f"data: {i}\n\n"
    

def run_worker(**kwargs):
    model_args, worker_args = hepai.parse_args_into_dataclasses((ModelArgs, WorkerArgs))  # 解析多个参数类
    # print(worker_args)
    model = WorkerModel(  # 获取模型
        name=model_args.name
        # 此处可以传入其他参数
        )
    
    if worker_args.test:
        ret = model.inference(input='test')
        print(ret)
        return

    hepai.worker.start(model=model, worker_args=worker_args, **kwargs)

# (1) 实现WorkerModel
@dataclass
class ModelArgs:
    name: str = "hepai/demo_worker"  # worker的名称，用于注册到控制器
    # 其他参数

# (2) worker的参数配置和启动代码
@dataclass
class WorkerArgs:
    host: str = "127.0.0.1"  # worker的地址，0.0.0.0表示外部可访问，127.0.0.1表示只有本机可访问
    port: str = "auto"  # worker的端口，默认从42902开始自动分配
    controller_address: str = "http://127.0.0.1:21601"  # 控制器的地址
    worker_address: str = "auto"  # 默认是http://<ip>:<port>
    limit_model_concurrency: int = 5  # 限制模型的并发请求
    stream_interval: float = 0.  # 额外的流式响应间隔
    no_register: bool = False  # 不注册到控制器
    permissions: str = 'groups: all; owner: admin'  # 模型的权限授予，分为用户和组，用;分隔，例如：需要授权给所有组、a用户、b用户：'groups: all; users: a, b; owner: c'
    description: str = 'This is a demo worker in HeiAI-Distributed Deploy Framework'  # 模型的描述
    author: str = 'hepai'  # 模型的作者
    test: bool = False  # 测试模式，不会真正启动worker，只会打印参数

if __name__ == '__main__':
    run_worker()