# -*- coding: utf-8 -*-

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
# from xiwu import YamlConfig
from xiwu.modules.models.llama_262k import Llama3_262k
from xiwu.modules.deployer.cli import XChatIO
from xiwu import BaseArgs, XBaseModel
import hepai
from hepai import BaseWorkerModel
import json

class WorkerModel(BaseWorkerModel):
    def __init__(self, args, **kwargs):
        self.name = args.name  # name属性用于用于请求指定调研的模型
        self.xmodel: XBaseModel = Llama3_262k(args=args, **kwargs)

    @BaseWorkerModel.auto_stream  # 自动将各种类型的输出转为流式输
    def inference(self, **kwargs):
        chatcmpl = self.inference_ori(**kwargs)

        # 初始化用于存储最终结果的字符串
        json_buffer = ""

        # 初始化标志变量，用于标记是否是第一个数据块
        first_chunk_processed = False

        # 迭代生成器，逐字符处理结果
        for chunk in chatcmpl:
            json_buffer += chunk
            if '\n' in json_buffer:
                lines = json_buffer.split('\n')
                for line in lines[:-1]:
                # 如果当前字符是换行符，表示一个数据块结束
                    if line.startswith('data: '):
                        json_str = json_buffer[len('data: '):]  # 去掉 'data: ' 前缀
                        data = json.loads(json_str)
                        # 提取 content 字段
                        content = str(data['choices'][0]['delta'].get('content', ''))
                        if content:
                            if not first_chunk_processed:
                                # 处理第一个数据块，删除最后一个换行符之前的内容
                                content = content.split('\n')[-1]
                                # 检查并删除 "USER: " 前缀
                                if content.endswith('USER: '):
                                    content = content[:-len('USER: ')]
                                first_chunk_processed = True
                            # 检查并删除末尾的 "</s>"
                            if content.endswith('</s>'):
                                content = content[:-len('</s>')]
                            # 检查并删除末尾的 " ASSISTANT:"
                            if content.endswith(' ASSISTANT:'):
                                content = content[:-len(' ASSISTANT:')]
                            # 更新数据
                            data['choices'][0]['delta']['content'] = content
                            processed_chunk = f"data: {json.dumps(data)}\n"
                            yield processed_chunk
                json_buffer = lines[-1]

        # 处理最后一个数据块，如果没有换行符结束
        if json_buffer.startswith('data: '):
            json_str = json_buffer[len('data: '):]  # 去掉 'data: ' 前缀
            data = json.loads(json_str)
            # 提取content字段
            content = str(data['choices'][0]['delta'].get('content', ''))
            if content:
                # 检查并删除末尾的 "</s>"
                if content.endswith('</s>'):
                    content = content[:-len('</s>')]
                # 检查并删除末尾的 " ASSISTANT:"
                if content.endswith(' ASSISTANT:'):
                    content = content[:-len(' ASSISTANT:')]
                # 更新数据
                data['choices'][0]['delta']['content'] = content
                processed_chunk = f"data: {json.dumps(data)}\n"
                yield processed_chunk
    
    def inference_ori(self, **kwargs):
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
    name: str = "Llama-3-8B-Instruct-262k"  # worker的名称，用于注册到控制器
    model_path: str = '/data1/sqr/Models/Llama-3-8B-Instruct-262k'
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
    description: str = 'Llama-3-262k from meta-llama'  # 模型的描述
    author: str = 'meta-llama'  # 模型的作者
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
            # 删除前两个单词及其后的换行符
            response = ' '.join(response.split()[1:]).replace('\n','')
            print(response)
        else:
            for i in chatcmpl:
                sys.stdout.write(i)
                sys.stdout.flush()
            print()
            # print("A: ", end='')
            # res = XChatIO.stream_oai_chatcompletions(chatcmpl)
            # full_response = ""
            # first_processed = False  # 用于标记是否已经处理了第一个 x
            # for x in res:
            #     if not first_processed:
            #         # 处理第一个 x，删除前两个单词及其后的换行符，并在末尾添加一个空格
            #         x = ' '.join(x.split()[1:]).replace('\n','') + ' '
            #         first_processed = True
            #     print(x, end='', flush=True)
            #     full_response += x
            # print()
            

    hepai.worker.start(model=model, worker_args=worker_args, **kwargs)

if __name__ == '__main__':
    run_worker()