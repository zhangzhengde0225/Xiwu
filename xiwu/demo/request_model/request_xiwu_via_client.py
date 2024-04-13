from typing import Any, Union
import os, sys
from pathlib import Path
here = Path(__file__).parent

try:
    from xiwu import __version__
except:
    sys.path.insert(1, f'{here.parent.parent.parent}')
    from xiwu import __version__
from hepai import HepAI
from hepai import ChatCompletion, ChatCompletionChunk

print(f'__version__: {__version__}')

client = HepAI(
    api_key=os.getenv("HEPAI_A100_API_KEY"),
    # base_url="http://localhost:4280/v1",
    base_url="http://127.0.0.1:21601/v1",
    timeout=36000,
    max_retries=1,

)
 
# 上传和解析文件
# file_object = client.files.create(file=Path(f'{here.parent.parent.parent}/data/2402.07939.pdf'), purpose="file-extract")
# dr.sai后端接收到文件后，保存在用户文件夹内，并解析文件，返回file_object

# 获取文件解析结果
# file_content = client.files.content(file_id=file_object.id).text
## 这是一个str, 解析成dict后包含：content, file_type: "application/pdf", filename: xx, title: "", type: "file"
# print(f'file_content: {len(file_content)}')

# 把文件内容放进请求中
messages=[
    {
        "role": "system",
        "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。",
    },
    # {
    #     "role": "system",
    #     "content": file_content,
    # },
    # {"role": "user", "content": "总结一下这篇论文"},
    # {"role": "user", "content": "帮我写一个快速排序代码"},
    {"role": "user", "content": "Hello"},
]

messages=[
    # {
    #     "role": "system",
    #     "content": file_content,
    # },
    # {"role": "user", "content": "总结一下这篇论文"},
    # {"role": "user", "content": "帮我写一个快速排序代码"},
    # {"role": "user", "content": "who are you?"},
    {"role": "user", "content": "Write a quick sort code for me"},
]
 
stream = True

# res: Union[ChatCompletion, str] = client.chat.completions.create(
res = client.chat.completions.create(
#   model="moonshot-v1-32k",
  model="lmsys/vicuna-7b",
#   model="lmsys/vicuna-7b-v1.5-16k",
    # model="hepai/demo_worker",
  messages=messages,
  temperature=0.3,
  stream=stream,
)

print(f'res [{type(res)}]: {res}')

# if res.status_code != 200:
    # print(f'Failed: {res}')
    # exit(1)
if not stream:
    if res.choices is None:
        print(res)
    else:
        message = res.choices[0].message  # ChatCompletionMessage, 有function_call, tool_calls, content, role
        print(message)
else:
    full_answer = ''
    for chunk in res:
        if chunk.choices:
            choices_list = []
            answer = chunk.choices[0].delta.content
            # print(f'{answer}')
            if answer:
                print(f'{answer}', end='', flush=True)
                full_answer += answer
    print()
