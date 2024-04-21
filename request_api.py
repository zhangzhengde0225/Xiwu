from typing import Generator
import os
from hepai import HepAI, Stream
from dataclasses import dataclass, field


client = HepAI(
    api_key="default",
    base_url="http://localhost:42902/v1")  # 需正确填入worker的地址
model = "default"
stream = False  # 是否流式输出

messages=[
            # {"role": "system", "content": "Answering questions conversationally"},
            {"role": "user", "content": 'Hello'},
            {"role": "assistant", "content": "Hello there! How may I assist you today?"},
            {"role": "user", "content": "who are you"}
        ]

res = client.chat.completions.create(
    model=model,
    messages=messages,
    stream=stream)

if isinstance(res, Stream):  # 流式输出
    for chunk in res:
        x = chunk.choices[0].delta.content
        print(f'{x}', end='', flush=True)
    print()
else:  # 非流式输出
    content = res.choices[0].message.content
    print(content)


