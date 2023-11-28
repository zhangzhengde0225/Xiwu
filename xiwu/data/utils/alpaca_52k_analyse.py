"""
分析alpaca-52k的数据格式：
结论：
+ 52002条数据
+ 每条数据都是2轮（一问一答）
+ 问题的开头都是Below is an instruction that describes a task
+ Prompt的结构有两种：
    + <Head1>.\n\n### Instruction:\n<Instruction>\n\n### Response:"
    + <Head2>.\n\n### Instruction:\n<Instruction>\n\n### Input:\n<Input>\n\n### Response:"
+ Head有2种：
    + 31323: Below is an instruction that describes a task. Write a response that appropriately completes the request.
    + 20679: Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
"""
import os, sys
from pathlib import Path
import json

here = Path(__file__).parent


class AlpacaDataAnalyse:
    def __init__(self):
        file = f'{here.parent.parent}/data/open_data/alpaca_52k/alpaca-data-conversation.json'
        self.data = self._load_json(file)
        pass

    def _load_json(self, file):
        with open(file, 'r') as f:
            data = json.load(f)
        return data

    def analyse(self):
        data = self.data
        print(f'共计：{len(data)}条数据')  # 52002条

        tasks = dict()
        for i, d in enumerate(data):
            # print(f'第{i}条数据：')
            # print(d.keys())  # id, conversation
            conv = d['conversations']  # 是一个list
            # print(f'对话共计：{len(conv)}轮')
            if len(conv) != 2:  # 全都是2轮
                print(conv, len(conv))  
                break
            # 第一轮是问题，第二轮是回答
            assert len(conv) == 2, '对话轮数不为2'
            quetion = conv[0]['value']
            answer = conv[1]['value']
            # print(f'问题：{quetion}')

            # 看看是不是所有的开头都是Below is an instruction that describes a task
            if not quetion.startswith('Below is an instruction that describes a task'):
                print(quetion)  # 结论，确实都是。
                raise Exception('不是所有的开头都是Below is an instruction that describes a task')

            # 看看任务有哪些？
            try:
                task = quetion.split('\n\n')[0].split('task. ')[1]
            except:
                task = quetion.split('\n\n')[0].split('further context. ')[1]
            
            if task not in tasks:
                tasks[task] = 1
            else:
                tasks[task] += 1

        print(len(tasks), tasks)

            # break  
            # if i == 10:
            #     break  
        pass


if __name__ == '__main__':
    AlpacaDataAnalyse().analyse()