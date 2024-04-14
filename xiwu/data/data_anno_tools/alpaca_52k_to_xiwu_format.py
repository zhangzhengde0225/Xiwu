"""
将alpaca_52k数据集转换为xiwu数据集格式
"""

import os, sys
from pathlib import Path
import json
import ast
from dataclasses import dataclass, field
here = Path(__file__).parent

import json

try:
    from xiwu.version import __version__
except:
    sys.path.append(f'{here.parent.parent.parent}')
    from xiwu.version import __version__
from xiwu.apis.xiwu_api import BaseQADatasetSaver, Entry
from xiwu.apis.xiwu_api import HepAILLM
from xiwu.configs.constant import DATASETS_DIR
import hepai


class Alpaca2XiwuQA(BaseQADatasetSaver):

    def __init__(self, file_path=None, **kwargs):
        version = '1.0'
        meta_data = {'description': 'Data from alpaca_52k'}
        super().__init__(
            file_path=file_path, 
            version=version,
            meta_data=meta_data,
            **kwargs)
    
    def _conversation2qas(self, conversations):
        assert len(conversations) == 2, f'conversion length is not 2: {len(conversations)}'
        q = conversations[0]
        assert q['from'] == 'human', f'role is not human: {q["from"]}'
        q_content = q['value']
        a = conversations[1]
        assert a['from'] == 'gpt', f'role is not gpt: {a["from"]}'
        a_content = a['value']
        return q_content, a_content


    def read_data(self, source):
        with open(source, 'r') as f:
            data = json.load(f)
        return data
    
    def _split_text(self, text):
        """
        ## 注意，这个训练数据自带system_message，需要保留
        格式为：<system_message>\n\n### Instruction:\n<question>\n\n### Response:
        或者: <system_message>\n\n### Instruction:\n<question>\n\n### Input:\n<input>\n\n### Response:
        """
        chunks = text.split('\n\n### ')
        assert len(chunks) in [3, 4], f'chunks length is not 3 or 4: {len(chunks)}'
        system_message = chunks[0]
        question = chunks[1]
        assert question.startswith('Instruction:\n'), f'question does not start with ### Instruction: {question}'
        question = question.replace('Instruction:\n', '')
        if len(chunks) == 4:
            input_ = chunks[2]
            assert input_.startswith('Input:\n'), f'input does not start with ### Input: {input_}'
            input_ = input_.replace('Input:\n', '')
            question = f'{question}\n{input_}'
        return system_message, question


    def __call__(self, source, **kwargs):
        alpaca_data = self.read_data(source)
        for i, x in enumerate(alpaca_data):
            print(f'\rprocessing: [{i+1:0>5}/{len(alpaca_data)}]', end='', flush=True)
            id = x['id']
            conversations = x['conversations']
            q_content, a_content = self._conversation2qas(conversations)

            system_msg, q_content = self._split_text(q_content)
            entry = Entry(
                question=q_content,
                answer=a_content,
                source='alpaca_52k',
                labeler='gpt',
                misc={'system_message': system_msg}
            )
            self.add_one_entry(entry, save_immediately=False)
            # if i ==3:
                # break
        print()
        self.save_data2file(print=True)
        
        pass


@dataclass
class Args:
    source: str = f'{DATASETS_DIR}/alpaca_52k/alpaca-data-conversation.json'
    file_path: str = f'{DATASETS_DIR}/hep_text_v1.0/alpaca-52k-xformat.json'
    initialize: bool = field(default=True, metadata={"help": "Initialize the even if the file exists."})

if __name__ == '__main__':
    args = hepai.parse_args(Args)
    alpaca2xiwu = Alpaca2XiwuQA(file_path=args.file_path, initialize=args.initialize)
    alpaca2xiwu(args.source)
