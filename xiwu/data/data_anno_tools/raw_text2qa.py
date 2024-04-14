"""
将一个或多个原始文本文件.txt转换为问答对格式，根据内容产生问题和对应的答案。
其中，原始文本的每个段落需要用</s>分隔
"""

import os, sys
from pathlib import Path
import json
import ast

here = Path(__file__).parent

try:
    from xiwu.version import __version__
except:
    sys.path.append(f'{here.parent.parent.parent}')
    from xiwu.version import __version__
from xiwu.apis.xiwu_api import BaseQADatasetSaver
from xiwu.apis.xiwu_api import HepAILLM
import hepai

class RawText2QA(BaseQADatasetSaver):

    def __init__(self, file_path=None, **kwargs):
        super().__init__(output_file=file_path, **kwargs)
        self.gpt = HepAILLM()

    def __call__(self, raw_text_file, **kwargs):
        paragraphs = self._load_raw_text(raw_text_file)
        print(f'paragraphs: {len(paragraphs)}')
        for i, p in enumerate(paragraphs):
            qa_pairs = self._paragraph2qas(p)  # 一个段落转换为多个qa对
            entities = self._qas2entities(qa_pairs, source=raw_text_file)
            self.add_entities_and_save(entities)

    def _qas2entities(self, qa_pairs, source):
        entities = list()
        for qs in qa_pairs:
            entity = dict()
            entity['question'] = qs[0]
            entity['answer'] = qs[1]
            entity['source'] = source
            entity['labeler'] = 'zhengde zhang'
            entities.append(entity)
        return entities

    def _paragraph2qas(self, paragraph, max_len=800):
        # 再将paragraph分成多个句子
        num_parts = len(paragraph) // max_len + 1
        # print(f'num_parts: {num_parts}')
        len_of_each_part = len(paragraph) // num_parts
        parts = [paragraph[i*len_of_each_part:(i+1)*len_of_each_part] for i in range(num_parts)]
        # print(f'parts: {len(parts)} {[len(x) for x in parts]}')
        qa_pairss = []
        for p in parts:
            qa_pairs = self._text2qas(p)
            qa_pairss.extend(qa_pairs)
        return qa_pairss

    def _text2qas(self, text):
        system_prompt = "You are a newbie, learning with curiosity, asking questions based on information."
        
        example = "[{'question1': <VALUE>, 'answer1': <VALUE>}, {'question2': <VALUE>, 'answer2': <VALUE>}, ...]"
        prompt = f"""
Please raise some questions from the input text delimited by triple backticks.
Prefer conceptual questioning while being specific, Each question needs to contain sufficient information and be brief.
Provide them in JSON format in a list, for example: {example}, only output one JSON.
用中文输出。

Input: 
```{text}```
"""
        print(f'Text: {text}')
        result = self.gpt(prompt, system_prompt=system_prompt)
        return self.result2qas(result)
    
    def result2qas(self, result):
        result = self.parse_by_ast(result)
        if isinstance(result, dict):
            print(f'result format error, pass: {result}')
            return []
        qas = []
        count = 0
        for content in result:
            keys = list(content.keys())
            question_keys = [x for x in keys if 'question' in x]
            assert len(question_keys) == 1, f'question_keys: {question_keys}'
            q = content[question_keys[0]]
            answer_key = question_keys[0].replace('question', 'answer')
            a = content.get(answer_key, None)
            count += 1
            if q is None or a is None:
                continue
            qas.append((q, a))
        return qas

    def parse_by_ast(self, string):
        try:
            out = ast.literal_eval(string)
        except ValueError as e:
            string = string.split('\n\n')[0]
            out = ast.literal_eval(string)
        except Exception as e:
            # print(f'ast.literal_eval(string) error: {e}')
            print(f'ast.literal_eval(string) error. \nString: [{[string]} \nError: {e}')
            out = []
        return out

    def _load_raw_text(self, raw_text_file):
        with open(raw_text_file, 'r') as f:
            paragraphs = f.read().split('</s>')
        return paragraphs
    
def doit():
    file_path = f'{here}/ihep-qa-460.json'
    meta_data = {"description": "data formated by RawText2QA"}
    rt2qa = RawText2QA(file_path=file_path, meta_data=meta_data)
    raw_text_files = [f'{here}/raw_text/ihep.txt', f'{here}/raw_text/cepc.txt']
    for raw_text_file in raw_text_files:
        rt2qa(raw_text_file)


if __name__ == "__main__":
    doit()
