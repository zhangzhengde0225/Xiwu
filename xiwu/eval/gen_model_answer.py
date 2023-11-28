"""
调用HEPAI的模型生成模型答案并保存
"""

import time
import json
import os, sys
from typing import Any
import shortuuid
from dataclasses import dataclass, field
from pathlib import Path
here  = Path(__file__).parent
import hai
import concurrent.futures
from tqdm import tqdm

try:
    from xiwu.version import __version__
except:
    sys.path.append(str(here.parent.parent))
    from xiwu.version import __version__   

from xiwu.apis.fastchat_api import (
    load_questions,
    temperature_config,
    chat_compeletion_openai,
    chat_compeletion_anthropic,
    chat_compeletion_palm,
    get_conversation_template,
    API_MAX_RETRY,
    API_RETRY_SLEEP,
    API_ERROR_OUTPUT,
    reorg_answer_file,
)

from xiwu.apis import HepAILLM

class XiwuEval:

    def __init__(self, args) -> None:
        self.system_prompt = args.system_prompt
        self.llm = HepAILLM(system_prompt=self.system_prompt, model=args.model)

        self.answer_file = args.answer_file
        self.answer_data = self._load_answer_data(self.answer_file)
    
    def _load_answer_data(self, answer_file):
        answer_data = {}
        if os.path.exists(answer_file):
            with open(answer_file, 'r') as f:
                for line in f:
                    answer = json.loads(line)
                    answer_data[answer['question_id']] = answer
        return answer_data

    def __call__(self, **kwds: Any) -> Any:
        question_file = kwds.pop('question_file')
        question_begin = kwds.pop('question_begin')
        question_end = kwds.pop('question_end')
        questions = load_questions(question_file=question_file, begin=question_begin, end=question_end)       
        self.get_answer(questions, **kwds)

    def get_answer(self, questions, **kwds):
        num_choices = kwds.pop('num_choices')
        answer_file = kwds.pop('answer_file')

        for question in tqdm(questions):
            # ans_id = question["answer_id"]
            qid = question["question_id"]
            if qid in self.answer_data:
                continue

            if question["category"] in temperature_config:
                temperature = temperature_config[question["category"]]
            else:
                temperature = 0.5

            choices = []
            messages = [
                    {"role": "system", "content": self.system_prompt},
                ]
            for i in range(num_choices):
                turns = []
                for j in range(len(question["turns"])):
                    qs = question["turns"][j]
                    messages.append({"role": "user", "content": qs})
                    output = self.llm(messages=messages, temperature=temperature, need_print=False)
                    turns.append(output)
                    messages.append({"role": "assistant", "content": output})
                choices.append({"index": i, "turns": turns})

            # Dump answers
            os.makedirs(os.path.dirname(answer_file), exist_ok=True)
            with open(os.path.expanduser(answer_file), "a") as fout:
                ans_id = shortuuid.uuid()
                ans_json = {
                    "question_id": question["question_id"],
                    "answer_id": ans_id,
                    "model_id": self.llm.model,
                    "choices": choices,
                    "tstamp": time.time(),
                }
                fout.write(json.dumps(ans_json) + "\n")
                self.answer_data[qid] = ans_json

def main(args):
    if args.hepai_api_key is not None:
        hai.api_key = args.hepai_api_key

    xiwue = XiwuEval(args)

    kwargs = args.__dict__
    xiwue(**kwargs)


@dataclass
class Args:
    # work_dir: str = field(default=f"{here}")
    bench_name: str = "mt_bench"
    answer_file: str = field(default=None)
    model: str = "openai/gpt-4"  # openai/gpt-3.5-turbo, openai/gpt-4
    num_choices: int = 1  # How many completion choices to generate.
    force_temperature: float = field(default=None)
    max_tokens: int = 1024
    question_begin: int = field(default=None)
    question_end: int = field(default=None)
    parallel: int = 1
    hepai_api_key: str = field(default_factory=lambda: os.environ.get("HEPAI_API_KEY"))
    system_prompt: str = "Answering questions conversationally"

if __name__ == "__main__":
    args = hai.parse_args(Args)

    data_root = f'{here.parent.parent}/data'
    args.question_file = f'{data_root}/{args.bench_name}/question.jsonl'
    if args.answer_file is None:
        args.answer_file = f'{data_root}/{args.bench_name}/model_answer/{args.model}.jsonl'

    main(args)

    
    