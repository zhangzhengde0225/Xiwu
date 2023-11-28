"""
读取问题、模型答案、参考答案、评测提示词，生成评测对比结果
结果存在：data/mt_bench/model_judgment/gpt-4_single.jsonl
"""

import time
import os, sys
import numpy as np
import json
import re
import ast
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import hai
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
here = Path(__file__).parent

try:
    from xiwu.version import __version__
except:
    sys.path.append(str(here.parent.parent))
    from xiwu.version import __version__   

from xiwu.apis.fastchat_api import (
    load_questions,
    load_model_answers,
    load_judge_prompts,
    check_data,
    play_a_match_pair,
    play_a_match_single,
    get_model_list,
    Judge,
    MatchPair,
    MatchSingle,
    NEED_REF_CATS,
    make_match_single,
    make_judge_single,
    make_judge_pairwise,
    make_match_all_pairs,
    make_match,
    get_conversation_template,
    chat_compeletion_openai,
    chat_compeletion_anthropic,
)

from xiwu.apis.xiwu_api import (
    HepAILLM
)

class XiwuJudgement:

    def __init__(self, args, **kwargs) -> None:
        self.args = args
        self.matches, self.output_file = self._init()
        self.hepai_llm = HepAILLM()


    @property
    def make_match_func(self):
        if self.args.mode == "single":
            return make_match_single
        elif self.args.mode == "pairwise-all":
            return make_match_all_pairs
        else:
            return make_match
        
    @property
    def play_a_match_func(self):
        if self.args.mode == "single":
            return self.play_a_match_single
        else:
            return play_a_match_pair  # TODO: 修改成HEPAI请求
        
    def run_judge_single(self, question, answer, judge, ref_answer=None, multi_turn=False, by_HepAI=False):
        """
        调用HepAI执行Judge评测
        :by_HepAI: 是否使用HepAI的模型进行评测
        """
        kwargs = {}
        model = judge.model_name  # str: gpt-4
        if ref_answer is not None:  # 存在参考答案时，合并到kwargs中
            kwargs["ref_answer_1"] = ref_answer["choices"][0]["turns"][0]
            if multi_turn:
                kwargs["ref_answer_2"] = ref_answer["choices"][0]["turns"][1]
        # 获取user_prompt
        if multi_turn:  # 多轮时
            user_prompt = judge.prompt_template["prompt_template"].format(
                question_1=question["turns"][0],
                question_2=question["turns"][1],
                answer_1=answer["choices"][0]["turns"][0],
                answer_2=answer["choices"][0]["turns"][1],
                **kwargs,
            )
        else:
            user_prompt = judge.prompt_template["prompt_template"].format(
                question=question["turns"][0],
                answer=answer["choices"][0]["turns"][0],
                **kwargs,
            )

        rating = -1

        system_prompt = judge.prompt_template["system_prompt"] 

        if by_HepAI:
            result = self.hepai_llm(
                prompt=user_prompt,
                sys_prompt=system_prompt,
                model=model,
                need_print=True,
                )
            
        else:
            conv = get_conversation_template(model)
            conv.set_system_message(system_prompt)
            conv.append_message(conv.roles[0], user_prompt)
            conv.append_message(conv.roles[1], None)

            if model in ["gpt-3.5-turbo", "gpt-4"]:
                judgment = chat_compeletion_openai(model, conv, temperature=0, max_tokens=2048)
            elif model in ["claude-v1", "claude-instant-v1"]:
                judgment = chat_compeletion_anthropic(
                    model, conv, temperature=0, max_tokens=1024
                )
            else:
                raise ValueError(f"Invalid judge model name: {model}")

        if judge.prompt_template["output_format"] == "[[rating]]":
            one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
            match = re.search(one_score_pattern, judgment)
            if not match:
                one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")
                match = re.search(one_score_pattern_backup, judgment)

            if match:
                rating = ast.literal_eval(match.groups()[0])
            else:
                rating = -1
        else:
            raise ValueError(
                f"invalid output format: {judge.prompt_template['output_format']}"
            )

        return rating, user_prompt, judgment

    def play_a_match_single(self, match, output_file=None):
        question, model, answer, judge, ref_answer, multi_turn = (
            match.question,
            match.model,
            match.answer,
            match.judge,
            match.ref_answer,
            match.multi_turn,
        )

        if judge.prompt_template["type"] == "single":
            score, user_prompt, judgment = self.run_judge_single(
                question, answer, judge, ref_answer, multi_turn=multi_turn
            )

            question_id = question["question_id"]
            turn = 1 if not multi_turn else 2
            result = {
                "question_id": question_id,
                "model": model,
                "judge": (judge.model_name, judge.prompt_template["name"]),
                "user_prompt": user_prompt,
                "judgment": judgment,
                "score": score,
                "turn": turn,
                "tstamp": time.time(),
            }
            print(
                f"question: {question_id}, turn: {turn}, model: {model}, "
                f"score: {score}, "
                f"judge: {(judge.model_name, judge.prompt_template['name'])}"
            )
        else:
            raise ValueError(f"invalid judge type: {judge['type']}")

        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "a") as fout:
                fout.write(json.dumps(result) + "\n")

        return result

    def _init(self):
        args = self.args
        question_file = f"{args.data_dir}/{args.bench_name}/question.jsonl"
        answer_dir = f"{args.data_dir}/{args.bench_name}/model_answer"
        ref_answer_dir = f"{args.data_dir}/{args.bench_name}/reference_answer"
        args.judge_file = f"{args.data_dir}/{args.bench_name}/judge_prompts.jsonl"

        # Load questions
        questions = load_questions(question_file, None, None)

        # Load answers
        model_answers = load_model_answers(answer_dir)  # 文件夹下所有模型的答案
        ref_answers = load_model_answers(ref_answer_dir)  # 参考答案：101-130

        # Load judge
        judge_prompts = load_judge_prompts(args.judge_file)

        if args.first_n:
            questions = questions[: args.first_n]

        if args.model_list is None:
            models = get_model_list(answer_dir)
        else:
            models = args.model_list

        if args.mode == "single":
            judges = make_judge_single(args.judge_model, judge_prompts)
            output_file = (
                f"data/{args.bench_name}/model_judgment/{args.judge_model}_single.jsonl"
            )
            baseline_model = None
        else:
            judges = make_judge_pairwise(args.judge_model, judge_prompts)
            output_file = (
                f"data/{args.bench_name}/model_judgment/{args.judge_model}_pair.jsonl"
            )
            if args.mode == "pairwise-all":
                baseline_model = None
            else:
                baseline_model = args.baseline_model

        check_data(questions, model_answers, ref_answers, models, judges)

        question_math = [q for q in questions if q["category"] in NEED_REF_CATS]
        question_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

        matches = self.make_matchs(question_default, question_math, models, model_answers, judges, baseline_model, ref_answers=None, multi_turn=False)
        
        match_stat = {}
        match_stat["bench_name"] = args.bench_name
        match_stat["mode"] = args.mode
        match_stat["judge"] = args.judge_model
        match_stat["baseline"] = baseline_model
        match_stat["model_list"] = models
        match_stat["total_num_questions"] = len(questions)
        match_stat["total_num_matches"] = len(matches)
        match_stat["output_path"] = output_file

        # Show match stats and prompt enter to continue
        print("Stats:")
        print(json.dumps(match_stat, indent=4))
        # input("Press Enter to confirm...")
        return matches, output_file
    
    def make_matchs(self, question_default, question_math, models, model_answers, judges, baseline_model, ref_answers=None, multi_turn=False):
        # Make matches
        matches = []
        matches += self.make_match_func(
            question_default, models, model_answers, judges["default"], baseline_model
        )
        matches += self.make_match_func(
            question_math,
            models,
            model_answers,
            judges["math"],
            baseline_model,
            ref_answers,
        )
        matches += self.make_match_func(
            question_default,
            models,
            model_answers,
            judges["default-mt"],
            baseline_model,
            multi_turn=True,
        )
        matches += self.make_match_func(
            question_math,
            models,
            model_answers,
            judges["math-mt"],
            baseline_model,
            ref_answers,
            multi_turn=True,
        )
        return matches

    def run(self):
        # Play matches
        matches = self.matches
        output_file = self.output_file

        if self.args.parallel == 1:
            for match in tqdm(matches):
                self.play_a_match_func(match, output_file=output_file)
        else:

            def play_a_match_wrapper(match):
                self.play_a_match_func(match, output_file=output_file)

            np.random.seed(0)
            np.random.shuffle(matches)

            with ThreadPoolExecutor(args.parallel) as executor:
                for match in tqdm(
                    executor.map(play_a_match_wrapper, matches), total=len(matches)):
                    pass
        pass

def main(args):
    xw_judge = XiwuJudgement(args)
    xw_judge.run()

@dataclass
class Args:
    data_dir: str = field(default=f"{here.parent.parent}/data")
    bench_name: str = "mt_bench"
    judge_file: str = "data/judge_prompts.jsonl"
    # judge_model: str = "hepai/gpt-3.5-turbo"
    judge_model: str = 'gpt-4'  # 与ref_answer的模型一致
    baseline_model: str = "gpt-3.5-turbo"
    mode: str = "single"
    model_list: Optional[List[str]] = field(default=None)
    parallel: int = 1
    first_n: Optional[int] = None

    def __post_init__(self):
        # Validate mode
        if self.mode not in ["pairwise-baseline", "pairwise-all", "single"]:
            raise ValueError(f"Invalid mode: {self.mode}. Choose from 'pairwise-baseline', 'pairwise-all', or 'single'.")

if __name__ == '__main__':
    args = hai.parse_args(Args)
    print(f'args: {args}')
    main(args=args)
   


