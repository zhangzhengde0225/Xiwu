"""
读取问题、模型答案、参考答案、评测提示词，生成评测对比结果
结果存在：data/mt_bench/model_judgment/gpt-4_single.jsonl
"""


import os, sys
import numpy as np
import json
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
)

class XiwuJudgement:

    def __init__(self, args, **kwargs) -> None:
        self.args = args

    def run(self):
        pass


def main(args):
    xw_judge = XiwuJudgement(args)
    xw_judge.run()

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
        play_a_match_func = play_a_match_single
        output_file = (
            f"data/{args.bench_name}/model_judgment/{args.judge_model}_single.jsonl"
        )
        make_match_func = make_match_single
        baseline_model = None
    else:
        judges = make_judge_pairwise(args.judge_model, judge_prompts)
        play_a_match_func = play_a_match_pair
        output_file = (
            f"data/{args.bench_name}/model_judgment/{args.judge_model}_pair.jsonl"
        )
        if args.mode == "pairwise-all":
            make_match_func = make_match_all_pairs
            baseline_model = None
        else:
            make_match_func = make_match
            baseline_model = args.baseline_model

    check_data(questions, model_answers, ref_answers, models, judges)

    question_math = [q for q in questions if q["category"] in NEED_REF_CATS]
    question_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

    # Make matches
    matches = []
    matches += make_match_func(
        question_default, models, model_answers, judges["default"], baseline_model
    )
    matches += make_match_func(
        question_math,
        models,
        model_answers,
        judges["math"],
        baseline_model,
        ref_answers,
    )
    matches += make_match_func(
        question_default,
        models,
        model_answers,
        judges["default-mt"],
        baseline_model,
        multi_turn=True,
    )
    matches += make_match_func(
        question_math,
        models,
        model_answers,
        judges["math-mt"],
        baseline_model,
        ref_answers,
        multi_turn=True,
    )

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
    input("Press Enter to confirm...")

    # Play matches
    if args.parallel == 1:
        for match in tqdm(matches):
            play_a_match_func(match, output_file=output_file)
    else:

        def play_a_match_wrapper(match):
            play_a_match_func(match, output_file=output_file)

        np.random.seed(0)
        np.random.shuffle(matches)

        with ThreadPoolExecutor(args.parallel) as executor:
            for match in tqdm(
                executor.map(play_a_match_wrapper, matches), total=len(matches)
            ):
                pass



@dataclass
class Args:
    data_dir: str = field(default=f"{here.parent.parent}/data")
    bench_name: str = "mt_bench"
    judge_file: str = "data/judge_prompts.jsonl"
    judge_model: str = "gpt-4"
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
   


