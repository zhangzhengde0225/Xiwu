
import time
import json
import os, sys
import shortuuid
from dataclasses import dataclass, field
from pathlib import Path
here  = Path(__file__).parent
import hai
import concurrent.futures
import tqdm

import openai

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

def request_model(model, messages):
    # system_prompt = system_prompt if system_prompt else "Answering questions conversationally"

    result = hai.LLM.chat(
            model=model,
            messages=messages,
            stream=True,
        )

    full_result = ""
    for i in result:
        full_result += i
        sys.stdout.write(i)
        sys.stdout.flush()
    print()
    return full_result

def chat_compeletion_hai(model, conv):
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            # models = hai.Model.list()  # 列出所有可用模型
            # print(models)

            messages = conv.to_hai_api_messages()
            result = hai.LLM.chat(
                 model=model,
                 messages=messages,
                 stream=True,
             )
            full_result = ""
            for i in result:
                full_result += i
                sys.stdout.write(i)
                sys.stdout.flush()
            output = full_result
            #output = request_model(model, messages)
            break
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return output

def get_answer(
    question: dict, model: str, num_choices: int, max_tokens: int, answer_file: str
):
    if args.force_temperature:
        temperature = args.force_temperature
    elif question["category"] in temperature_config:
        temperature = temperature_config[question["category"]]
    else:
        temperature = 0.7

    choices = []
    chat_state = None  # for palm-2 model
    for i in range(num_choices):
        conv = get_conversation_template(model)
        # print(f'conv: {conv}', file=sys.stderr)
        turns = []
        for j in range(len(question["turns"])):
            conv.append_message(conv.roles[0], question["turns"][j])
            conv.append_message(conv.roles[1], None)

            if model in ["claude-v1", "claude-instant-v1"]:
                output = chat_compeletion_anthropic(
                    model, conv, temperature, max_tokens
                )
            elif model == "palm-2-chat-bison-001":
                chat_state, output = chat_compeletion_palm(
                    chat_state, model, conv, temperature, max_tokens
                )
            elif model in ["hepai/vicuna-7B" ,"hepai/vicuna-13B", "hepai/xiwu-13B","openai/gpt-4","openai/gpt-3.5-turbo"]:
                output = chat_compeletion_hai(model, conv)
            
            else:
                output = chat_compeletion_openai(model, conv, temperature, max_tokens)

            conv.update_last_message(output)
            turns.append(output)

        choices.append({"index": i, "turns": turns})

    # Dump answers
    ans = {
        "question_id": question["question_id"],
        "answer_id": shortuuid.uuid(),
        "model_id": model,
        "choices": choices,
        "tstamp": time.time(),
    }

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "a") as fout:
        fout.write(json.dumps(ans) + "\n")

def main(args):
    if args.hepai_api_key is not None:
        hai.api_key = args.hepai_api_key

    prefix = f"{here.parent}/repos/FastChat/fastchat/llm_judge"
    # question_file = f"{prefix}/data/{args.bench_name}/question.jsonl"
    question_file = f"{here.parent.parent}/data/xiwu_eval_dataset/question_data.jsonl"
    questions = load_questions(question_file, args.question_begin, args.question_end)

    if args.answer_file:
        answer_file = args.answer_file
    else:
        # answer_file = f"{prefix}/data/{args.bench_name}/model_answer/{args.model}.jsonl"
        answer_file = f"{here.parent.parent}/data/xiwu_eval_dataset/model_answer/{args.model}.jsonl"
    # assert os.path.exists(answer_file), f"Answer file {answer_file} does not exist."
    print(f"Output to {answer_file}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = []
        for question in questions:
            future = executor.submit(
                get_answer,
                question,
                args.model,
                args.num_choices,
                args.max_tokens,
                answer_file,
            )
            futures.append(future)
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()
    reorg_answer_file(answer_file)


@dataclass
class Args:
    # work_dir: str = field(default=f"{here}")
    bench_name: str = "mt_bench"
    answer_file: str = field(default=None)
    model: str = "gpt-3.5-turbo"
    num_choices: int = 1  # How many completion choices to generate.
    force_temperature: float = field(default=None)
    max_tokens: int = 1024
    question_begin: int = field(default=None)
    question_end: int = field(default=None)
    parallel: int = 1
    hepai_api_key: str = field(default_factory=lambda: os.environ.get("HEPAI_API_KEY"))

if __name__ == "__main__":
    args = hai.parse_args(Args)
    

    main(args)

    
    