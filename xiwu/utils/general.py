
from typing import Generator
import json


def test_model(model, stream=False):
    """Note: the model is the instance of WorkerModel."""
    # assert isinstance(model, WorkerModel), f"model must be an instance of WorkerModel, got {type(model)}"
    messages=[
                # {"role": "system", "content": "Answering questions conversationally"},
                {"role": "user", "content": 'Hello'},
                {"role": "assistant", "content": "Hello there! How may I assist you today?"},
                {"role": "user", "content": "who are you"}
            ]
    res: Generator = model.chat_completions(messages=messages, stream=stream)
    if isinstance(res, Generator):
        for ret in res:
            if isinstance(ret, str) and ret.startswith("data: ") and ret.endswith("\n\n"):
                ret = ret[6:-2]
                ret = json.loads(ret)
            if isinstance(ret, dict) and "choices" in ret:
                x = ret["choices"][0]['delta']["content"]
                print(f'{x}', end="", flush=True)
            else:
                print(ret)
        print()
    else:
        if isinstance(res, dict) and "choices" in res:
            print(res["choices"][0]['message']["content"])
        else:
            print(res)