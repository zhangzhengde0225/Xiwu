from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.demo.demo_gsm8k_chat_gen import gsm8k_datasets
    from opencompass.configs.datasets.demo.demo_math_chat_gen import math_datasets
    from opencompass.configs.models.openai.gpt_4o_2024_05_13 import models as gpt4

from opencompass.models import OpenAI
from custom_model import MyModelAPI

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
], )

api_key = "sk-123qwe"

models = [
    dict(
        abbr="meta-llama/Llama-3.1-8B-Instruct",
        type=OpenAI,
        # type_class=MyModelAPI,
        path='meta-llama/Llama-3.1-8B-Instruct',
        key=api_key,  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
        meta_template=api_meta_template,
        query_per_second=1,
        max_out_len=2048,
        max_seq_len=4096,
        batch_size=8,
        openai_api_base="http://localhost:8000/v1/chat/completions",
        ),
]



datasets = gsm8k_datasets + math_datasets
models = models
# models = gpt4
