from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.demo.demo_gsm8k_chat_gen import gsm8k_datasets
    from opencompass.configs.datasets.demo.demo_math_chat_gen import math_datasets
    from opencompass.configs.models.openai.gpt_4o_2024_05_13 import models as gpt4

from opencompass.models import OpenAI

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
], )

api_key = "sk-123qwe"

models = [
    dict(
        path='meta-llama/Llama-3.1-8B-Instruct',  # Your model name
        openai_api_base="http://localhost:8000/v1/chat/completions",  # Your API base
        key="",  # Your API key
        type=OpenAI,
        abbr="meta-llama/Llama-3.1-8B-Instruct",  # Your model abbreviation
        meta_template=api_meta_template,
        query_per_second=1,
        max_out_len=2048,
        max_seq_len=4096,
        batch_size=8,
        ),
]



datasets = gsm8k_datasets + math_datasets
models = models
# models = gpt4
