

from opencompass.models import OpenAI

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
], )

models = [
    dict(
        path='meta-llama/Llama-3.1-8B-Instruct',
        openai_api_base="http://localhost:8000/v1/chat/completions",
        key="",
        type=OpenAI,
        abbr="meta-llama/Llama-3.1-8B-Instruct",
        meta_template=api_meta_template,
        query_per_second=1,
        max_out_len=2048,
        max_seq_len=4096,
        batch_size=8,
        ),
]
