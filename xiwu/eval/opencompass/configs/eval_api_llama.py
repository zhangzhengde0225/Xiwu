from mmengine.config import read_base

with read_base():
    # from opencompass.configs.models.openai.gpt_4o_2024_05_13 import models as gpt4
    from .models.api.llama import models as llama_model

    # 英语综合
    from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_cot_gen_08c1de import mmlu_pro_datasets
    from opencompass.configs.datasets.IFEval.IFEval_gen import ifeval_datasets
    from opencompass.configs.datasets.gpqa.gpqa_gen import gpqa_datasets

    # 代码
    from opencompass.configs.datasets.humaneval.humaneval_gen import humaneval_datasets
    from opencompass.configs.datasets.livecodebench.livecodebench_gen import LCB_datasets
    from opencompass.configs.datasets.mbpp.mbpp_gen import mbpp_datasets
    from opencompass.configs.datasets.mbpp_plus.mbpp_plus_gen import mbpp_plus_datasets

    # 数学
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from opencompass.configs.datasets.math.math_gen import math_datasets
    from opencompass.configs.datasets.math.math_prm800k_500_gen import math_datasets as math_500_datasets

    # 中文
    from opencompass.configs.datasets.cmmlu.cmmlu_gen import cmmlu_datasets
    from opencompass.configs.datasets.GaokaoBench.GaokaoBench_gen import GaokaoBench_datasets
    from opencompass.configs.datasets.ceval.ceval_gen import ceval_datasets
    from opencompass.configs.datasets.CLUE_CMRC.CLUE_CMRC_gen import CMRC_datasets
    from opencompass.configs.datasets.mbpp_cn.mbpp_cn_gen import mbpp_cn_datasets
    
    # 多语言
    from opencompass.configs.datasets.mgsm.mgsm_gen import mgsm_datasets

    # 工具使用  TODO, Opencompass中没有这个数据集
    # from opencompass.configs.datasets



# datasets = gsm8k_datasets + math_datasets
datasets = mmlu_datasets + mmlu_pro_datasets + ifeval_datasets + gpqa_datasets + \
    humaneval_datasets + LCB_datasets + mbpp_datasets + mbpp_plus_datasets + \
    gsm8k_datasets + math_datasets + math_500_datasets + \
    cmmlu_datasets + GaokaoBench_datasets + ceval_datasets + CMRC_datasets + mbpp_cn_datasets + \
    mgsm_datasets

models = llama_model
