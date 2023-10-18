"""
从Huggingface下载llama.
默认存在当前目录下文件夹中

需要先安装git lfs并初始化
 centos安装: `sudo yum install git-lfs`
 初始化：`git lfs install`
"""

import os, sys
import argparse
from pathlib import Path

here = Path(__file__).parent

allowed_names = [
    # "huggyllama/llama-7b",
    # "decapoda-research/llama-smallint-pt",
    # "decapoda-research/llama-7b-hf",
    # "decapoda-research/llama-7b-hf-int4",
    # "decapoda-research/llama-7b-hf-int8",
    # "decapoda-research/llama-13b-hf",
    # "decapoda-research/llama-13b-hf-int4",
    # "decapoda-research/llama-30b-hf",
    # "decapoda-research/llama-30b-hf-int4",
    # "decapoda-research/llama-65b-hf",
    # "decapoda-research/llama-65b-hf-int4",
    "lmsys/vicuna-7b-v1.5-16k",
]

def download(model_name, output_dir=None):
    # from transformers import AutoTokenizer,   m_pretrained(model_name)

    repo, model = model_name.split("/")
    command = f"git clone --progress https://huggingface.co/{repo}/{model}"
    if output_dir:
        assert os.path.exists(output_dir), f"{output_dir} not exists"
        os.chdir(output_dir)
    print(command)
    os.system(command)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--model_name", type=str, default="decapoda-research/llama-7b-hf")
    parser.add_argument("-s", "--size", type=str, default=None)
    parser.add_argument("-o", "--output-dir", type=str, default=f"{here}")
    args = parser.parse_args()

    # mn = "decapoda-research/llama-13b-hf"
    mn = args.model_name
    if args.size:
        mn = mn.replace("7b", args.size)
    assert mn in allowed_names, f"model_name must be one of {allowed_names}"
    
    download(model_name=mn, output_dir=args.output_dir)



